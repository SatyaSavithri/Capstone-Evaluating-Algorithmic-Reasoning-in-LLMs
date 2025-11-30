# evaluation_runner.py
"""
Patched evaluation runner — fixes for tensor detach/cpu conversion and hybrid return types.
Saves results under evaluation_results/<model>/<timestamp>/...
"""

import os
import time
import json
import csv
import logging
from pathlib import Path

import networkx as nx

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from hybrid_runner import run_hybrid
from scratchpad_runner import run_scratchpad
from models_transformers import TransformersLLM
import attention_analysis as att_analysis
import rsa_analysis as rsa_analysis
import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_runner")

RESULTS_ROOT = Path("evaluation_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


def extract_path_from_text(text):
    """Try JSON extraction first, then fallback to simple 'Room N' regex."""
    path = utils.parse_final_json_path(text)
    if path:
        return path
    import re
    tokens = re.findall(r'(Room\s*\d+)', text)
    return tokens


def compute_edit_distance(a, b):
    # reuse utils-like implementation
    if a is None: a = []
    if b is None: b = []
    lena = len(a); lenb = len(b)
    dp = [[0]*(lenb+1) for _ in range(lena+1)]
    for i in range(lena+1):
        dp[i][0] = i
    for j in range(lenb+1):
        dp[0][j] = j
    for i in range(1, lena+1):
        for j in range(1, lenb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[lena][lenb]


def path_accuracy(pred, gt):
    if pred is None or gt is None:
        return 0
    return int(list(pred) == list(gt))


def safe_save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def __find_room_token_positions(llm, prompt, rooms):
    """Find positions of room tokens in tokenized prompt. Works with HF tokenizers."""
    toks = llm.tokenizer.tokenize(prompt)
    positions = {}
    for r in rooms:
        parts = r.split()
        matches = []
        for i, t in enumerate(toks):
            if any(part.lower() in t.lower() for part in parts):
                matches.append(i)
        positions[r] = matches
    return positions, toks


def _to_numpy(x):
    """Utility: accept torch tensor or numpy array or list, return numpy array on CPU detached."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    # assume numpy compatible
    import numpy as np
    return np.asarray(x)


def main(model_id="microsoft/phi-3-mini-4k-instruct", device="cpu", max_new_tokens=150):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / model_id.replace("/", "_") / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {run_dir}")

    # load model (your models_transformers.TransformersLLM)
    logger.info(f"Loading model {model_id} on device={device} ...")
    llm = TransformersLLM(model_id=model_id, device=device)

    # graphs
    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph()
    }

    csv_path = run_dir / "experiment_metrics.csv"
    fields = [
        "run_time", "graph_key", "graph_type", "task_type", "method",
        "ground_truth", "predicted", "path_accuracy", "edit_distance",
        "gt_reward", "pred_reward", "reward_diff",
        "attention_ratio", "rsa_corr", "rsa_p",
        "heatmap_file", "rsm_file", "notes"
    ]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        for key, G in experiments.items():
            for task in ["valuePath", "rewardReval"]:
                graph_key = f"{key}_{task}"
                logger.info(f"Starting experiment: {graph_key}")

                # apply reward re-evaluation if requested
                if task == "rewardReval":
                    G_task = G.copy()
                    rewards = nx.get_node_attributes(G_task, "reward")
                    if len(rewards) >= 2:
                        sorted_nodes = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
                        best_node = sorted_nodes[0][0]
                        second_node = sorted_nodes[1][0]
                        new_rewards = rewards.copy()
                        new_rewards[second_node] = new_rewards[best_node] + 10
                        nx.set_node_attributes(G_task, new_rewards, "reward")
                else:
                    G_task = G

                start_node = list(G_task.nodes())[0]
                gt_path = bfs_optimal_path_to_max_reward(G_task, start_node)
                gt_reward = nx.get_node_attributes(G_task, "reward").get(gt_path[-1], 0) if gt_path else 0

                desc = base_description_text(G_task, start_node)
                sprompt = scratchpad_prompt(desc, task)

                # ---- Scratchpad method ----
                notes = ""
                try:
                    sp_out = llm.generate(sprompt, max_new_tokens=max_new_tokens, temperature=0.0)
                except Exception as e:
                    sp_out = ""
                    notes += f"generate_error:{e};"
                    logger.warning(f"Scratchpad generate failed for {graph_key}: {e}")

                pred_sp = extract_path_from_text(sp_out)
                acc_sp = path_accuracy(pred_sp, gt_path)
                edit_sp = compute_edit_distance(pred_sp, gt_path)
                pred_reward_sp = nx.get_node_attributes(G_task, "reward").get(pred_sp[-1], 0) if pred_sp else 0
                reward_diff_sp = gt_reward - pred_reward_sp

                # get activations + attentions (if supported)
                attention_ratio = None
                heatmap_file = None
                rsm_file = None
                rsa_corr = None
                rsa_p = None

                try:
                    activations = llm.generate_with_activations(sprompt, max_new_tokens=max_new_tokens)
                    hidden_states = activations.get("hidden_states", None)
                    attentions = activations.get("attentions", None)
                except Exception as e:
                    hidden_states = None
                    attentions = None
                    notes += f"activations_error:{e};"
                    logger.warning(f"generate_with_activations failed for {graph_key}: {e}")

                # attention: take last layer, batch 0, mean over heads -> (seq, seq)
                if attentions is not None:
                    try:
                        last_layer = attentions[-1]  # tensor or array shape (batch, heads, seq, seq)
                        last_layer_np = _to_numpy(last_layer)  # (batch, heads, seq, seq)
                        att_mat = last_layer_np[0].mean(axis=0)  # (seq, seq)
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        attention_ratio = att_analysis.attention_to_room_ratio(att_mat, positions)
                        heatmap_file = run_dir / f"{graph_key}_heatmap.png"
                        att_analysis.save_attention_heatmap_from_tensor(att_mat, tokens, str(heatmap_file))
                        heatmap_file = str(heatmap_file)
                    except Exception as e:
                        notes += f"attn_proc_error:{e};"
                        logger.warning(f"Attention processing failed for {graph_key}: {e}")

                # RSA: use last hidden layer and compute embeddings for rooms
                if hidden_states is not None:
                    try:
                        # hidden_states may be tuple/list of tensors (layers) with shape (batch, seq, hidden)
                        last_hidden = hidden_states[-1]
                        last_hidden_np = _to_numpy(last_hidden)  # (batch, seq, hidden)
                        # use batch 0
                        hs_seq = last_hidden_np[0]
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        room_embs = rsa_analysis.compute_room_embeddings_from_hidden_states(hs_seq, positions, method="mean")
                        empirical = rsa_analysis.rsm_from_embeddings(room_embs)
                        theoretical = rsa_analysis.build_theoretical_rsm(G_task, list(G_task.nodes()))
                        rsa_corr, rsa_p = rsa_analysis.rsa_correlation(empirical, theoretical)
                        rsm_file = run_dir / f"{graph_key}_rsm.png"
                        rsa_analysis.save_rsm_plot(empirical, str(rsm_file))
                        rsm_file = str(rsm_file)
                    except Exception as e:
                        notes += f"rsa_proc_error:{e};"
                        logger.warning(f"RSA processing failed for {graph_key}: {e}")

                # save scratchpad raw outputs
                safe_save_json({
                    "prompt": sprompt,
                    "scratchpad_text": sp_out,
                    "predicted_path": pred_sp,
                    "ground_truth": gt_path
                }, run_dir / f"{graph_key}_scratch_raw.json")

                writer.writerow({
                    "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "graph_key": graph_key,
                    "graph_type": key,
                    "task_type": task,
                    "method": "scratchpad",
                    "ground_truth": " -> ".join(gt_path),
                    "predicted": " -> ".join(pred_sp),
                    "path_accuracy": acc_sp,
                    "edit_distance": edit_sp,
                    "gt_reward": gt_reward,
                    "pred_reward": pred_reward_sp,
                    "reward_diff": reward_diff_sp,
                    "attention_ratio": attention_ratio,
                    "rsa_corr": rsa_corr,
                    "rsa_p": rsa_p,
                    "heatmap_file": heatmap_file,
                    "rsm_file": rsm_file,
                    "notes": notes
                })
                csvfile.flush()
                logger.info(f"Scratchpad finished for {graph_key}")

                # ---- Hybrid method ----
                notes_h = ""
                try:
                    hybrid_out = run_hybrid(llm, G_task, task=task, k=3, max_new_tokens=max_new_tokens)
                    candidates = hybrid_out.get("candidates", [])
                    best_label = hybrid_out.get("best", None)
                    validator_text = hybrid_out.get("validator_text", "")
                    pred_hybrid = []
                    if best_label and isinstance(best_label, str) and best_label.upper().startswith("P"):
                        idx = int(best_label[1:]) - 1
                        if 0 <= idx < len(candidates):
                            pred_hybrid = candidates[idx]
                except Exception as e:
                    pred_hybrid = []
                    notes_h += f"hybrid_error:{e};"
                    logger.warning(f"Hybrid runner failed for {graph_key}: {e}")

                acc_h = path_accuracy(pred_hybrid, gt_path)
                edit_h = compute_edit_distance(pred_hybrid, gt_path)
                pred_reward_h = nx.get_node_attributes(G_task, "reward").get(pred_hybrid[-1], 0) if pred_hybrid else 0
                reward_diff_h = gt_reward - pred_reward_h

                safe_save_json({
                    "candidates": candidates,
                    "best": hybrid_out.get("best") if 'hybrid_out' in locals() else None,
                    "validator_text": hybrid_out.get("validator_text") if 'hybrid_out' in locals() else None
                }, run_dir / f"{graph_key}_hybrid_raw.json")

                writer.writerow({
                    "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "graph_key": graph_key,
                    "graph_type": key,
                    "task_type": task,
                    "method": "hybrid",
                    "ground_truth": " -> ".join(gt_path),
                    "predicted": " -> ".join(pred_hybrid),
                    "path_accuracy": acc_h,
                    "edit_distance": edit_h,
                    "gt_reward": gt_reward,
                    "pred_reward": pred_reward_h,
                    "reward_diff": reward_diff_h,
                    "attention_ratio": None,
                    "rsa_corr": None,
                    "rsa_p": None,
                    "heatmap_file": None,
                    "rsm_file": None,
                    "notes": notes_h
                })
                csvfile.flush()
                logger.info(f"Hybrid finished for {graph_key}")

    logger.info(f"All experiments complete. Results saved under: {run_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()
    main(model_id=args.model, device=args.device, max_new_tokens=args.max_tokens)
