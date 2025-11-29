# evaluation_runner.py
"""
Evaluation runner (final) — integrated pipeline using existing repo modules.
Saves results, attention heatmaps and RSA metrics to evaluation_results/.
"""

import os
import time
import json
import csv
from pathlib import Path
import logging

import networkx as nx

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from hybrid_runner import run_hybrid
from scratchpad_runner import run_scratchpad
from hybrid_runner_eval import TransformersLLM
import attention_analysis as att_analysis
import rsa_analysis as rsa_analysis
import utils

# Output root
RESULTS_ROOT = Path("evaluation_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_runner")


def extract_path_from_text(text):
    """
    Simple extractor: finds tokens like 'Room 1' in the text in order.
    Falls back to utils.parse_final_json_path if present.
    """
    path = utils.parse_final_json_path(text)
    if path:
        return path
    import re
    tokens = re.findall(r'(Room\s*\d+)', text)
    return tokens


def compute_edit_distance(a, b):
    # small wrapper to reuse existing utils if present
    return utils.parse_final_json_path and __compute_edit_distance_local(a, b) or __compute_edit_distance_local(a, b)


def __compute_edit_distance_local(a, b):
    a = a or []
    b = b or []
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
    if (pred is None) or (gt is None):
        return 0
    return int(list(pred) == list(gt))


def safe_save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main(model_id="microsoft/phi-3-mini-4k-instruct", device="cpu", max_new_tokens=150):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / model_id.replace("/", "_") / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {run_dir}")

    # Instantiate model wrapper
    llm = TransformersLLM(model_id, device=device)

    # Build experiment stimuli using available graph generators (consistent with your repo)
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

        # iterate experiments
        for key, G in experiments.items():
            for task in ["valuePath", "rewardReval"]:
                graph_key = f"{key}_{task}"
                logger.info(f"Starting: {graph_key}")

                # For rewardReval we adapt the rewards in a copy (same behavior as earlier generator)
                if task == "rewardReval":
                    G_task = G.copy()
                    rewards = nx.get_node_attributes(G_task, "reward")
                    if len(rewards) >= 2:
                        # bump the second-best
                        sorted_nodes = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
                        if len(sorted_nodes) > 1:
                            best_node = sorted_nodes[0][0]
                            second_node = sorted_nodes[1][0]
                            new_rewards = rewards.copy()
                            new_rewards[second_node] = new_rewards[best_node] + 10
                            nx.set_node_attributes(G_task, new_rewards, "reward")
                    else:
                        G_task = G.copy()
                else:
                    G_task = G

                start_node = list(G_task.nodes())[0]
                gt_path = bfs_optimal_path_to_max_reward(G_task, start_node)
                gt_reward = nx.get_node_attributes(G_task, "reward").get(gt_path[-1], 0) if gt_path else 0

                # Build prompts (use your prompts.py)
                desc = base_description_text(G_task, start_node)
                sprompt = scratchpad_prompt(desc, task if task != "rewardReval" else "rewardReval")

                # -------------------------
                # Method A: Scratchpad
                # -------------------------
                notes = ""
                try:
                    sp_out = llm.generate(sprompt, max_new_tokens=max_new_tokens, temperature=0.0)
                except Exception as e:
                    sp_out = ""
                    notes += f"scratchpad_error:{e};"
                    logger.warning(f"Scratchpad generation failed: {e}")

                pred_sp = extract_path_from_text(sp_out)
                acc_sp = path_accuracy(pred_sp, gt_path)
                edit_sp = __compute_edit_distance_local(pred_sp, gt_path)
                pred_reward_sp = nx.get_node_attributes(G_task, "reward").get(pred_sp[-1], 0) if pred_sp else 0
                reward_diff_sp = gt_reward - pred_reward_sp

                # get activations/attentions for further analyses
                try:
                    activations_dict = llm.generate_with_activations(sprompt, max_new_tokens=max_new_tokens)
                    hidden_states = activations_dict.get("hidden_states", None)
                    attentions = activations_dict.get("attentions", None)
                except Exception as e:
                    hidden_states = None
                    attentions = None
                    notes += f"activations_error:{e};"
                    logger.warning(f"generate_with_activations failed: {e}")

                attention_ratio = None
                heatmap_file = None
                rsm_file = None
                rsa_corr = None
                rsa_p = None

                # attention analysis
                if attentions:
                    try:
                        # attentions: list of tensors (layers) shaped (batch, heads, seq, seq)
                        last_layer = attentions[-1]
                        att_mat = last_layer[0].mean(axis=0).cpu().numpy()
                        # find token positions for rooms using tokenizer
                        from models_transformers import AutoTokenizer  # fallback if needed
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        attention_ratio = att_analysis.attention_to_room_ratio(att_mat, positions)
                        heatmap_file = run_dir / f"{graph_key}_scratch_heatmap.png"
                        att_analysis.save_attention_heatmap_from_tensor(att_mat, tokens, str(heatmap_file))
                        heatmap_file = str(heatmap_file)
                    except Exception as e:
                        notes += f"attn_process_error:{e};"
                        logger.warning(f"Attention processing failed: {e}")

                # RSA
                if hidden_states is not None:
                    try:
                        # hidden_states can be a tuple/list of arrays/lazy tensors
                        # take last layer -> shape (batch, seq, hidden)
                        hs_last = hidden_states[-1] if isinstance(hidden_states, (list, tuple)) else hidden_states
                        hs_np = hs_last[0].cpu().numpy() if hasattr(hs_last, "cpu") else hs_last[0]
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        room_embs = rsa_analysis.compute_room_embeddings_from_hidden_states(hs_np, positions, method="mean")
                        empirical = rsa_analysis.rsm_from_embeddings(room_embs)
                        theoretical = rsa_analysis.build_theoretical_rsm(G_task, list(G_task.nodes()))
                        rsa_corr, rsa_p = rsa_analysis.rsa_correlation(empirical, theoretical)
                        rsm_file = run_dir / f"{graph_key}_empirical_rsm.png"
                        rsa_analysis.save_rsm_plot(empirical, str(rsm_file))
                        rsm_file = str(rsm_file)
                    except Exception as e:
                        notes += f"rsa_error:{e};"
                        logger.warning(f"RSA processing failed: {e}")

                # Save raw outputs
                raw_out = {
                    "scratchpad_output": sp_out,
                    "pred_path": pred_sp,
                    "ground_truth": gt_path,
                }
                safe_save_json(raw_out, run_dir / f"{graph_key}_scratch_raw.json")

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
                    "heatmap_file": heatm_
