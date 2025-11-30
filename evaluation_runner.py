"""
Multi-model evaluation runner — runs Phi + Gemma models, computes scratchpad metrics,
hybrid metrics, attention heatmaps, and RSA similarity for each model separately.

Each model gets:
evaluation_results/<model_name>/<timestamp>/experiment_metrics.csv

This file keeps your full existing logic (Option A) and simply wraps it in a
two-model loop without breaking anything.
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


# ============================================================
# ---------------------- UTILITIES ---------------------------
# ============================================================

def extract_path_from_text(text):
    """Try JSON extraction first, fallback to simple regex."""
    path = utils.parse_final_json_path(text)
    if path:
        return path
    import re
    tokens = re.findall(r'(Room\s*\d+)', text)
    return tokens


def compute_edit_distance(a, b):
    if a is None: a = []
    if b is None: b = []
    lena, lenb = len(a), len(b)
    dp = [[0]*(lenb+1) for _ in range(lena+1)]
    for i in range(lena+1): dp[i][0] = i
    for j in range(lenb+1): dp[0][j] = j
    for i in range(1, lena+1):
        for j in range(1, lenb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[lena][lenb]


def path_accuracy(pred, gt):
    if pred is None or gt is None:
        return 0
    return int(list(pred) == list(gt))


def safe_save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _to_numpy(x):
    """Convert torch → numpy safe."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    import numpy as np
    return np.asarray(x)


def __find_room_token_positions(llm, prompt, rooms):
    toks = llm.tokenizer.tokenize(prompt)
    positions = {}
    for r in rooms:
        parts = r.split()
        matches = []
        for i, t in enumerate(toks):
            if any(p.lower() in t.lower() for p in parts):
                matches.append(i)
        positions[r] = matches
    return positions, toks


# ============================================================
# ------------------ RUN EXPERIMENT SET ----------------------
# ============================================================

def run_all_experiments_for_model(model_id, device="cpu", max_new_tokens=150):
    """
    Runs your entire scratchpad + hybrid evaluation
    EXACTLY as before, but for ONE model.

    This function is called twice: one for Phi, one for Gemma.
    """

    # ------------------------------------------------------------
    # Setup result directory for this model
    # ------------------------------------------------------------
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / model_id.replace("/", "_") / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info(f"Running evaluation for model: {model_id}")
    logger.info(f"Results directory: {run_dir}")
    logger.info("="*80)

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    logger.info(f"Loading model {model_id} ...")
    llm = TransformersLLM(model_id=model_id, device=device)

    # ------------------------------------------------------------
    # Graph sets
    # ------------------------------------------------------------
    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph()
    }

    # ------------------------------------------------------------
    # CSV writer
    # ------------------------------------------------------------
    csv_path = run_dir / "experiment_metrics.csv"
    fields = [
        "run_time", "graph_key", "graph_type", "task_type", "method",
        "ground_truth", "predicted", "path_accuracy", "edit_distance",
        "gt_reward", "pred_reward", "reward_diff",
        "attention_ratio", "rsa_corr", "rsa_p",
        "heatmap_file", "rsm_file", "notes"
    ]

    # ------------------------------------------------------------
    # Begin writing CSV
    # ------------------------------------------------------------
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()

        # ============================================================
        # LOOP: Each graph × Each task
        # ============================================================
        for key, G in experiments.items():
            for task in ["valuePath", "rewardReval"]:

                graph_key = f"{key}_{task}"
                logger.info(f"Starting experiment: {graph_key}")

                # ------------------------------------------------------------
                # Reward Reval adjustment
                # ------------------------------------------------------------
                if task == "rewardReval":
                    G_task = G.copy()
                    rewards = nx.get_node_attributes(G_task, "reward")
                    if len(rewards) >= 2:
                        sorted_nodes = sorted(rewards.items(), key=lambda x: x[1], reverse=True)
                        best_node = sorted_nodes[0][0]
                        second_node = sorted_nodes[1][0]
                        new_r = rewards.copy()
                        new_r[second_node] = new_r[best_node] + 10
                        nx.set_node_attributes(G_task, new_r, "reward")
                else:
                    G_task = G

                # ------------------------------------------------------------
                # Compute ground truth path
                # ------------------------------------------------------------
                start = list(G_task.nodes())[0]
                gt_path = bfs_optimal_path_to_max_reward(G_task, start)
                gt_reward = nx.get_node_attributes(G_task, "reward").get(gt_path[-1], 0) if gt_path else 0

                # ------------------------------------------------------------
                # Build prompt
                # ------------------------------------------------------------
                desc = base_description_text(G_task, start)
                sprompt = scratchpad_prompt(desc, task)

                # ============================================================
                # -------------------- SCRATCHPAD -----------------------------
                # ============================================================
                notes = ""
                try:
                    sp_out = llm.generate(sprompt, max_new_tokens=max_new_tokens, temperature=0.0)
                except Exception as e:
                    sp_out = ""
                    notes += f"generate_error:{e};"
                    logger.warning(f"Scratchpad generation error for {graph_key}: {e}")

                pred_sp = extract_path_from_text(sp_out)
                acc_sp = path_accuracy(pred_sp, gt_path)
                edit_sp = compute_edit_distance(pred_sp, gt_path)
                pred_reward_sp = nx.get_node_attributes(G_task,"reward").get(pred_sp[-1],0) if pred_sp else 0
                reward_diff_sp = gt_reward - pred_reward_sp

                # ---------------- Attention + RSA ----------------
                attention_ratio = None
                rsa_corr = None
                rsa_p = None
                heatmap_file = None
                rsm_file = None

                try:
                    acts = llm.generate_with_activations(sprompt, max_new_tokens=max_new_tokens)
                    hidden_states = acts.get("hidden_states")
                    attentions = acts.get("attentions")
                except Exception as e:
                    hidden_states, attentions = None, None
                    notes += f"activ_error:{e};"

                # Attention
                if attentions is not None:
                    try:
                        last_layer = _to_numpy(attentions[-1])  # (batch, heads, seq, seq)
                        att_mat = last_layer[0].mean(axis=0)
                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))

                        attention_ratio = att_analysis.attention_to_room_ratio(att_mat, positions)

                        heatmap_file = run_dir / f"{graph_key}_heatmap.png"
                        att_analysis.save_attention_heatmap_from_tensor(att_mat, tokens, str(heatmap_file))
                        heatmap_file = str(heatmap_file)
                    except Exception as e:
                        notes += f"attn_error:{e};"

                # RSA
                if hidden_states is not None:
                    try:
                        last_hidden = _to_numpy(hidden_states[-1])[0]  # (seq, hidden)
                        positions, _ = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        room_embs = rsa_analysis.compute_room_embeddings_from_hidden_states(
                            last_hidden, positions, method="mean"
                        )
                        empirical = rsa_analysis.rsm_from_embeddings(room_embs)
                        theoretical = rsa_analysis.build_theoretical_rsm(G_task, list(G_task.nodes()))
                        rsa_corr, rsa_p = rsa_analysis.rsa_correlation(empirical, theoretical)

                        rsm_file = run_dir / f"{graph_key}_rsm.png"
                        rsa_analysis.save_rsm_plot(empirical, str(rsm_file))
                        rsm_file = str(rsm_file)
                    except Exception as e:
                        notes += f"rsa_error:{e};"

                # Save raw scratchpad output
                safe_save_json({
                    "prompt": sprompt,
                    "scratchpad_text": sp_out,
                    "predicted_path": pred_sp,
                    "ground_truth": gt_path,
                }, run_dir / f"{graph_key}_scratch_raw.json")

                # Write scratchpad row
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
                logger.info(f"Scratchpad done: {graph_key}")

                # ============================================================
                # ----------------------- HYBRID -----------------------------
                # ============================================================
                notes_h = ""
                candidates = []
                pred_hybrid = []

                try:
                    hybrid_out = run_hybrid(llm, G_task, task=task, k=3, max_new_tokens=max_new_tokens)
                    candidates = hybrid_out.get("candidates", [])
                    best = hybrid_out.get("best")

                    if best and best.startswith("P"):
                        idx = int(best[1:]) - 1
                        if 0 <= idx < len(candidates):
                            pred_hybrid = candidates[idx]

                except Exception as e:
                    pred_hybrid = []
                    notes_h += f"hybrid_error:{e};"

                acc_h = path_accuracy(pred_hybrid, gt_path)
                edit_h = compute_edit_distance(pred_hybrid, gt_path)
                pred_reward_h = nx.get_node_attributes(G_task,"reward").get(pred_hybrid[-1],0) if pred_hybrid else 0
                reward_diff_h = gt_reward - pred_reward_h

                # save hybrid raw
                safe_save_json({
                    "candidates": candidates,
                    "best": hybrid_out.get("best") if 'hybrid_out' in locals() else None,
                    "validator_text": hybrid_out.get("validator_text") if 'hybrid_out' in locals() else None
                }, run_dir / f"{graph_key}_hybrid_raw.json")

                # Write hybrid row
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
                logger.info(f"Hybrid done: {graph_key}")

    logger.info(f"Model complete. Results saved in {run_dir}")


# ============================================================
# ---------------------- ENTRYPOINT --------------------------
# ============================================================

if __name__ == "__main__":
    # Run for BOTH models automatically
    model_list = [
        "microsoft/phi-3-mini-4k-instruct",
        "google/gemma-2-9b-it"
    ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    for model_id in model_list:
        run_all_experiments_for_model(model_id, device=args.device, max_new_tokens=args.max_tokens)
