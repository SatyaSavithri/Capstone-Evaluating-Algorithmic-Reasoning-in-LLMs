# evaluation_runner.py
"""
Evaluation runner (final) — integrated pipeline using your existing repo modules.
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
from hybrid_runner import run_hybrid        # your hybrid runner
from scratchpad_runner import run_scratchpad
from models_transformers import TransformersLLM   # <<<<<< HERE — USE YOUR OWN CLASS
import attention_analysis as att_analysis
import rsa_analysis as rsa_analysis
import utils


# Output root
RESULTS_ROOT = Path("evaluation_results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("evaluation_runner")


# -------------------------------------------------------
# Utility helpers
# -------------------------------------------------------

def extract_path_from_text(text):
    """Extract path either from JSON or arrow-separated string."""
    path = utils.parse_final_json_path(text)
    if path:
        return path

    import re
    tokens = re.findall(r'(Room\s*\d+)', text)
    return tokens


def compute_edit_distance(a, b):
    return __compute_edit_distance_local(a, b)


def __compute_edit_distance_local(a, b):
    a = a or []
    b = b or []
    lena = len(a); lenb = len(b)
    dp = [[0] * (lenb + 1) for _ in range(lena + 1)]
    for i in range(lena + 1):
        dp[i][0] = i
    for j in range(lenb + 1):
        dp[0][j] = j
    for i in range(1, lena + 1):
        for j in range(1, lenb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[lena][lenb]


def path_accuracy(pred, gt):
    return int(list(pred) == list(gt))


def safe_save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# -------------------------------------------------------
#              MAIN PIPELINE
# -------------------------------------------------------

def main(model_id="microsoft/phi-3-mini-4k-instruct", device="cpu", max_new_tokens=150):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / model_id.replace("/", "_") / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results directory: {run_dir}")

    # -------------------------------------------------------
    # Instantiate model from YOUR models_transformers.py
    # -------------------------------------------------------
    llm = TransformersLLM(model_id=model_id, device=device)

    # -------------------------------------------------------
    # Load your graphs
    # -------------------------------------------------------
    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph()
    }

    # -------------------------------------------------------
    # CSV output
    # -------------------------------------------------------
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

        # ===============================================================
        # LOOP: For each (graph x task)
        # ===============================================================
        for key, G in experiments.items():
            for task in ["valuePath", "rewardReval"]:

                graph_key = f"{key}_{task}"
                logger.info(f"Running {graph_key}")

                # -------------------------------------------------------
                # adjust reward for rewardReval
                # -------------------------------------------------------
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
                gt_reward = nx.get_node_attributes(G_task, "reward").get(gt_path[-1], 0)

                # build prompt
                desc = base_description_text(G_task, start_node)
                sprompt = scratchpad_prompt(desc, task)

                # -------------------------------------------------------
                # Method A: Scratchpad Reasoning
                # -------------------------------------------------------
                notes = ""
                try:
                    sp_out = llm.generate(sprompt, max_new_tokens=max_new_tokens, temperature=0.0)
                except Exception as e:
                    sp_out = ""
                    notes += f"scratch_error:{e};"

                pred_sp = extract_path_from_text(sp_out)
                acc_sp = path_accuracy(pred_sp, gt_path)
                edit_sp = compute_edit_distance(pred_sp, gt_path)
                pred_reward_sp = nx.get_node_attributes(G_task, "reward").get(pred_sp[-1], 0) if pred_sp else 0
                reward_diff_sp = gt_reward - pred_reward_sp

                # ACTIVATIONS / ATTENTION / RSA
                attention_ratio = None
                heatmap_file = None
                rsm_file = None
                rsa_corr = None
                rsa_p = None

                try:
                    activations = llm.generate_with_activations(sprompt)
                    hidden_states = activations["hidden_states"]
                    attentions = activations["attentions"]
                except Exception as e:
                    hidden_states, attentions = None, None
                    notes += f"act_error:{e};"

                # Attention heatmap + ratio
                if attentions is not None:
                    try:
                        last_layer = attentions[-1]
                        att_mat = last_layer[0].mean(axis=0).detach().cpu().numpy()

                        positions, tokens = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        attention_ratio = att_analysis.attention_to_room_ratio(att_mat, positions)

                        heatmap_file = run_dir / f"{graph_key}_heatmap.png"
                        att_analysis.save_attention_heatmap_from_tensor(att_mat, tokens, str(heatmap_file))
                        heatmap_file = str(heatmap_file)
                    except Exception as e:
                        notes += f"heatmap_error:{e};"

                # RSA
                if hidden_states is not None:
                    try:
                        hs_last = hidden_states[-1][0].detach().cpu().numpy()   # (seq, hidden)
                        positions, _ = __find_room_token_positions(llm, sprompt, list(G_task.nodes()))
                        room_embs = rsa_analysis.compute_room_embeddings_from_hidden_states(hs_last, positions)
                        empirical = rsa_analysis.rsm_from_embeddings(room_embs)
                        theoretical = rsa_analysis.build_theoretical_rsm(G_task, list(G_task.nodes()))
                        rsa_corr, rsa_p = rsa_analysis.rsa_correlation(empirical, theoretical)

                        rsm_file = run_dir / f"{graph_key}_rsm.png"
                        rsa_analysis.save_rsm_plot(empirical, str(rsm_file))
                        rsm_file = str(rsm_file)
                    except Exception as e:
                        notes += f"rsa_error:{e};"

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
                csvfile.flush()

                # -------------------------------------------------------
                # Method B: Hybrid Reasoning
                # -------------------------------------------------------
                notes_h = ""
                try:
                    hybrid_out = run_hybrid(llm, G_task)
                    candidates = hybrid_out.get("candidates", [])
                    best = hybrid_out.get("best", None)

                    if best and best.startswith("P"):
                        idx = int(best[1:]) - 1
                        pred_hybrid = candidates[idx] if idx < len(candidates) else []
                    else:
                        pred_hybrid = []
                except Exception as e:
                    pred_hybrid = []
                    notes_h += f"hybrid_error:{e};"

                acc_h = path_accuracy(pred_hybrid, gt_path)
                edit_h = compute_edit_distance(pred_hybrid, gt_path)
                pred_reward_h = nx.get_node_attributes(G_task, "reward").get(pred_hybrid[-1], 0) if pred_hybrid else 0
                reward_diff_h = gt_reward - pred_reward_h

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

                logger.info(f"Completed: {graph_key}")

    logger.info(f"All experiments complete. Results saved in {run_dir}")


# -------------------------------------------------------
# Room token position finder
# -------------------------------------------------------
def __find_room_token_positions(llm, prompt, rooms):
    """Find positions of tokens matching room names."""
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_tokens", type=int, default=150)
    args = parser.parse_args()

    main(
        model_id=args.model,
        device=args.device,
        max_new_tokens=args.max_tokens
    )
