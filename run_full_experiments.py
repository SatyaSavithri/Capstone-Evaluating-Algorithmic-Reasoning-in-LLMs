# ===================================================================
# run_full_experiments.py
# Runs ALL experiments, computes ALL metrics, saves attention heatmaps
# ===================================================================

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from models_transformers import TransformersLLM
from GraphGenerator import generate_all_stimuli
from planner import bfs_optimal_path_to_max_reward
from hybrid_runner import run_hybrid
from attention_analysis import attention_to_room_ratio
from rsa_analysis import (
    build_theoretical_rsm, 
    compute_room_embeddings_from_hidden_states,
    rsm_from_embeddings,
    rsa_correlation
)


# ================================================================
#                 Utility Functions
# ================================================================

def extract_path_from_text(model_output: str):
    """
    Extracts a path like:
       Room 1 -> Room 2 -> Room 5
    or JSON like:
       {"path": ["Room 1","Room 2","Room 7"]}
    """
    import re

    # JSON format (scratchpad)
    json_match = re.search(r'\"path\"\s*:\s*\[(.*?)\]', model_output, re.DOTALL)
    if json_match:
        rooms = [x.strip().strip('"') for x in json_match.group(1).split(",")]
        return rooms

    # Arrow format
    arrow_match = re.findall(r'(Room\s*\d+)', model_output)
    if arrow_match:
        return arrow_match

    return []


def compute_edit_distance(a, b):
    """Levenshtein distance for path similarity."""
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1): dp[i][0] = i
    for j in range(len(b)+1): dp[0][j] = j
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


def save_attention_heatmap(attn_matrix, tokens, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ================================================================
#                   MAIN EXPERIMENT LOOP
# ================================================================

def run_all_experiments():

    # -----------------------------
    # Choose the model ONCE
    # -----------------------------
    model_id = "microsoft/phi-3-mini-4k-instruct"
    llm = TransformersLLM(model_id)

    # -----------------------------
    # Load stimuli from your generator
    # -----------------------------
    stimuli = generate_all_stimuli()

    # -----------------------------
    # Output csv
    # -----------------------------
    results_file = "experiment_results.csv"
    csv_fields = [
        "graph_type", "task_type", "method",
        "ground_truth", "predicted",
        "path_accuracy", "edit_distance", "reward_diff",
        "attention_ratio", "rsa_corr", "rsa_p",
        "heatmap_file"
    ]

    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

        # ===========================================
        # For each experiment (graph x task)
        # ===========================================
        for key, data in stimuli.items():
            graph_type = data["graph_type"]
            task_type  = data["task_type"]
            base_prompt = data["prompt"]
            scratch_prompt = data["scratchpad_prompt"]
            G = data["raw_graph_data"]
            gt_path = data["ground_truth_path"]

            # ====================================================
            # Method 1: Scratchpad Reasoning
            # ====================================================
            scratch_output = llm.generate(scratch_prompt, max_new_tokens=200)
            scratch_pred = extract_path_from_text(scratch_output)

            # ------------------------
            # Metrics
            # ------------------------
            acc = int(scratch_pred == gt_path)
            edit = compute_edit_distance(scratch_pred, gt_path)

            # reward diff
            rewards = nx.get_node_attributes(G, "reward")
            pred_reward = rewards.get(scratch_pred[-1], 0) if scratch_pred else 0
            gt_reward   = rewards.get(gt_path[-1], 0)
            reward_diff = gt_reward - pred_reward

            # attention
            act = llm.generate_with_activations(scratch_prompt)
            attn_matrix = act["attentions"][-1][0].mean(dim=0).detach().numpy()
            tokens = llm.tokenizer.tokenize(scratch_prompt)

            # find token positions for rooms
            positions = {
                room: [i for i,t in enumerate(tokens) if t.replace("Ġ","") == room.split()[1]]
                for room in G.nodes()
            }
            att_ratio = attention_to_room_ratio(attn_matrix, positions)

            # heatmap
            heatmap_file = f"heatmap_{key}_scratch.png"
            save_attention_heatmap(attn_matrix, tokens, heatmap_file)

            # RSA
            hidden_states = act["hidden_states"][-1][0].detach().numpy()
            room_embs = compute_room_embeddings_from_hidden_states(hidden_states, positions)
            empirical = rsm_from_embeddings(room_embs)
            theoretical = build_theoretical_rsm(G, list(G.nodes()))
            rsa_corr, rsa_p = rsa_correlation(empirical, theoretical)

            writer.writerow({
                "graph_type": graph_type,
                "task_type": task_type,
                "method": "scratchpad",
                "ground_truth": " -> ".join(gt_path),
                "predicted": " -> ".join(scratch_pred),
                "path_accuracy": acc,
                "edit_distance": edit,
                "reward_diff": reward_diff,
                "attention_ratio": att_ratio,
                "rsa_corr": rsa_corr,
                "rsa_p": rsa_p,
                "heatmap_file": heatmap_file
            })

            print(f"[✓] Scratchpad finished for {key}")

            # ====================================================
            # Method 2: Hybrid Reasoning
            # ====================================================
            hybrid_out = run_hybrid(llm, G)
            pred = hybrid_out["best"]

            # Convert "P1" into actual candidate path:
            if pred and pred.startswith("P"):
                idx = int(pred[1:]) - 1
                pred_path = hybrid_out["candidates"][idx]
            else:
                pred_path = []

            acc = int(pred_path == gt_path)
            edit = compute_edit_distance(pred_path, gt_path)
            pred_reward = rewards.get(pred_path[-1], 0) if pred_path else 0
            reward_diff = gt_reward - pred_reward

            writer.writerow({
                "graph_type": graph_type,
                "task_type": task_type,
                "method": "hybrid",
                "ground_truth": " -> ".join(gt_path),
                "predicted": " -> ".join(pred_path),
                "path_accuracy": acc,
                "edit_distance": edit,
                "reward_diff": reward_diff,
                "attention_ratio": None,
                "rsa_corr": None,
                "rsa_p": None,
                "heatmap_file": None
            })

            print(f"[✓] Hybrid finished for {key}")

    print("\n===============================================")
    print(" ALL EXPERIMENTS COMPLETE ")
    print(f" Results saved to: {results_file}")
    print("===============================================")


if __name__ == "__main__":
    run_all_experiments()
