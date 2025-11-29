# evaluation_runner.py

import os
import csv
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from attention_analysis import attention_to_room_ratio
from rsa_analysis import build_theoretical_rsm, compute_room_embeddings_from_hidden_states, rsm_from_embeddings, rsa_correlation
from utils import parse_final_json_path
from prompts import base_description_text, scratchpad_prompt

# Import only the run_hybrid function from hybrid_runner (safe at top-level)
from hybrid_runner import run_hybrid

# -----------------------------
# Configuration
# -----------------------------
EXPERIMENTS = [
    {"name": "line_graph", "builder": create_line_graph},
    {"name": "tree_graph", "builder": create_tree_graph},
    {"name": "clustered_graph", "builder": create_clustered_graph},
]

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Utility Functions
# -----------------------------
def normalize_path(path, start_node=None):
    if not path:
        return []
    if start_node and path[0] != start_node:
        path = [start_node] + path
    normalized = [path[0]]
    for node in path[1:]:
        if node != normalized[-1]:
            normalized.append(node)
    return normalized

def compute_reward_regret(path, rewards, optimal_reward):
    path_reward = sum(rewards.get(node, 0) for node in path)
    return optimal_reward - path_reward

def compute_value_regret(path_values, optimal_values):
    path_len = min(len(path_values), len(optimal_values))
    return [optimal_values[i] - path_values[i] for i in range(path_len)]

def sequence_edit_distance(path1, path2):
    n, m = len(path1), len(path2)
    dp = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1):
        dp[i,0] = i
    for j in range(m+1):
        dp[0,j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if path1[i-1]==path2[j-1]:
                dp[i,j] = dp[i-1,j-1]
            else:
                dp[i,j] = 1 + min(dp[i-1,j-1], dp[i-1,j], dp[i,j-1])
    return dp[n,m]

def generate_attention_heatmap(attention_matrix, nodes, output_file):
    plt.figure(figsize=(8,6))
    sns.heatmap(attention_matrix, xticklabels=nodes, yticklabels=nodes,
                annot=False, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def generate_rsm_heatmap(rsm, nodes, output_file, title="RSM"):
    plt.figure(figsize=(6,5))
    sns.heatmap(rsm, xticklabels=nodes, yticklabels=nodes, cmap="viridis", annot=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment(exp, model_wrapper):
    # Build graph
    G = exp["builder"]()
    nodes = list(G.nodes)
    rewards = nx.get_node_attributes(G, "reward")
    optimal_reward = max(rewards.values())
    start_node = nodes[0]

    # Symbolic ground-truth path
    gt_path = bfs_optimal_path_to_max_reward(G, start_node)
    print(f"\n[INFO] Ground-truth BFS path: {gt_path}")

    # Generate LLM path via Hybrid
    hybrid_out = run_hybrid(model_wrapper, G)
    llm_path = parse_final_json_path(hybrid_out["best"] or " -> ".join(nodes))
    llm_path = normalize_path(llm_path, start_node=start_node)
    print(f"[INFO] LLM generated path: {llm_path}")

    # ---------------------------
    # Behavioral metrics
    # ---------------------------
    reward_regret = compute_reward_regret(llm_path, rewards, optimal_reward)
    path_values = [rewards.get(node,0) for node in llm_path]
    value_regret = compute_value_regret(path_values, [optimal_reward]*len(llm_path))

    traversal_accuracy = 1.0 if llm_path==gt_path else 0.0
    seq_edit_distance = sequence_edit_distance(llm_path, gt_path)

    # ---------------------------
    # Internal metrics
    # ---------------------------
    activations = model_wrapper.generate_with_activations(
        scratchpad_prompt(base_description_text(G), "valuePath")
    )
    hidden_states = torch.cat([h.detach() for h in activations["hidden_states"]], dim=0).cpu().numpy()
    positions_map = {n:[i for i in range(hidden_states.shape[0])] for n in nodes}  # Placeholder mapping
    embs = compute_room_embeddings_from_hidden_states(hidden_states, positions_map)
    emp_rsm = rsm_from_embeddings(embs)
    theo_rsm = build_theoretical_rsm(G, nodes)
    rsa_r, rsa_p = rsa_correlation(emp_rsm, theo_rsm)

    attn_ratio = None
    attentions = activations["attentions"]
    if len(attentions)>0:
        attn_ratio = attention_to_room_ratio(attentions[-1][0].cpu().numpy(), positions_map)

    # ---------------------------
    # Save per-experiment CSV
    # ---------------------------
    csv_file = os.path.join(OUTPUT_DIR, f"{exp['name']}_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "reward", "value_regret"])
        for node, val_reg in zip(llm_path, value_regret):
            writer.writerow([node, rewards.get(node,0), val_reg])
        writer.writerow([])
        writer.writerow(["reward_regret", reward_regret])
        writer.writerow(["traversal_accuracy", traversal_accuracy])
        writer.writerow(["sequence_edit_distance", seq_edit_distance])
        writer.writerow(["rsa_correlation", rsa_r])
        writer.writerow(["attention_ratio_analysis", attn_ratio])

    # ---------------------------
    # Heatmaps
    # ---------------------------
    if len(attentions)>0:
        generate_attention_heatmap(attentions[-1][0].cpu().numpy(), nodes,
                                   os.path.join(OUTPUT_DIR, f"{exp['name']}_attention.png"))
    generate_rsm_heatmap(emp_rsm, nodes, os.path.join(OUTPUT_DIR, f"{exp['name']}_rsm.png"))

    # ---------------------------
    # Summary metrics
    # ---------------------------
    return {
        "experiment": exp['name'],
        "reward_regret": reward_regret,
        "avg_value_regret": np.mean(value_regret),
        "max_value_regret": np.max(value_regret),
        "traversal_accuracy": traversal_accuracy,
        "sequence_edit_distance": seq_edit_distance,
        "rsa_correlation": rsa_r,
        "attention_ratio_analysis": attn_ratio,
        "path_length": len(llm_path)
    }

# -----------------------------
# Main
# -----------------------------
def main():
    # Import TransformersLLM inside main to avoid ImportError
    from hybrid_runner import TransformersLLM

    # Load model
    model_id = "microsoft/phi-3-mini-4k-instruct"  # Change if desired
    model_wrapper = TransformersLLM(model_id)

    summary_metrics = []
    for exp in EXPERIMENTS:
        metrics = run_experiment(exp, model_wrapper)
        summary_metrics.append(metrics)

    # Save summary CSV
    summary_file = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_metrics[0].keys())
        writer.writeheader()
        writer.writerows(summary_metrics)

    print(f"\nAll experiments completed. Summary metrics saved to {summary_file}")

if __name__ == "__main__":
    main()
