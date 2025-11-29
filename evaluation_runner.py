# evaluation_runner.py

import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from graphs import create_line_graph, create_tree_graph, create_clustered_graph

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
    """
    Normalize a path:
    - Ensure it starts at the start_node (default first node)
    - Remove consecutive duplicates
    """
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
    """
    Reward regret = optimal_reward - sum of rewards along the path
    """
    path_reward = sum(rewards.get(node, 0) for node in path)
    return optimal_reward - path_reward

def compute_value_regret(path_values, optimal_values):
    """
    Computes per-node value regret
    """
    path_len = min(len(path_values), len(optimal_values))
    regret = [optimal_values[i] - path_values[i] for i in range(path_len)]
    return regret

def generate_attention_heatmap(attention_matrix, nodes, output_file):
    """
    Plot and save attention heatmap using Seaborn
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, xticklabels=nodes, yticklabels=nodes,
                annot=True, fmt=".2f", cmap="viridis")
    plt.title("Attention Heatmap")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

# -----------------------------
# Experiment Runner
# -----------------------------

def run_experiment(exp):
    # --- Build Graph ---
    G = exp["builder"]()
    nodes = list(G.nodes)
    rewards = nx.get_node_attributes(G, "reward")
    optimal_reward = max(rewards.values())
    start_node = nodes[0]

    # --- Path Generation (placeholder) ---
    # Replace this with your model's output path
    example_path = nodes  # Example: visiting all nodes sequentially
    normalized_path = normalize_path(example_path, start_node=start_node)

    # --- Metrics ---
    path_values = [rewards.get(node, 0) for node in normalized_path]
    reward_regret = compute_reward_regret(normalized_path, rewards, optimal_reward)
    value_regret = compute_value_regret(path_values, [optimal_reward]*len(normalized_path))

    # --- Print metrics ---
    print(f"\n--- Metrics for {exp['name']} ---")
    print(f"Reward Regret: {reward_regret}")
    print(f"Value Regret (per node): {value_regret}")
    print(f"Average Value Regret: {np.mean(value_regret):.2f}")
    print(f"Max Value Regret: {np.max(value_regret):.2f}")
    print(f"Path Length: {len(normalized_path)}\n")

    # --- Attention Heatmap (placeholder) ---
    attention_matrix = np.random.rand(len(nodes), len(nodes))  # replace with real attention
    heatmap_file = os.path.join(OUTPUT_DIR, f"{exp['name']}_attention.png")
    generate_attention_heatmap(attention_matrix, nodes, heatmap_file)

    # --- Save per-experiment CSV ---
    csv_file = os.path.join(OUTPUT_DIR, f"{exp['name']}_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "reward", "value_regret"])
        for node, val_reg in zip(normalized_path, value_regret):
            writer.writerow([node, rewards.get(node, 0), val_reg])
        writer.writerow([])
        writer.writerow(["reward_regret", reward_regret])

    return {
        "experiment": exp['name'],
        "reward_regret": reward_regret,
        "avg_value_regret": np.mean(value_regret),
        "max_value_regret": np.max(value_regret),
        "path_length": len(normalized_path)
    }

# -----------------------------
# Main Function
# -----------------------------

def main():
    summary_metrics = []
    print(f"Running {len(EXPERIMENTS)} experiments...\n")
    for exp in EXPERIMENTS:
        metrics = run_experiment(exp)
        summary_metrics.append(metrics)

    # --- Save summary CSV ---
    summary_file = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    with open(summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_metrics[0].keys())
        writer.writeheader()
        writer.writerows(summary_metrics)

    print(f"\nAll experiments completed. Summary metrics saved to {summary_file}")

if __name__ == "__main__":
    main()
