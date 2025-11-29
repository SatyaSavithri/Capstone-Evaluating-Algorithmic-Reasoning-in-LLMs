# evaluation_runner.py
import os
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from graphs import create_line_graph, create_tree_graph, create_clustered_graph, NODE_PREFIX

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

    # --- Attention Heatmap (placeholder) ---
    # Replace with real attention weights from your model
    attention_matrix = np.random.rand(len(nodes), len(nodes))
    heatmap_file = os.path.join(OUTPUT_DIR, f"{exp['name']}_attention.png")
    generate_attention_heatmap(attention_matrix, nodes, heatmap_file)

    # --- Save Results to CSV ---
    csv_file = os.path.join(OUTPUT_DIR, f"{exp['name']}_results.csv")
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "reward", "value_regret"])
        for node, val_reg in zip(normalized_path, value_regret):
            writer.writerow([node, rewards.get(node, 0), val_reg])
        writer.writerow([])
        writer.writerow(["reward_regret", reward_regret])

    print(f"[✔] {exp['name']} completed. Results saved to {csv_file} and {heatmap_file}")

def main():
    print(f"Running {len(EXPERIMENTS)} experiments...\n")
    for exp in EXPERIMENTS:
        run_experiment(exp)
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
