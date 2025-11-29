import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from graphs import (
    create_line_graph,
    create_grid_graph,
    create_random_graph,
    create_tree_graph,
    create_fully_connected_graph
)

from task_generator import generate_task
from model_reasoning import run_model_reasoning
from evaluation_metrics import (
    compute_exact_match,
    compute_edit_distance,
    compute_path_length_difference,
    compute_reward_accuracy,
)
from attention_analysis import extract_attention_scores


# --------------------------------------------------------
#                CONFIGURATION
# --------------------------------------------------------

EXPERIMENTS = [
    {
        "name": "line_graph_experiment",
        "graph_fn": create_line_graph,
        "params": {"n_nodes": 7},
        "num_tasks": 20,
    },
    {
        "name": "grid_graph_experiment",
        "graph_fn": create_grid_graph,
        "params": {"rows": 3, "cols": 3},
        "num_tasks": 20,
    },
    {
        "name": "random_graph_experiment",
        "graph_fn": create_random_graph,
        "params": {"n_nodes": 10, "edge_prob": 0.25},
        "num_tasks": 20,
    },
]

OUTPUT_DIR = "evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
HEATMAP_DIR = os.path.join(OUTPUT_DIR, "attention_heatmaps")
os.makedirs(HEATMAP_DIR, exist_ok=True)


# --------------------------------------------------------
#              RUN EXPERIMENT
# --------------------------------------------------------

def run_single_experiment(exp_cfg):
    exp_name = exp_cfg["name"]
    graph_fn = exp_cfg["graph_fn"]
    graph_params = exp_cfg["params"]
    num_tasks = exp_cfg["num_tasks"]

    print(f"\nRunning experiment: {exp_name}")

    # Create graph
    G = graph_fn(**graph_params)

    # Store metrics for all tasks
    all_results = []

    for i in range(num_tasks):
        task = generate_task(G)

        # Run model
        model_output, attention = run_model_reasoning(task)

        # Compute metrics
        result = {
            "experiment": exp_name,
            "task_id": i,
            "task": task,
            "model_output": model_output,
            "exact_match": compute_exact_match(task["gold_path"], model_output),
            "edit_distance": compute_edit_distance(task["gold_path"], model_output),
            "path_length_diff": compute_path_length_difference(task["gold_path"], model_output),
            "reward_accuracy": compute_reward_accuracy(task, model_output),
        }

        all_results.append(result)

        # ---- Save attention heatmap ----
        if attention is not None:
            save_attention_heatmap(attention, exp_name, i)

    return pd.DataFrame(all_results)


# --------------------------------------------------------
#         SAVE ATTENTION HEATMAPS
# --------------------------------------------------------

def save_attention_heatmap(attention_matrix, exp_name, task_id):
    attention = np.array(attention_matrix)

    plt.figure(figsize=(6, 5))
    sns.heatmap(attention, cmap="viridis")
    plt.title(f"Attention Heatmap - {exp_name} - Task {task_id}")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")

    heatmap_path = os.path.join(
        HEATMAP_DIR, f"{exp_name}_task_{task_id}_attention.png"
    )
    plt.savefig(heatmap_path)
    plt.close()


# --------------------------------------------------------
#          MASTER RUNNER
# --------------------------------------------------------

def run_all_experiments():
    print("\n================ RUNNING ALL EXPERIMENTS ================\n")
    all_dfs = []

    for exp in EXPERIMENTS:
        df = run_single_experiment(exp)
        all_dfs.append(df)

        # Save intermediate CSV
        out_path = os.path.join(OUTPUT_DIR, f"{exp['name']}.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved results → {out_path}")

    # Merge all experiments
    final_df = pd.concat(all_dfs, ignore_index=True)

    final_csv = os.path.join(OUTPUT_DIR, "all_experiments_combined.csv")
    final_df.to_csv(final_csv, index=False)

    print("\n==========================================================")
    print("All experiments completed successfully!")
    print(f"Final merged results saved at: {final_csv}")
    print("Attention heatmaps saved to:", HEATMAP_DIR)
    print("==========================================================\n")

    return final_df


# --------------------------------------------------------
#                    MAIN
# --------------------------------------------------------

if __name__ == "__main__":
    run_all_experiments()
