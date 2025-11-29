# ================================================================
# evaluation_runner.py
# ================================================================

import os
import json
import torch
import datetime
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from hybrid_runner import run_hybrid, TransformersLLM
from rsa_analysis import (
    build_theoretical_rsm,
    compute_room_embeddings_from_hidden_states,
    rsm_from_embeddings,
    rsa_correlation
)
from attention_analysis import attention_to_room_ratio

# ---------------------------
# Output directory
# ---------------------------
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------
# Experiment configurations
# ---------------------------
EXPERIMENTS = [
    {"name": "n7line", "graph": create_line_graph, "start_node": "Room 1"},
    {"name": "n7tree", "graph": create_tree_graph, "start_node": "Room 1"},
    {"name": "n15clustered", "graph": create_clustered_graph, "start_node": "Room 1"},
]

# ---------------------------
# Utility functions
# ---------------------------
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def draw_graph(G, path=None, title="Graph", save_path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color="lightblue",
        node_size=900,
        font_size=10
    )
    if path and len(path) > 1:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color='red')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------
# Main experiment runner
# ---------------------------
def run_experiment(exp, model_wrapper, max_new_tokens=20):
    G = exp["graph"]()
    start_node = exp["start_node"]
    graph_name = exp["name"]

    results = {
        "experiment": graph_name,
        "ground_truth_path": [],
        "hybrid_best": None,
        "rsa_spearman_r": None,
        "rsa_pval": None,
        "attention_ratio": None
    }

    try:
        # Ground-truth BFS path
        gt_path = bfs_optimal_path_to_max_reward(G, start_node=start_node)
        results["ground_truth_path"] = gt_path
        print(f"[INFO] Ground-truth BFS path: {gt_path}")

        # Run Hybrid LLM experiment
        print("[INFO] Generating LLM activations...")
        hybrid_out = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)
        results["hybrid_best"] = hybrid_out.get("best")

        # Save activations for RSA / attention analysis
        prompt = base_description_text(G)
        activations = model_wrapper.generate_with_activations(prompt, max_new_tokens=max_new_tokens)
        save_json(
            {
                "prompt": prompt,
                "hidden_states_shapes": [h.shape for h in activations["hidden_states"]],
                "attentions_shapes": [a.shape for a in activations["attentions"]]
            },
            os.path.join(RESULTS_DIR, f"{graph_name}_activations.json")
        )

        # ---------------------------
        # RSA Analysis
        # ---------------------------
        rooms = list(G.nodes())
        positions_map = {r: [i for i in range(activations["hidden_states"][-1].shape[0])] for r in rooms}
        llm_embs = compute_room_embeddings_from_hidden_states(
            activations["hidden_states"][-1].detach().cpu().numpy(),
            positions_map,
            method="mean"
        )

        if llm_embs.ndim == 1:
            llm_embs = llm_embs.reshape(1, -1)

        empirical_rsm = rsm_from_embeddings(llm_embs)
        theoretical_rsm = build_theoretical_rsm(G, rooms)
        r, p = rsa_correlation(empirical_rsm, theoretical_rsm)
        results["rsa_spearman_r"] = r
        results["rsa_pval"] = p

        # ---------------------------
        # Attention Analysis
        # ---------------------------
        attn_ratio = attention_to_room_ratio(
            activations["attentions"][-1].detach().cpu().numpy()[0],
            positions_map
        )
        results["attention_ratio"] = attn_ratio

        # ---------------------------
        # Save graph image
        # ---------------------------
        draw_graph(
            G,
            path=gt_path,
            title=f"{graph_name} - Ground Truth Path",
            save_path=os.path.join(RESULTS_DIR, f"{graph_name}_graph.png")
        )

    except Exception as e:
        print(f"[ERROR] Experiment {graph_name} failed: {e}")

    return results

# ---------------------------
# Main entry point
# ---------------------------
def main():
    print("[INFO] Loading model...")
    model_id = "microsoft/phi-3-mini-4k-instruct"
    model_wrapper = TransformersLLM(model_id=model_id, device="cpu")
    print("[INFO] Model loaded.")

    all_results = []
    for exp in EXPERIMENTS:
        print(f"[INFO] Running graph '{exp['name']}' from start node '{exp['start_node']}'")
        metrics = run_experiment(exp, model_wrapper, max_new_tokens=20)
        all_results.append(metrics)

    # Save all results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to {csv_path}")
    print("[INFO] All experiments completed successfully!")

if __name__ == "__main__":
    main()
