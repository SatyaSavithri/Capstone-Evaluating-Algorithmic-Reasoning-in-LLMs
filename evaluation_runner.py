# evaluation_runner.py
import os
import csv
import json
import numpy as np
from datetime import datetime

import networkx as nx
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from hybrid_runner import run_hybrid
from rsa_analysis import build_theoretical_rsm, compute_room_embeddings_from_hidden_states, rsm_from_embeddings, rsa_correlation
from attention_analysis import attention_to_room_ratio

# ================================
# Import TransformersLLM from correct file
# ================================
from run_capstone_transformers import TransformersLLM

# ================================
# Config
# ================================
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================
# Experiments
# ================================
EXPERIMENTS = [
    {"graph_name": "n7line", "graph_fn": create_line_graph, "start_node": "Room 1"},
    {"graph_name": "n7tree", "graph_fn": create_tree_graph, "start_node": "Room 1"},
    {"graph_name": "n15clustered", "graph_fn": create_clustered_graph, "start_node": "Room 1"},
]

# ================================
# Utilities
# ================================
def save_csv(results, filename):
    keys = results[0].keys() if results else []
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"[INFO] Results saved to {filename}")

# ------------------------
# Simple Levenshtein edit distance
# ------------------------
def edit_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j] = dp[i-1][j-1] if seq1[i-1]==seq2[j-1] else 1+min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
    return dp[n][m]

# ================================
# Run Experiment
# ================================
def run_experiment(exp, model_wrapper, max_new_tokens=20):
    G = exp["graph_fn"]()
    start_node = exp["start_node"]

    # ------------------------
    # Ground-truth symbolic path
    # ------------------------
    symbolic_path = bfs_optimal_path_to_max_reward(G, start_node)
    print(f"[INFO] Ground-truth BFS path: {symbolic_path}")

    # ------------------------
    # Hybrid / LLM generation
    # ------------------------
    print("[INFO] Generating LLM activations...")
    hybrid_out = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)
    llm_best = hybrid_out["best"]

    # ------------------------
    # Attention Analysis
    # ------------------------
    att_ratio = None
    if "attentions" in hybrid_out.get("model_output", {}):
        attentions = hybrid_out["model_output"]["attentions"]
        positions_map = {n: [i] for i, n in enumerate(G.nodes()) if i < attentions[0].shape[-1]}
        att_ratio = attention_to_room_ratio(attentions[0][0].detach().cpu().numpy(), positions_map)

    # ------------------------
    # Dynamic RSA
    # ------------------------
    rsa_r, rsa_p = None, None
    if "hidden_states" in hybrid_out.get("model_output", {}):
        hidden_states = hybrid_out["model_output"]["hidden_states"][-1][0].detach().cpu().numpy()
        rooms = list(G.nodes())
        positions_map = {room: [i] for i, room in enumerate(rooms) if i < hidden_states.shape[0]}
        llm_embs = compute_room_embeddings_from_hidden_states(hidden_states, positions_map, method="mean")
        llm_embs = np.array(llm_embs)
        theoretical_rsm = build_theoretical_rsm(G, rooms)
        empirical_rsm = rsm_from_embeddings(llm_embs)
        rsa_r, rsa_p = rsa_correlation(empirical_rsm, theoretical_rsm)

    # ------------------------
    # Behavioral metrics
    # ------------------------
    traversal_accuracy = float(llm_best.split("Room")[1:]==symbolic_path[1:]) if llm_best else 0.0
    sequence_edit_distance = edit_distance(symbolic_path, llm_best.split(" -> ")) if llm_best else len(symbolic_path)

    # ------------------------
    # Prepare result
    # ------------------------
    result = {
        "graph": exp["graph_name"],
        "start_node": start_node,
        "symbolic_path": " -> ".join(symbolic_path),
        "llm_best": llm_best,
        "traversal_accuracy": traversal_accuracy,
        "sequence_edit_distance": sequence_edit_distance,
        "rsa_spearman_r": rsa_r,
        "rsa_pval": rsa_p,
        "attention_ratio": att_ratio
    }
    return result

# ================================
# Main
# ================================
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/phi-3-mini-4k-instruct")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    # ------------------------
    # Load model
    # ------------------------
    print(f"[INFO] Loading model {args.model}...")
    model_wrapper = TransformersLLM(args.model)

    results = []
    for exp in EXPERIMENTS:
        print(f"[INFO] Running graph '{exp['graph_name']}' from start node '{exp['start_node']}'")
        try:
            res = run_experiment(exp, model_wrapper, max_new_tokens=args.max_new_tokens)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] Experiment {exp['graph_name']} failed: {e}")

    # ------------------------
    # Save CSV
    # ------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    save_csv(results, csv_file)

    print("\n[INFO] All experiments completed successfully!")

if __name__ == "__main__":
    main()
