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
from scipy.spatial.distance import pdist, squareform

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from hybrid_runner import run_hybrid
from prompts import base_description_text
from rsa_analysis import (
    build_theoretical_rsm,
    compute_room_embeddings_from_hidden_states,
    rsm_from_embeddings,
    rsa_correlation
)
from attention_analysis import attention_to_room_ratio
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENTS = [
    {"name": "n7line", "graph": create_line_graph, "start_node": "Room 1"},
    {"name": "n7tree", "graph": create_tree_graph, "start_node": "Room 1"},
    {"name": "n15clustered", "graph": create_clustered_graph, "start_node": "Room 1"},
]

# ---------------------------
def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def draw_graph(G, path=None, title="Graph", save_path=None):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=900, font_size=10)
    if path and len(path) > 1:
        edges = list(zip(path, path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color='red')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

# ---------------------------
def run_experiment(exp, model, tokenizer, device="cpu", max_new_tokens=20):
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
        gt_path = bfs_optimal_path_to_max_reward(G, start_node=start_node)
        results["ground_truth_path"] = gt_path
        print(f"[INFO] Ground-truth BFS path: {gt_path}")

        print("[INFO] Running hybrid model...")
        hybrid_out = run_hybrid(model, G, max_new_tokens=max_new_tokens, device=device)
        results["hybrid_best"] = hybrid_out.get("best")

        # Save activations for analysis
        prompt = base_description_text(G)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
        hidden_states = [h.cpu().numpy() for h in outputs.hidden_states]
        attentions = [a.cpu().numpy() for a in outputs.attentions]

        save_json(
            {
                "prompt": prompt,
                "hidden_states_shapes": [h.shape for h in hidden_states],
                "attentions_shapes": [a.shape for a in attentions]
            },
            os.path.join(RESULTS_DIR, f"{graph_name}_activations.json")
        )

        # RSA
        rooms = list(G.nodes())
        positions_map = {r: [i for i in range(hidden_states[-1].shape[1])] for r in rooms}
        llm_embs = compute_room_embeddings_from_hidden_states(hidden_states[-1], positions_map, method="mean")
        if llm_embs.ndim == 1:
            llm_embs = llm_embs.reshape(1, -1)
        empirical_rsm = rsm_from_embeddings(llm_embs)
        theoretical_rsm = build_theoretical_rsm(G, rooms)
        r, p = rsa_correlation(empirical_rsm, theoretical_rsm)
        results["rsa_spearman_r"] = r
        results["rsa_pval"] = p

        # Attention
        attn_ratio = attention_to_room_ratio(attentions[-1][0], positions_map)
        results["attention_ratio"] = attn_ratio

        draw_graph(G, path=gt_path, title=f"{graph_name} - Ground Truth Path",
                   save_path=os.path.join(RESULTS_DIR, f"{graph_name}_graph.png"))

    except Exception as e:
        print(f"[ERROR] Experiment {graph_name} failed: {e}")

    return results

# ---------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/phi-3-mini-4k-instruct"
    print(f"[INFO] Loading model {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto").to(device)
    print("[INFO] Model loaded.")

    all_results = []
    for exp in EXPERIMENTS:
        print(f"[INFO] Running graph '{exp['name']}' from start node '{exp['start_node']}'")
        metrics = run_experiment(exp, model, tokenizer, device=device, max_new_tokens=20)
        all_results.append(metrics)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Results saved to {csv_path}")
    print("[INFO] All experiments completed successfully!")

if __name__ == "__main__":
    main()
