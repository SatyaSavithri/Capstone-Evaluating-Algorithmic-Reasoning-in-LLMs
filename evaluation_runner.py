# evaluation_runner.py
import os
import time
import logging
import torch
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner import run_hybrid  # We use this as-is, no TransformersLLM import

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Experiments configuration ---
EXPERIMENTS = [
    {"name": "n7line", "graph_fn": create_line_graph},
    {"name": "n7tree", "graph_fn": create_tree_graph},
    {"name": "n15clustered", "graph_fn": create_clustered_graph},
]

# --- Helper function: compute Representational Similarity Matrix ---
def rsm_from_embeddings(embs):
    embs = torch.stack(embs).detach().cpu().numpy()  # Convert to numpy
    if len(embs.shape) != 2:
        raise ValueError("Embeddings must be 2D")
    d = pdist(embs, metric="cosine")
    return squareform(d)

# --- Ground-truth BFS path function ---
def bfs_optimal_path_to_max_reward(G, start_node):
    """Compute BFS path from start node to node with maximum reward."""
    from collections import deque

    visited = set()
    queue = deque([[start_node]])
    max_reward_node = max(G.nodes, key=lambda n: G.nodes[n].get("reward", 0))

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == max_reward_node:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in G.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return [start_node]  # fallback

# --- Run single experiment ---
def run_experiment(exp, model_wrapper, max_new_tokens=20):
    G = exp["graph_fn"]()
    start_node = "Room 1"
    try:
        # Ground-truth BFS path
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        logging.info(f"Ground-truth BFS path: {gt_path}")

        # Run hybrid model (activations) — do NOT pass start_node
        logging.info("Generating LLM activations...")
        activations, attention_matrices = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)

        # Compute Representational Similarity Matrix (RSM) from embeddings
        llm_embs = activations
        empirical_rsm = rsm_from_embeddings(llm_embs)

        # Save activations and RSM
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        torch.save(activations, os.path.join(RESULTS_DIR, f"{exp['name']}_activations_{timestamp}.pt"))
        torch.save(attention_matrices, os.path.join(RESULTS_DIR, f"{exp['name']}_attentions_{timestamp}.pt"))
        pd.DataFrame(empirical_rsm).to_csv(os.path.join(RESULTS_DIR, f"{exp['name']}_rsm_{timestamp}.csv"), index=False)

        return {"experiment": exp["name"], "success": True, "gt_path": gt_path}

    except Exception as e:
        logging.error(f"Experiment {exp['name']} failed: {e}")
        return {"experiment": exp["name"], "success": False, "error": str(e)}

# --- Main function ---
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/phi-3-mini-4k-instruct"

    logging.info(f"Loading model {model_id} on {device} (this may take time)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    model_wrapper = {"model": model, "tokenizer": tokenizer, "device": device}

    results = []
    for exp in EXPERIMENTS:
        logging.info(f"Running graph '{exp['name']}' from start node 'Room 1'")
        res = run_experiment(exp, model_wrapper)
        results.append(res)

    # Save summary CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(summary_file, index=False)
    logging.info(f"Results saved to {summary_file}")
    logging.info("All experiments completed!")

if __name__ == "__main__":
    main()
