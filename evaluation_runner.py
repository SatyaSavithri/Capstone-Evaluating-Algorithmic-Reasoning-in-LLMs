import os
import torch
import datetime
import networkx as nx
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from transformers import AutoModelForCausalLM, AutoTokenizer
from hybrid_runner import run_hybrid, bfs_optimal_path_to_max_reward  # Only function imports

# Import graphs from your graphs.py
from graphs import create_line_graph, create_tree_graph, create_clustered_graph

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EXPERIMENTS = [
    {"name": "n7line", "graph_fn": create_line_graph},
    {"name": "n7tree", "graph_fn": create_tree_graph},
    {"name": "n15clustered", "graph_fn": create_clustered_graph},
]

def rsm_from_embeddings(embs: torch.Tensor) -> torch.Tensor:
    """
    Compute representational similarity matrix from embeddings.
    emb: Tensor of shape (num_nodes, embedding_dim)
    """
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D tensor for embeddings, got {emb.ndim}D")
    dists = pdist(embs.cpu().numpy(), metric="cosine")
    rsm = squareform(dists)
    return rsm

def run_experiment(exp, model_wrapper, max_new_tokens=20):
    """
    Run one experiment: generate activations, compute metrics.
    """
    G = exp["graph_fn"]()
    start_node = "Room 1"
    try:
        # Ground-truth BFS path
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        print(f"[INFO] Ground-truth BFS path: {gt_path}")

        # Run hybrid model to get predictions + activations
        print("[INFO] Generating LLM activations...")
        activations, attention_matrices = run_hybrid(
            model_wrapper, G, start_node=start_node, max_new_tokens=max_new_tokens
        )

        # Save activations and attention
        exp_dir = os.path.join(RESULTS_DIR, exp["name"])
        os.makedirs(exp_dir, exist_ok=True)
        torch.save(activations, os.path.join(exp_dir, "activations.pt"))
        torch.save(attention_matrices, os.path.join(exp_dir, "attention.pt"))
        print(f"[INFO] Saved activations and attention matrices in '{exp_dir}'")

        # Compute RSM
        llm_embs = torch.stack(list(activations.values()))
        empirical_rsm = rsm_from_embeddings(llm_embs)

        return {"experiment": exp["name"], "gt_path": gt_path, "rsm": empirical_rsm}

    except Exception as e:
        print(f"[ERROR] Experiment {exp['name']} failed: {e}")
        return {"experiment": exp["name"], "error": str(e)}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/phi-3-mini-4k-instruct"
    print(f"[INFO] Loading model {model_id} on {device} (this may take time)...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with device_map="auto" and proper dtype (no .to(device))
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16
    )

    print(f"[INFO] Model {model_id} loaded.")

    results_list = []

    # Iterate over experiments
    for exp in EXPERIMENTS:
        print(f"[INFO] Running graph '{exp['name']}' from start node 'Room 1'")
        metrics = run_experiment(exp, model)
        results_list.append(metrics)

    # Save all results to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    pd.DataFrame(results_list).to_csv(results_file, index=False)
    print(f"[INFO] Results saved to {results_file}")
    print("[INFO] All experiments completed!")

if __name__ == "__main__":
    main()
