# evaluation_runner.py
import os
import sys
import torch
import argparse
import pandas as pd
from datetime import datetime
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import run_hybrid  # your new hybrid_runner_eval.py
from rsa_analysis import rsm_from_embeddings  # make sure this is available

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)


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


def run_experiment(graph_name, G, model_wrapper, start_node="Room 1", max_new_tokens=20):
    """Run a single experiment: generate LLM outputs and compute metrics."""
    try:
        print(f"[INFO] Ground-truth BFS path: {bfs_optimal_path_to_max_reward(G, start_node)}")
        print("[INFO] Generating LLM activations...")
        
        hybrid_out = run_hybrid(model_wrapper, G, start_node=start_node, max_new_tokens=max_new_tokens)
        
        llm_activations = hybrid_out.get("activations")
        attention_matrices = hybrid_out.get("attentions")
        
        # Save activations and attention matrices as binary
        torch.save(llm_activations, os.path.join(RESULTS_DIR, f"{graph_name}_activations.pt"))
        torch.save(attention_matrices, os.path.join(RESULTS_DIR, f"{graph_name}_attentions.pt"))
        
        # Compute RSM metrics safely
        if llm_activations is not None and len(llm_activations) > 0:
            empirical_rsm = rsm_from_embeddings(llm_activations)
        else:
            empirical_rsm = None
        
        return {
            "experiment": graph_name,
            "success": True,
            "gt_path": bfs_optimal_path_to_max_reward(G, start_node),
            "rsm": empirical_rsm
        }
    except Exception as e:
        print(f"[ERROR] Experiment {graph_name} failed: {e}")
        return {
            "experiment": graph_name,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default="all",
                        choices=["line_graph", "tree_graph", "clustered_graph", "all"])
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    # Load graphs
    graphs = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph()
    }

    selected_graphs = {k: v for k, v in graphs.items() if args.graph == "all" or args.graph in k}

    # Load LLM model wrapper
    from hybrid_runner_eval import TransformersLLM  # new wrapper
    model_wrapper = TransformersLLM(model_id="microsoft/phi-3-mini-4k-instruct", device="cuda")
    
    results = []
    for graph_name, G in selected_graphs.items():
        print(f"[INFO] Running graph '{graph_name}' from start node 'Room 1'")
        exp_result = run_experiment(graph_name, G, model_wrapper, start_node="Room 1",
                                    max_new_tokens=args.max_new_tokens)
        results.append(exp_result)

    # Save results CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    csv_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"[INFO] Results saved to {csv_file}")
    print("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()
