# evaluation_runner.py
import os
import logging
from datetime import datetime
import torch
import pandas as pd
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import TransformersLLM, run_hybrid

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Directory for results
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiments configuration
EXPERIMENTS = [
    ("n7line", create_line_graph),
    ("n7tree", create_tree_graph),
    ("n15clustered", create_clustered_graph),
]

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

def main():
    # Initialize model
    model_name = "microsoft/phi-3-mini-4k-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading model {model_name} on {device}...")
    model_wrapper = TransformersLLM(model_name=model_name, device=device)
    logger.info(f"Model {model_name} loaded.")

    results = []

    for exp_name, graph_fn in EXPERIMENTS:
        logger.info(f"Running experiment '{exp_name}'")
        G = graph_fn()
        start_node = list(G.nodes)[0]
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        logger.info(f"Ground-truth BFS path: {gt_path}")

        # Generate activations and attentions
        try:
            activations, attentions = run_hybrid(model_wrapper, G, start_node=start_node)
            
            # Save as PyTorch binary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.pt")
            att_file = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{timestamp}.pt")
            torch.save(activations, act_file)
            torch.save(attentions, att_file)
            logger.info(f"Saved activations: {act_file}")
            logger.info(f"Saved attentions: {att_file}")

            # Convert activations to CSV for inspection
            act_csv_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.csv")
            # Flatten tensors for CSV
            act_data = {node: act.cpu().numpy().flatten() for node, act in activations.items()}
            df = pd.DataFrame.from_dict(act_data, orient="index")
            df.to_csv(act_csv_file)
            logger.info(f"Activations CSV saved: {act_csv_file}")

            results.append({
                "experiment": exp_name,
                "success": True,
                "gt_path": gt_path
            })
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results.append({
                "experiment": exp_name,
                "success": False,
                "error": str(e)
            })

    # Save summary CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(summary_file, index=False)
    logger.info(f"Results saved to {summary_file}")
    logger.info("All experiments completed!")

if __name__ == "__main__":
    main()
