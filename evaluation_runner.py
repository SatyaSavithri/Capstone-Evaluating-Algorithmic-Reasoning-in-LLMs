import os
import torch
import logging
import pandas as pd
from datetime import datetime
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import TransformersLLM, run_hybrid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def run_experiment(exp_name, G, model_wrapper):
    """Runs hybrid experiment and saves activations and attention matrices."""
    start_node = list(G.nodes())[0]
    gt_path = bfs_optimal_path_to_max_reward(G, start_node)
    logger.info(f"[INFO] Ground-truth BFS path: {gt_path}")

    try:
        activations, attentions = run_hybrid(model_wrapper, G)
        # Save activations and attentions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.pt")
        att_file = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{timestamp}.pt")
        torch.save(activations, act_file)
        torch.save(attentions, att_file)
        logger.info(f"[INFO] Saved activations: {act_file}")
        logger.info(f"[INFO] Saved attentions: {att_file}")
        success = True
        error_msg = ""
    except Exception as e:
        success = False
        error_msg = str(e)
        logger.error(f"[ERROR] Experiment {exp_name} failed: {error_msg}")

    return {
        "experiment": exp_name,
        "success": success,
        "error": error_msg,
        "gt_path": gt_path,
    }


def main():
    model_wrapper = TransformersLLM(model_id="microsoft/phi-3-mini-4k-instruct", device="cuda")

    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph(),
    }

    results = []
    for exp_name, G in experiments.items():
        logger.info(f"[INFO] Running experiment '{exp_name}'")
        result = run_experiment(exp_name, G, model_wrapper)
        results.append(result)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    logger.info(f"[INFO] Results saved to {results_file}")
    logger.info("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()
