# evaluation_runner.py
import os
import logging
from datetime import datetime
import torch
import networkx as nx
import pandas as pd
from hybrid_runner_eval import TransformersLLM, run_hybrid  # your custom hybrid runner
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from rsa_analysis import rsm_from_embeddings  # assuming this is your embedding/RSM computation

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define experiments
EXPERIMENTS = [
    ("n7line", create_line_graph),
    ("n7tree", create_tree_graph),
    ("n15clustered", create_clustered_graph)
]

def run_experiment(exp_name, G, model_wrapper, max_new_tokens=20):
    logger.info(f"[INFO] Running experiment '{exp_name}'")
    
    try:
        # Compute ground-truth BFS path
        start_node = list(G.nodes)[0]
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        logger.info(f"[INFO] Ground-truth BFS path: {gt_path}")

        # Generate LLM activations
        logger.info("[INFO] Generating LLM activations...")
        activations, attentions = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)

        # Save activations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.pt")
        att_file = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{timestamp}.pt")
        torch.save(activations, act_file)
        torch.save(attentions, att_file)
        logger.info(f"[INFO] Saved activations to {act_file}")
        logger.info(f"[INFO] Saved attentions to {att_file}")

        # Compute embeddings and RSM
        if activations is not None:
            emb = torch.stack(list(activations.values())).detach().cpu()
            if emb.ndim == 2:
                empirical_rsm = rsm_from_embeddings(emb)
            else:
                logger.warning(f"[WARN] Activations tensor has invalid shape {emb.shape}, skipping RSM")
                empirical_rsm = None
        else:
            logger.warning(f"[WARN] Activations is None, skipping RSM computation")
            empirical_rsm = None

        return {
            "experiment": exp_name,
            "success": True,
            "gt_path": gt_path,
            "activations_file": act_file,
            "attentions_file": att_file,
            "rsm": empirical_rsm
        }

    except Exception as e:
        logger.error(f"[ERROR] Experiment {exp_name} failed: {e}")
        return {
            "experiment": exp_name,
            "success": False,
            "error": str(e)
        }


# BFS path to max reward
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
    model_wrapper = TransformersLLM(model_name="microsoft/phi-3-mini-4k-instruct", device="cuda")
    results = []

    for exp_name, graph_fn in EXPERIMENTS:
        logger.info(f"[INFO] Running graph '{exp_name}' from start node 'Room 1'")
        G = graph_fn()
        result = run_experiment(exp_name, G, model_wrapper, max_new_tokens=20)
        results.append(result)

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    logger.info(f"[INFO] Results saved to {results_file}")
    logger.info("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()
