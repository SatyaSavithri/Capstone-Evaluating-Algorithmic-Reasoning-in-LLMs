# evaluation_runner.py
import os
import logging
from datetime import datetime
import torch
import pandas as pd
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import TransformersLLM, run_hybrid
from rsa_analysis import rsm_from_embeddings

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Experiments
EXPERIMENTS = [
    ("n7line", create_line_graph),
    ("n7tree", create_tree_graph),
    ("n15clustered", create_clustered_graph)
]


# BFS ground-truth path
def bfs_optimal_path_to_max_reward(G, start_node):
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
                queue.append(path + [neighbor])
    return [start_node]


def run_experiment(exp_name, G, model_wrapper, max_new_tokens=20):
    logger.info(f"[INFO] Running experiment '{exp_name}'")
    result = {"experiment": exp_name}

    start_node = list(G.nodes())[0]
    gt_path = bfs_optimal_path_to_max_reward(G, start_node)
    result["gt_path"] = gt_path
    logger.info(f"[INFO] Ground-truth BFS path: {gt_path}")

    try:
        logger.info("[INFO] Generating LLM activations...")
        activations, attentions = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.pt")
        att_file = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{timestamp}.pt")
        torch.save(activations, act_file)
        torch.save(attentions, att_file)
        logger.info(f"[INFO] Saved activations: {act_file}")
        logger.info(f"[INFO] Saved attentions: {att_file}")

        # Compute embeddings and RSM
        emb_list = [activations[node] for node in G.nodes()]
        emb_tensor = torch.stack(emb_list)
        result["rsm"] = rsm_from_embeddings(emb_tensor)

        result["success"] = True
        result["activations_file"] = act_file
        result["attentions_file"] = att_file
    except Exception as e:
        logger.error(f"[ERROR] Experiment {exp_name} failed: {e}")
        result["success"] = False
        result["error"] = str(e)

    return result


def main():
    model_wrapper = TransformersLLM(model_id="microsoft/phi-3-mini-4k-instruct", device="cuda")
    results = []

    for exp_name, graph_fn in EXPERIMENTS:
        G = graph_fn()
        results.append(run_experiment(exp_name, G, model_wrapper))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.csv")
    pd.DataFrame(results).to_csv(results_file, index=False)
    logger.info(f"[INFO] Results saved to {results_file}")
    logger.info("[INFO] All experiments completed!")


if __name__ == "__main__":
    main()
