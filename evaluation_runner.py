# evaluation_runner.py
import os
import torch
import logging
import datetime
import networkx as nx
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import TransformersLLM, run_hybrid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def save_results(exp_name, activations_list, attentions_list):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{timestamp}.pt")
    att_file = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{timestamp}.pt")
    torch.save(activations_list, act_file)
    torch.save(attentions_list, att_file)
    logger.info(f"Saved activations: {act_file}")
    logger.info(f"Saved attentions: {att_file}")

def main():
    llm = TransformersLLM(model_name="microsoft/phi-3-mini-4k-instruct", device="cuda")

    experiments = {
        "n7line": create_line_graph(),
        "n7tree": create_tree_graph(),
        "n15clustered": create_clustered_graph(),
    }

    for exp_name, graph in experiments.items():
        logger.info(f"Running experiment '{exp_name}'")
        start_node = list(graph.nodes)[0]
        gt_path = bfs_optimal_path_to_max_reward(graph, start_node)
        logger.info(f"Ground-truth BFS path: {gt_path}")

        activations, attentions = run_hybrid(llm, gt_path)
        save_results(exp_name, activations, attentions)

if __name__ == "__main__":
    main()
