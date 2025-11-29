# evaluation_runner.py
import os
import torch
import pandas as pd
from datetime import datetime
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import TransformersLLM, run_hybrid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Ground-truth BFS path function ---
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
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return [start_node]


def save_tensor_as_csv(tensor, filename):
    """Save tensor or list of tensors as CSV for easy inspection"""
    if isinstance(tensor, list):
        # convert each tensor to DataFrame and save separately
        for idx, t in enumerate(tensor):
            if t is not None:
                df = pd.DataFrame(t.numpy() if torch.is_tensor(t) else t)
                df.to_csv(filename.replace(".csv", f"_{idx}.csv"), index=False)
    elif torch.is_tensor(tensor):
        df = pd.DataFrame(tensor.numpy())
        df.to_csv(filename, index=False)


def main():
    experiments = [
        ("n7line", create_line_graph()),
        ("n7tree", create_tree_graph()),
        ("n15clustered", create_clustered_graph())
    ]

    # Initialize model
    model_wrapper = TransformersLLM(model_name="microsoft/phi-3-mini-4k-instruct", device="cuda")

    results_summary = []

    for exp_name, G in experiments:
        start_node = "Room 1"
        logger.info(f"Running experiment '{exp_name}'")
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        logger.info(f"Ground-truth BFS path: {gt_path}")

        try:
            activations, attentions = run_hybrid(G, start_node, model_wrapper)

            if activations is not None:
                pt_file_act = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                torch.save(activations, pt_file_act)
                csv_file_act = pt_file_act.replace(".pt", ".csv")
                save_tensor_as_csv(activations, csv_file_act)
                logger.info(f"Saved activations: {pt_file_act} and {csv_file_act}")

            if attentions is not None:
                pt_file_att = os.path.join(RESULTS_DIR, f"{exp_name}_attentions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                torch.save(attentions, pt_file_att)
                csv_file_att = pt_file_att.replace(".pt", ".csv")
                save_tensor_as_csv(attentions, csv_file_att)
                logger.info(f"Saved attentions: {pt_file_att} and {csv_file_att}")

            results_summary.append({"experiment": exp_name, "success": True, "gt_path": gt_path})
        except Exception as e:
            logger.error(f"Experiment {exp_name} failed: {e}")
            results_summary.append({"experiment": exp_name, "success": False, "error": str(e)})

    # Save summary
    summary_file = os.path.join(RESULTS_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(results_summary).to_csv(summary_file, index=False)
    logger.info(f"Results saved to {summary_file}")
    logger.info("All experiments completed!")


if __name__ == "__main__":
    main()
