# evaluation_runner.py
import os
import torch
import pandas as pd
from datetime import datetime
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from hybrid_runner_eval import run_hybrid_eval  # our new compatible hybrid runner

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
                if neighbor not in visited:
                    queue.append(path + [neighbor])
    return [start_node]  # fallback

# --- Experiment definitions ---
EXPERIMENTS = [
    ("n7line", create_line_graph),
    ("n7tree", create_tree_graph),
    ("n15clustered", create_clustered_graph),
]

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_experiment(exp_name, graph_fn, device="cuda"):
    G = graph_fn()
    start_node = "Room 1"
    try:
        # Ground-truth BFS path
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        print(f"[INFO] Ground-truth BFS path: {gt_path}")

        # Run hybrid evaluation
        activations, success = run_hybrid_eval(G, start_node=start_node, device=device)

        # Save activations safely
        act_file = os.path.join(RESULTS_DIR, f"{exp_name}_activations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
        torch.save(activations, act_file)
        print(f"[INFO] Saved activations to {act_file}")

        return {"experiment": exp_name, "success": success, "gt_path": gt_path}

    except Exception as e:
        print(f"[ERROR] Experiment {exp_name} failed: {e}")
        return {"experiment": exp_name, "success": False, "error": str(e)}

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    print(f"[INFO] Running evaluation on device: {device}")

    for exp_name, graph_fn in EXPERIMENTS:
        print(f"[INFO] Running graph '{exp_name}' from start node 'Room 1'")
        res = run_experiment(exp_name, graph_fn, device=device)
        results.append(res)

    # Save results CSV
    csv_file = os.path.join(RESULTS_DIR, f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(results).to_csv(csv_file, index=False)
    print(f"[INFO] Results saved to {csv_file}")
    print("[INFO] All experiments completed!")

if __name__ == "__main__":
    main()
