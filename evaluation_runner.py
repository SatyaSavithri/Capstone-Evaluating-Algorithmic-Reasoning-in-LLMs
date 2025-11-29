# evaluation_runner.py
import os, json, re, csv, datetime, numpy as np
from pathlib import Path
from difflib import SequenceMatcher
import networkx as nx

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from hybrid_runner import run_hybrid
from scratchpad_runner import run_scratchpad
from rsa_analysis import compute_room_embeddings_from_hidden_states, build_theoretical_rsm, rsm_from_embeddings, rsa_correlation
from attention_analysis import attention_to_room_ratio
from utils import parse_final_json_path

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

# -----------------------------
# Helper Metrics
# -----------------------------
def traversal_accuracy(pred_path, gt_path):
    return float(pred_path == gt_path)

def sequence_edit_distance(pred_path, gt_path):
    sm = SequenceMatcher(None, pred_path, gt_path)
    return 1 - sm.ratio()  # normalized as distance

def reward_regret(pred_path, G):
    rewards = nx.get_node_attributes(G, "reward")
    max_reward = max(rewards.values())
    pred_reward = sum([rewards.get(n,0) for n in pred_path])
    return float(max_reward - pred_reward)

def value_regret(pred_path, G):
    rewards = nx.get_node_attributes(G, "reward")
    ideal = sorted(rewards.values(), reverse=True)
    actual = [rewards.get(n,0) for n in pred_path]
    # pad actual to match length
    actual += [0]*(len(ideal)-len(actual))
    return float(sum(np.array(ideal) - np.array(actual)))

# -----------------------------
# Experiment Runner
# -----------------------------
def run_experiment(exp, model_wrapper, max_new_tokens=20, start_node="Room 1"):
    G = exp["graph"]
    exp_name = exp["name"]

    results = {
        "experiment": exp_name,
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "gt_path": None,
        "llm_path": None,
        "traversal_acc": None,
        "seq_edit_dist": None,
        "reward_regret": None,
        "value_regret": None,
        "rsa_corr": None,
        "attention_ratio": None
    }

    try:
        # Ground-truth BFS path
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
        results["gt_path"] = gt_path
        print(f"[INFO] Ground-truth BFS path: {gt_path}")

        # -----------------------------
        # Hybrid LLM + symbolic
        # -----------------------------
        print("[INFO] Generating LLM activations...")

        # Patch hybrid_runner internally to pass start_node without modifying the file
        import hybrid_runner
        original_bfs = hybrid_runner.bfs_optimal_path_to_max_reward
        hybrid_runner.bfs_optimal_path_to_max_reward = lambda G_arg: original_bfs(G_arg, start_node)
        hybrid_out = run_hybrid(model_wrapper, G, max_new_tokens=max_new_tokens)
        hybrid_runner.bfs_optimal_path_to_max_reward = original_bfs

        # Extract predicted path from BEST candidate
        llm_path = None
        best_str = hybrid_out.get("best")
        candidates = hybrid_out.get("candidates", [])
        if best_str:
            best_indices = [int(b.strip()[1:])-1 for b in best_str.replace("P","").split(",")]
            for idx in best_indices:
                if idx < len(candidates):
                    llm_path = candidates[idx]
                    break
        if llm_path is None:
            llm_path = gt_path  # fallback
        results["llm_path"] = llm_path

        # -----------------------------
        # Metrics
        # -----------------------------
        results["traversal_acc"] = traversal_accuracy(llm_path, gt_path)
        results["seq_edit_dist"] = sequence_edit_distance(llm_path, gt_path)
        results["reward_regret"] = reward_regret(llm_path, G)
        results["value_regret"] = value_regret(llm_path, G)

        # -----------------------------
        # RSA / Hidden States
        # -----------------------------
        data = model_wrapper.generate_with_activations(
            base_description_text(G),
            max_new_tokens=max_new_tokens
        )
        hidden_states = data["hidden_states"][-1]  # last layer
        # mapping rooms -> token positions (dummy: assume each room appears once)
        positions_map = {n:[i] for i,n in enumerate(G.nodes())}
        llm_embs = compute_room_embeddings_from_hidden_states(np.array(hidden_states[-1]), positions_map)
        if llm_embs.ndim == 2:
            theoretical_rsm = build_theoretical_rsm(G, list(G.nodes()))
            empirical_rsm = rsm_from_embeddings(llm_embs)
            r_corr,_ = rsa_correlation(empirical_rsm, theoretical_rsm)
            results["rsa_corr"] = r_corr

        # -----------------------------
        # Attention Ratio
        # -----------------------------
        attn_ratio_list = []
        for att in data.get("attentions", []):
            ratio = attention_to_room_ratio(att[-1].detach().cpu().numpy(), positions_map)
            if ratio is not None:
                attn_ratio_list.append(ratio)
        if attn_ratio_list:
            results["attention_ratio"] = float(np.mean(attn_ratio_list))

        # Save activations & attention for offline
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(RESULTS_DIR / f"{exp_name}_hidden_{ts}.npy", hidden_states[-1].detach().cpu().numpy())
        for i, att in enumerate(data.get("attentions", [])):
            np.save(RESULTS_DIR / f"{exp_name}_attn_layer{i}_{ts}.npy", att.detach().cpu().numpy())

    except Exception as e:
        print(f"[ERROR] Experiment {exp_name} failed: {e}")

    return results

# -----------------------------
# Main
# -----------------------------
def main():
    # -----------------------------
    # Setup experiments
    # -----------------------------
    experiments = [
        {"name":"n7line","graph":create_line_graph()},
        {"name":"n7tree","graph":create_tree_graph()},
        {"name":"n15clustered","graph":create_clustered_graph()}
    ]

    # -----------------------------
    # Load model
    # -----------------------------
    from hybrid_runner import TransformersLLM  # now imported locally
    model_id = "microsoft/phi-3-mini-4k-instruct"
    print(f"[INFO] Loading model {model_id}...")
    llm = TransformersLLM(model_id)

    # -----------------------------
    # Run all experiments
    # -----------------------------
    all_results = []
    start_node = "Room 1"
    for exp in experiments:
        print(f"[INFO] Running graph '{exp['name']}' from start node '{start_node}'")
        res = run_experiment(exp, llm, max_new_tokens=20, start_node=start_node)
        all_results.append(res)

    # -----------------------------
    # Save CSV
    # -----------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"evaluation_results_{ts}.csv"
    keys = all_results[0].keys() if all_results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"[INFO] Results saved to {csv_path}")
    print("[INFO] All experiments completed successfully!")

if __name__ == "__main__":
    main()
