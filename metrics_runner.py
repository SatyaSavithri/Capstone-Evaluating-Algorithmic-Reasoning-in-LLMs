"""
metrics_runner.py
Interactive evaluation metrics runner for your Capstone project.

NOW UPDATED: Each trial uses a DIFFERENT RANDOM START NODE.
"""

import os
import json
import csv
import math
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from utils import parse_final_json_path
from rsa_analysis import (
    build_theoretical_rsm,
    compute_room_embeddings_from_hidden_states,
    rsm_from_embeddings,
    rsa_correlation
)
from attention_analysis import attention_to_room_ratio
from visualize import plot_rsm

# Try to import your LLM wrapper
try:
    from models_transformers import TransformersLLM
except:
    from hybrid_runner import TransformersLLM

import networkx as nx


# ================================================================
# Helper Functions (same as before)
# ================================================================
def levenshtein_seq(a, b):
    if a == b:
        return 0
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+cost
            )
    return dp[n][m]

def safe_generate_text(llm, prompt, max_new_tokens=128):
    if hasattr(llm, "text_generate"):
        try:
            return llm.text_generate(prompt, max_new_tokens=max_new_tokens)
        except: pass
    if hasattr(llm, "generate"):
        try:
            return llm.generate(prompt, max_new_tokens=max_new_tokens)
        except: pass
    raise RuntimeError("No generation method available.")

def safe_generate_with_activations(llm, prompt, max_new_tokens=64):
    if hasattr(llm, "generate_with_activations"):
        return llm.generate_with_activations(prompt, max_new_tokens=max_new_tokens)
    if hasattr(llm, "tokenizer") and hasattr(llm, "model"):
        inputs = llm.tokenizer(prompt, return_tensors="pt").to(llm.device)
        outputs = llm.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False
        )
        return {
            "prompt": prompt,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions
        }
    raise RuntimeError("No activation method available.")

def compute_dynamic_rsa(hidden_states, positions_map, G):
    layer_embs = []
    try:
        for layer in hidden_states:
            layer_embs.append(layer[0].detach().cpu().numpy())
    except:
        return {"error": "Invalid hidden states"}

    rooms = list(positions_map.keys())
    theoretical = build_theoretical_rsm(G, rooms)
    corrs = []
    for h in layer_embs:
        emb = compute_room_embeddings_from_hidden_states(h, positions_map)
        emp_rsm = rsm_from_embeddings(emb)
        r, _ = rsa_correlation(emp_rsm, theoretical)
        corrs.append(r)

    valid = [c for c in corrs if not math.isnan(c)]
    return {
        "layer_corrs": corrs,
        "mean_corr": float(np.mean(valid)) if valid else float("nan"),
        "max_corr": float(np.nanmax(corrs)) if corrs else float("nan")
    }

def compute_ara(attentions, positions_map):
    ratios = {}
    for l, att in enumerate(attentions):
        att_np = att[0].mean(0).detach().cpu().numpy()
        ratios[f"layer_{l}"] = attention_to_room_ratio(att_np, positions_map)
    return ratios


# ================================================================
# Run a single trial
# ================================================================
def run_single_trial(llm, G, start_node, method):

    print(f"\n➡ Running Trial (start={start_node}, method={method})")

    # Ground truth
    try:
        gt_path = bfs_optimal_path_to_max_reward(G, start_node)
    except:
        gt_path = bfs_optimal_path_to_max_reward(G)

    base = base_description_text(G, start_node)
    prompt = scratchpad_prompt(base, "valuePath")

    text = safe_generate_text(llm, prompt)
    pred_path = parse_final_json_path(text)

    accuracy = int(pred_path == gt_path)
    edit_dist = levenshtein_seq(pred_path, gt_path)

    rsa_res, ara_res = {}, {}

    if method in ["RSA", "ATTENTION"]:
        act = safe_generate_with_activations(llm, prompt)
        hidden_states = act["hidden_states"]
        attentions = act["attentions"]

        rooms = list(G.nodes())
        positions_map = {r: [] for r in rooms}

        if method == "RSA":
            rsa_res = compute_dynamic_rsa(hidden_states, positions_map, G)
        if method == "ATTENTION":
            ara_res = compute_ara(attentions, positions_map)

    return {
        "gt_path": gt_path,
        "pred_path": pred_path,
        "accuracy": accuracy,
        "edit_distance": edit_dist,
        "rsa": rsa_res,
        "ara": ara_res
    }


# ================================================================
# INTERACTIVE UI
# ================================================================
def choose_model():
    print("\nSelect model:")
    print("1) microsoft/phi-3-mini-4k-instruct")
    print("2) google/gemma-2b")
    print("3) Custom")
    c = input("Enter choice: ").strip()

    if c == "1": return "microsoft/phi-3-mini-4k-instruct"
    if c == "2": return "google/gemma-2b"
    return input("Enter custom HuggingFace model ID: ")

def choose_graphs():
    print("\nSelect graphs:")
    print("1) Line")
    print("2) Tree")
    print("3) Clustered")
    print("4) All")
    c = input("Enter choice: ").strip()

    if c == "1": return [("line", create_line_graph())]
    if c == "2": return [("tree", create_tree_graph())]
    if c == "3": return [("cluster", create_clustered_graph())]
    return [
        ("line", create_line_graph()),
        ("tree", create_tree_graph()),
        ("cluster", create_clustered_graph())
    ]

def choose_methods():
    print("\nSelect methods:")
    print("1) Scratchpad")
    print("2) Hybrid")
    print("3) RSA")
    print("4) Attention")
    print("5) All")
    c = input("Enter choice: ").strip()

    if c == "1": return ["SCRATCHPAD"]
    if c == "2": return ["HYBRID"]
    if c == "3": return ["RSA"]
    if c == "4": return ["ATTENTION"]
    return ["SCRATCHPAD", "HYBRID", "RSA", "ATTENTION"]


# ================================================================
# MAIN
# ================================================================
def main():

    print("\n=== CAPSTONE: Interactive Metrics Runner ===")

    model_id = choose_model()
    llm = TransformersLLM(model_id=model_id, device="cpu")

    graphs = choose_graphs()
    methods = choose_methods()

    trials = int(input("\nHow many trials per graph? ").strip())

    out_csv = input("\nEnter output CSV filename (default metrics_results.csv): ").strip()
    if out_csv == "":
        out_csv = "metrics_results.csv"

    results = []
    trial_id = 0

    for gname, G in graphs:

        nodes = list(G.nodes())  # ⭐ NEW: choose random start nodes per trial

        for method in methods:
            for _ in range(trials):

                trial_id += 1

                # ⭐ NEW — random start node each trial
                start = random.choice(nodes)

                print(f"\n================ Trial {trial_id} =================")

                r = run_single_trial(llm, G, start, method)

                results.append({
                    "id": trial_id,
                    "graph": gname,
                    "method": method,
                    "start_node": start,
                    "gt_path": json.dumps(r["gt_path"]),
                    "pred_path": json.dumps(r["pred_path"]),
                    "accuracy": r["accuracy"],
                    "edit_distance": r["edit_distance"],
                    "rsa_mean": r["rsa"].get("mean_corr", ""),
                    "rsa_max": r["rsa"].get("max_corr", ""),
                    "ara": json.dumps(r["ara"])
                })

    # Write output CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n✔ Completed! Metrics saved to {out_csv}\n")


if __name__ == "__main__":
    main()
