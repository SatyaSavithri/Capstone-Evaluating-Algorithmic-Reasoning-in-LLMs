# evaluation_runner.py
import os
import json
import csv
import difflib
import numpy as np

from GraphGenerator import generate_all_stimuli
from models_transformers import TransformersLLM
from attention_analysis import attention_to_room_ratio
from hybrid_runner import run_hybrid
from planner import bfs_optimal_path_to_max_reward


# -------------------------------------------------------
# --- Utility Functions for Metrics ---------------------
# -------------------------------------------------------

def normalize_path(text):
    """
    Extracts a path from model output:
    Example: "Room 1 -> Room 2 -> Room 5"
    """
    text = text.replace(",", " ").replace("->", " ").replace("  ", " ")
    parts = [p for p in text.split() if p.startswith("Room")]
    # Convert "Room 2" into consistent string
    return [" ".join(p.split()[:2]) for p in parts]


def path_accuracy(pred, gt):
    return int(pred == gt)


def path_edit_distance(pred, gt):
    sm = difflib.SequenceMatcher(None, pred, gt)
    return 1 - sm.ratio() 


def reward_regret(pred, gt, G):
    rewards = {n: G.nodes[n]["reward"] for n in G.nodes()}
    pred_reward = rewards[pred[-1]] if pred else -999
    gt_reward = rewards[gt[-1]]
    return gt_reward - pred_reward


def prefix_step_accuracy(pred, gt):
    correct = 0
    for p, g in zip(pred, gt):
        if p == g:
            correct += 1
        else:
            break
    return correct / len(gt)


# -------------------------------------------------------
# --- Attention Processing ------------------------------
# -------------------------------------------------------

def compute_attention_metric(model, prompt, positions_map):
    activations = model.generate_with_activations(prompt)
    all_layers = activations["attentions"]

    # Average over all heads, over all layers → (seq_len, seq_len)
    layer_avgs = []
    for layer in all_layers:
        h, q, k = layer.shape[1], layer.shape[2], layer.shape[3]
        avg_layer = layer.mean(dim=0)[0]  # (seq_len, seq_len)
        layer_avgs.append(avg_layer)

    avg_attn_matrix = sum(layer_avgs) / len(layer_avgs)
    attn = avg_attn_matrix.detach().cpu().numpy()

    return attention_to_room_ratio(attn, positions_map)


# -------------------------------------------------------
# --- Running a Single Method ---------------------------
# -------------------------------------------------------

def run_condition(model, stimulus, method_name="scratchpad"):

    if method_name == "valuePath":
        output = model.generate(stimulus["prompt"], max_new_tokens=150)

    elif method_name == "scratchpad":
        output = model.generate(stimulus["scratchpad_prompt"], max_new_tokens=200)

    elif method_name == "reval":
        # same input as valuePath; but graph contains updated reward version
        output = model.generate(stimulus["prompt"], max_new_tokens=150)

    else:
        raise ValueError(method_name)

    pred = normalize_path(output)
    gt = stimulus["ground_truth_path"]
    G = stimulus["raw_graph_data"]

    # --- Metrics ---
    metrics = {
        "pred_path": pred,
        "gt_path": gt,
        "accuracy": path_accuracy(pred, gt),
        "edit_distance": path_edit_distance(pred, gt),
        "reward_regret": reward_regret(pred, gt, G),
        "prefix_accuracy": prefix_step_accuracy(pred, gt),
    }

    return metrics, output


# -------------------------------------------------------
# --- Full Evaluation Loop ------------------------------
# -------------------------------------------------------

def run_all_evaluations(model_id="gpt2"):
    os.makedirs("results", exist_ok=True)

    print("Generating stimuli...")
    all_stimuli = generate_all_stimuli()
    model = TransformersLLM(model_id)

    results = []

    CONDITIONS = ["valuePath", "scratchpad", "rewardReval"]
    for key, stimulus in all_stimuli.items():
        graph_type = stimulus["graph_type"]
        task_type = stimulus["task_type"]

        print(f"\n--- Running {key} ---")

        for cond in CONDITIONS:

            if cond == "rewardReval" and task_type != "rewardReval":
                continue

            metrics, raw_output = run_condition(model, stimulus, 
                method_name=("reval" if cond=="rewardReval" else cond)
            )

            results.append({
                "graph": graph_type,
                "task": task_type,
                "condition": cond,
                **metrics,
                "raw_output": raw_output
            })

        # --- Hybrid evaluation ---
        hybrid = run_hybrid(model, stimulus["raw_graph_data"])
        chosen = hybrid["best"]
        gt = " -> ".join(stimulus["ground_truth_path"])
        hybrid_correct = int(chosen and (chosen.replace(" ", "") == gt.replace(" ", "")))

        results.append({
            "graph": graph_type,
            "task": task_type,
            "condition": "hybrid",
            "accuracy": hybrid_correct,
            "edit_distance": None,
            "reward_regret": None,
            "prefix_accuracy": None,
            "pred_path": chosen,
            "gt_path": stimulus["ground_truth_path"],
            "raw_output": hybrid["validator_text"]
        })


    # -------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------

    out_file = "results/metrics_results.csv"
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved evaluation results to {out_file}")


if __name__ == "__main__":
    run_all_evaluations("gpt2")
