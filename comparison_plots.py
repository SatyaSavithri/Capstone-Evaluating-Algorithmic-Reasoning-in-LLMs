# ================================================================
# comparison_plots.py (Interactive, Professional Version)
# ================================================================

import os
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Optional for inline display
try:
    from IPython.display import Image, display
    JUPYTER = True
except:
    JUPYTER = False


# ================================================================
# Utility: Load summary file
# ================================================================
def load_summary(summary_path):
    with open(summary_path, "r") as f:
        return json.load(f)


# ================================================================
# Utility: Plot helper
# ================================================================
def save_and_show(fig, save_path):
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {save_path}")

    if JUPYTER:
        display(Image(filename=save_path))


# ================================================================
# Plot: RSA correlation comparison
# ================================================================
def plot_rsa_comparison(summaries, save_dir, graph, method):
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = []
    rsa_values = []

    for model, content in summaries.items():
        if graph in content and method in content[graph]:
            rsa_values.append(content[graph][method]["rsa"])
            model_names.append(model)

    ax.bar(model_names, rsa_values, color="skyblue")
    ax.set_title(f"RSA Comparison – {graph} – {method}", fontsize=14)
    ax.set_ylabel("RSA Correlation", fontsize=12)
    ax.set_ylim(0, 1)

    save_path = os.path.join(save_dir, f"RSA_{graph}_{method}.png")
    save_and_show(fig, save_path)


# ================================================================
# Plot: Attention-to-room ratio comparison
# ================================================================
def plot_attention_comparison(summaries, save_dir, graph, method):
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = []
    attn_values = []

    for model, content in summaries.items():
        if graph in content and method in content[graph]:
            attn_values.append(content[graph][method]["attention_ratio"])
            model_names.append(model)

    ax.bar(model_names, attn_values, color="lightgreen")
    ax.set_title(f"Attention Ratio Comparison – {graph} – {method}", fontsize=14)
    ax.set_ylabel("Attention Ratio", fontsize=12)

    save_path = os.path.join(save_dir, f"ATTN_{graph}_{method}.png")
    save_and_show(fig, save_path)


# ================================================================
# Plot: Path Accuracy comparison
# ================================================================
def plot_path_accuracy_comparison(summaries, save_dir, graph, method):
    fig, ax = plt.subplots(figsize=(8, 5))

    model_names = []
    acc_values = []

    for model, content in summaries.items():
        if graph in content and method in content[graph]:
            acc_values.append(content[graph][method]["path_accuracy"])
            model_names.append(model)

    ax.bar(model_names, acc_values, color="salmon")
    ax.set_title(f"Path Accuracy Comparison – {graph} – {method}", fontsize=14)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1)

    save_path = os.path.join(save_dir, f"ACCURACY_{graph}_{method}.png")
    save_and_show(fig, save_path)


# ================================================================
# Interactive driver
# ================================================================
def main():
    print("\n=== INTERACTIVE MODEL COMPARISON TOOL ===\n")

    # ------------------------------------------------------------
    # Step 1: Detect all available metric runs
    # ------------------------------------------------------------
    runs = sorted(glob.glob("results/advanced_metrics/run_*"))

    if not runs:
        print("❌ No metric runs found. Run advanced_metrics.py first.")
        return

    print("Available metric runs:")
    for i, r in enumerate(runs):
        print(f"{i+1}) {r}")

    choice = int(input("\nSelect run: ").strip())
    run_dir = runs[choice - 1]
    print(f"\nSelected run: {run_dir}\n")

    summary_files = glob.glob(os.path.join(run_dir, "*_summary.json"))
    summaries = {}

    for sf in summary_files:
        model_name = Path(sf).stem.replace("_summary", "")
        summaries[model_name] = load_summary(sf)

    # ------------------------------------------------------------
    # Step 2: Select graph
    # ------------------------------------------------------------
    graphs = ["n7line", "n7tree", "n15clustered"]
    print("\nAvailable graphs:")
    for i, g in enumerate(graphs):
        print(f"{i+1}) {g}")
    g_choice = int(input("\nSelect graph: ").strip())
    selected_graph = graphs[g_choice - 1]

    # ------------------------------------------------------------
    # Step 3: Select method
    # ------------------------------------------------------------
    methods = ["Scratchpad", "Hybrid", "DynamicRSA", "Attention"]
    print("\nAvailable methods:")
    for i, m in enumerate(methods):
        print(f"{i+1}) {m}")
    m_choice = int(input("\nSelect method: ").strip())
    selected_method = methods[m_choice - 1]

    # ------------------------------------------------------------
    # Step 4: Select metric (RSA, Attention, Accuracy, All)
    # ------------------------------------------------------------
    metrics = ["RSA", "Attention", "PathAccuracy", "ALL"]
    print("\nMetrics:")
    for i, m in enumerate(metrics):
        print(f"{i+1}) {m}")
    mt_choice = int(input("\nSelect metric: ").strip())
    selected_metric = metrics[mt_choice - 1]

    # ------------------------------------------------------------
    # Step 5: Create output directory
    # ------------------------------------------------------------
    out_dir = os.path.join(run_dir, "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Step 6: Generate selected plots
    # ------------------------------------------------------------
    if selected_metric in ["RSA", "ALL"]:
        plot_rsa_comparison(summaries, out_dir, selected_graph, selected_method)

    if selected_metric in ["Attention", "ALL"]:
        plot_attention_comparison(summaries, out_dir, selected_graph, selected_method)

    if selected_metric in ["PathAccuracy", "ALL"]:
        plot_path_accuracy_comparison(summaries, out_dir, selected_graph, selected_method)

    print("\n=== Comparison Plots Generated Successfully ===\n")


# ================================================================
# Run if executed directly
# ================================================================
if __name__ == "__main__":
    main()
