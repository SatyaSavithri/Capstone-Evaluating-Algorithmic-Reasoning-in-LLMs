# comparison_plots.py
"""
Utility to generate side-by-side comparison plots across models for a given graph & method.
Reads summary JSONs produced by advanced_metrics and plots bar charts with error bars.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ADV_ROOT = Path("results/advanced_metrics")

def gather_summaries(run_dir):
    summaries = list(Path(run_dir).glob("*_summary.json"))
    out = []
    for s in summaries:
        out.append(json.load(open(s)))
    return out

def plot_model_comparison(run_dir, graph, method, out_file):
    summaries = gather_summaries(run_dir)
    # filter
    filtered = [s for s in summaries if s["graph"] == graph and s["method"] == method]
    # group by model
    by_model = {}
    for s in filtered:
        m = s["model"]
        by_model.setdefault(m, []).append(s["accuracy"])
    models = list(by_model.keys())
    means = [np.mean(by_model[m]) for m in models]
    stds = [np.std(by_model[m]) for m in models]
    plt.figure(figsize=(6,4))
    plt.bar(models, means, yerr=stds, capsize=5)
    plt.ylabel("Accuracy")
    plt.title(f"Model comparison ({graph}, {method})")
    plt.ylim(0,1)
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to {out_file}")
