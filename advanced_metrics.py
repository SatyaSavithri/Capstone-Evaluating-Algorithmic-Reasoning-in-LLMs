# advanced_metrics.py
"""
advanced_metrics.py
Advanced RSA / Attention / Behavioral analysis and PDF report generator.

Usage (interactive):
    python advanced_metrics.py

Usage (non-interactive):
    python advanced_metrics.py --graph n7line --models phi-3-mini,mistral --methods Scratchpad,DynamicRSA

Outputs:
    - results/advanced_metrics/<timestamp>/report_<graph>_<models>_<methods>.pdf
    - results/advanced_metrics/<timestamp>/figs/*.png
    - results/advanced_metrics/<timestamp>/metrics_summary.csv
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import List, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from scipy.stats import spearmanr
from metrics import rsm_from_embeddings, theoretical_rsm_from_graph, rsa_correlation, permutation_test_rsa, build_room_embeddings_from_hidden_states, attention_to_room_ratio, attention_entropy, next_step_attention_predictivity, plot_rsa_timeseries, plot_attention_heatmap, plot_behavioral_summary
# metrics.py functions assumed available as implemented earlier

RAW_DIR = Path("data/raw")
OUT_ROOT = Path("results/advanced_metrics")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

def list_all_meta():
    metas = []
    for f in RAW_DIR.glob("*_meta.json"):
        with open(f, "r") as fh:
            metas.append(json.load(fh))
    return metas

def filter_trials(graph=None, models:List[str]=None, methods:List[str]=None):
    all_meta = list_all_meta()
    out = []
    for m in all_meta:
        if graph and m.get("graph_name") != graph: continue
        if models and (m.get("model_short") not in models): continue
        if methods and (m.get("method") not in methods): continue
        out.append(m)
    return sorted(out, key=lambda x: (x["model_short"], x["graph_name"], x["method"], x["trial_id"]))

def load_npz_dict(path:Path):
    # returns dict of arrays
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    out = {}
    for k in data.files:
        out[k] = data[k]
    return out

def analyze_single_trial(meta, out_dir:Path):
    trial_id = meta["trial_id"]
    print(f"[analyze] {trial_id}")
    hidden_path = RAW_DIR / f"{trial_id}_hidden.npz"
    attn_path = RAW_DIR / f"{trial_id}_attn.npz"

    hidden_dict = load_npz_dict(hidden_path)
    attn_dict = load_npz_dict(attn_path)

    rooms = meta.get("room_names", [])
    positions_map = meta.get("positions_map", {r: [] for r in rooms})

    # Reconstruct graph object
    graph_name = meta.get("graph_name")
    from graphs import create_line_graph, create_tree_graph, create_clustered_graph
    if graph_name == "n7line":
        G = create_line_graph()
    elif graph_name == "n7tree":
        G = create_tree_graph()
    else:
        G = create_clustered_graph()

    symbolic = meta.get("symbolic_path", [])
    pred = meta.get("pred_path", [])

    # Behavioral metrics
    acc = int(pred == symbolic)
    ned = 0.0
    if symbolic:
        # normalized edit distance via simple function
        from metrics import normalized_edit_distance
        ned = normalized_edit_distance(pred, symbolic)

    # Save behavioral plot
    fig_behav = out_dir / f"{trial_id}_behavior.png"
    plot_behavioral_summary({"accuracy": acc, "norm_edit": ned, "reward_diff": 0.0}, title=f"{trial_id} Behavioral", save_path=str(fig_behav))

    layer_rsa_summary = []
    rsa_time_series = {}
    # Hidden states: keys layer_0 .. layer_N each is (seq_len, hidden_dim)
    # assume hidden_dict has 'layer_0', 'layer_1' etc where each is array shape (seq_len, dim)
    if hidden_dict:
        # For each layer, compute RSA across rooms using the pooled embeddings (rooms x dim)
        for k in sorted(hidden_dict.keys(), key=lambda x: int(x.split('_')[1])):
            layer_arr = hidden_dict[k]  # (seq_len, dim)
            # Build room embeddings from hidden states for this single forward snapshot
            embs, room_order = build_room_embeddings_from_hidden_states(layer_arr, positions_map, method="mean")
            if embs.shape[0] != len(room_order):
                # fallback: skip
                continue
            emp_rsm = rsm_from_embeddings(embs)
            theo = theoretical_rsm_from_graph(G, room_order)
            r, p = rsa_correlation(emp_rsm, theo)
            # permutation test (smaller n for speed)
            obs_r, perm_p = permutation_test_rsa(emp_rsm, theo, n_perm=200, seed=42)
            layer_rsa_summary.append({"layer": k, "rho": r, "p_perm": perm_p})
        # Also compute time-series RSA if we have per-step hidden snapshots; if you saved hidden states per step, adapt here.
        # For many wrappers hidden_dict contains one snapshot (final). RSA timeseries requires capturing per-step hidden states; if available adapt.

    # Attention analysis: attn_dict keys layer_0 etc, where arr is (heads, seq, seq)
    attn_plots = []
    if attn_dict:
        for k in sorted(attn_dict.keys(), key=lambda x: int(x.split('_')[1])):
            arr = attn_dict[k]  # (heads, seq, seq)
            # average across heads for room-level heatmap
            avg = arr.mean(axis=0)  # seq x seq
            # compute per-room attention for last row (or a chosen token index)
            seq_len = avg.shape[0]
            last_idx = seq_len - 1
            room_attn = []
            for r in rooms:
                pos = positions_map.get(r, [])
                if pos:
                    room_attn.append(avg[last_idx, pos].mean())
                else:
                    room_attn.append(0.0)
            # save small bar plot or heatmap per layer
            fig_attn = out_dir / f"{trial_id}_attn_{k}.png"
            plot_attention_heatmap(np.array([room_attn]), rooms, title=f"{trial_id} Attn {k}", save_path=str(fig_attn))
            attn_plots.append(str(fig_attn))

    # Save per-trial summary json
    summary = {
        "trial_id": trial_id,
        "model": meta.get("model_short"),
        "graph": graph_name,
        "method": meta.get("method"),
        "accuracy": acc,
        "norm_edit": ned,
        "rsa_layer_summary": layer_rsa_summary,
        "attn_plots": attn_plots,
        "behavior_plot": str(fig_behav)
    }
    with open(out_dir / f"{trial_id}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary

def generate_report(trials_meta, out_root:Path, report_name:str):
    ts = int(time.time())
    od = out_root / f"run_{ts}"
    od.mkdir(parents=True, exist_ok=True)
    figs_dir = od / "figs"
    figs_dir.mkdir(exist_ok=True)
    summaries = []
    # analyze each trial and collect summaries
    for meta in trials_meta:
        s = analyze_single_trial(meta, figs_dir)
        summaries.append(s)

    # Generate PDF report with key images
    pdf_path = od / report_name
    with PdfPages(pdf_path) as pdf:
        # cover page
        plt.figure(figsize=(8.5,11))
        plt.text(0.5, 0.8, f"Advanced Metrics Report", ha="center", fontsize=20)
        plt.text(0.5, 0.75, f"Generated: {time.ctime()}", ha="center", fontsize=10)
        plt.axis("off")
        pdf.savefig(); plt.close()

        # include each trial behavior + attn images if exist
        for s in summaries:
            # behavior
            bp = s.get("behavior_plot")
            if bp and Path(bp).exists():
                img = plt.imread(bp)
                plt.figure(figsize=(8,4))
                plt.imshow(img); plt.axis("off")
                pdf.savefig(); plt.close()
            # attention plots
            for ap in s.get("attn_plots", []):
                if Path(ap).exists():
                    img = plt.imread(ap)
                    plt.figure(figsize=(8,4))
                    plt.imshow(img); plt.axis("off")
                    pdf.savefig(); plt.close()

    # also save CSV/TSV summary
    import csv
    csv_path = od / "metrics_summary.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = ["trial_id","model","graph","method","accuracy","norm_edit"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({
                "trial_id": s["trial_id"],
                "model": s["model"],
                "graph": s["graph"],
                "method": s["method"],
                "accuracy": s["accuracy"],
                "norm_edit": s["norm_edit"]
            })

    print(f"[advanced_metrics] Report saved to: {pdf_path}")
    print(f"[advanced_metrics] Summary CSV: {csv_path}")
    return od

def main_interactive():
    metas = list_all_meta()
    graphs = sorted(set([m["graph_name"] for m in metas]))
    models = sorted(set([m["model_short"] for m in metas]))
    methods = sorted(set([m["method"] for m in metas]))

    # simple interactive choices
    print("Available graphs:", graphs)
    graph = input("Graph to analyze (e.g., n7line): ").strip()
    print("Available models:", models)
    model_input = input("Model(s) to include (comma separated model_short, or 'all'): ").strip()
    if model_input.lower() == "all":
        sel_models = models
    else:
        sel_models = [m.strip() for m in model_input.split(",")]

    method_input = input("Method(s) to include (comma separated, or 'all'): ").strip()
    if method_input.lower() == "all":
        sel_methods = methods
    else:
        sel_methods = [s.strip() for s in method_input.split(",")]

    trials = filter_trials(graph=graph, models=sel_models, methods=sel_methods)
    print(f"Found {len(trials)} trials.")

    if not trials:
        print("No trials found. Run experiments first.")
        return

    report_name = f"report_{graph}_{'_'.join(sel_models)}_{'_'.join(sel_methods)}.pdf"
    od = generate_report(trials, OUT_ROOT, report_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=None)
    parser.add_argument("--models", type=str, default=None, help="comma separated model_short")
    parser.add_argument("--methods", type=str, default=None, help="comma separated methods")
    args = parser.parse_args()
    if args.graph is None:
        main_interactive()
    else:
        # non-interactive mode
        models = args.models.split(",") if args.models else None
        methods = args.methods.split(",") if args.methods else None
        trials = filter_trials(graph=args.graph, models=models, methods=methods)
        report_name = f"report_{args.graph}_{'_'.join(models) if models else 'all'}_{'_'.join(methods) if methods else 'all'}.pdf"
        generate_report(trials, OUT_ROOT, report_name)
