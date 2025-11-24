# analyze_trial.py
import os
import argparse
import numpy as np
import json
from pathlib import Path
import networkx as nx

import metrics as M

def load_trial_raw(raw_dir, trial_id):
    """
    This function expects common-naming raw files. Adapt your pipeline saving to match.
    - hidden: saved as npz with arrays per step or single array
    - attn: saved as npz or npy
    - metadata: meta json with 'graph', 'model', 'start', 'gold_path', 'pred_path', etc.
    """
    raw_dir = Path(raw_dir)
    hidden_path = raw_dir / f"{trial_id}_hidden.npz"
    attn_path = raw_dir / f"{trial_id}_attn.npz"
    meta_path = raw_dir / f"{trial_id}_meta.json"

    hidden = None
    attn = None
    meta = {}

    if hidden_path.exists():
        loaded = np.load(hidden_path, allow_pickle=True)
        # if saved as layers: steps x seq x dim or dict of arrays
        # Choose a convention used when capturing
        if 'arr_0' in loaded:
            hidden = loaded['arr_0']
        else:
            # try keys
            hidden = [loaded[k] for k in loaded.files]
    if attn_path.exists():
        loaded2 = np.load(attn_path, allow_pickle=True)
        if 'arr_0' in loaded2:
            attn = loaded2['arr_0']
        else:
            attn = [loaded2[k] for k in loaded2.files]
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return hidden, attn, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hidden, attn, meta = load_trial_raw(args.raw_dir, args.trial_id)
    if not meta:
        print("No meta file found for trial; aborting.")
        return

    # reconstruct graph (you can save networkx pickles in meta)
    # Here we assume meta contains serialized adjacency or graph_type
    graph = None
    if 'graph_type' in meta:
        from graphs import create_line_graph, create_tree_graph, create_clustered_graph
        if meta['graph_type'] == 'n7line':
            graph = create_line_graph()
        elif meta['graph_type'] == 'n7tree':
            graph = create_tree_graph()
        elif meta['graph_type'] == 'n15clustered':
            graph = create_clustered_graph()
    else:
        graph = nx.Graph()

    gold_path = meta.get("gold_path", [])
    pred_path = meta.get("pred_path", [])

    # Behavioral metrics
    acc = M.path_accuracy(pred_path, gold_path)
    ned = M.normalized_edit_distance(pred_path, gold_path)
    rdiff = M.reward_difference(graph, pred_path, gold_path)

    behav = {"accuracy": acc, "norm_edit": ned, "reward_diff": rdiff}
    M.plot_behavioral_summary(behav, save_path=str(out_dir / f"{args.trial_id}_behavior.png"))

    # RSA if hidden available
    if hidden is not None:
        # hidden is expected shape: steps x seq_len x dim (or adapt accordingly)
        steps = list(range(len(hidden)))
        rsa_values = []
        theo = None
        # build positions map from meta if present
        positions_map = meta.get("positions_map", {})  # room -> list of token indices
        rooms = list(positions_map.keys())
        if rooms:
            theo = M.theoretical_rsm_from_graph(graph, rooms)
            for t, hs in enumerate(hidden):
                # assume hs shape (seq_len, dim)
                embs, rooms_out = M.build_room_embeddings_from_hidden_states(hs, positions_map, method="mean")
                emp_rsm = M.rsm_from_embeddings(embs)
                r, p = M.rsa_correlation(emp_rsm, theo)
                rsa_values.append(r)
            M.plot_rsa_timeseries(steps, rsa_values, save_path=str(out_dir / f"{args.trial_id}_rsa.png"))

    # Attention
    if attn is not None and positions_map:
        # attn expected: steps x seq_len x seq_len  (or steps x layers x heads x seq x seq)
        # compute room-level attention per step (averaging across layers/heads if needed)
        room_labels = rooms
        attn_seq = []
        for t, a in enumerate(attn):
            # if a has shape (layers, heads, seq, seq) average across layers & heads
            if a.ndim == 4:
                avg = a.mean(axis=(0,1))  # seq x seq
            elif a.ndim == 3:
                avg = a.mean(axis=0)
            else:
                avg = a  # seq x seq
            # take last row (new token attention)
            last_row = avg[min(avg.shape[0]-1, 0), :] if avg.shape[0] > 0 else avg[0,:]
            # compute per-room attention
            per_room = [last_row[positions_map[room]].mean() if len(positions_map[room])>0 else 0.0 for room in room_labels]
            attn_seq.append(per_room)
        attn_arr = np.array(attn_seq)
        M.plot_attention_heatmap(attn_arr, room_labels, save_path=str(out_dir / f"{args.trial_id}_attn.png"))

    # Save summary metrics to JSON
    out_summary = {"trial_id": args.trial_id, "behav": behav}
    with open(out_dir.parent / "metrics" / f"{args.trial_id}_metrics.json", "w") as f:
        json.dump(out_summary, f, indent=2)
    print("Analysis complete. Outputs saved to", out_dir)

if __name__ == "__main__":
    main()
