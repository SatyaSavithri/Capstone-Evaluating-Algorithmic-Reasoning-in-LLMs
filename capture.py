# capture.py
"""
capture.py

Utility functions to run one trial and save raw data in the canonical format:
- data/raw/<trial_id>_hidden.npz
- data/raw/<trial_id>_attn.npz
- data/raw/<trial_id>_meta.json
- results/figures/<trial_id>_graph.png

Assumes TransformersLLM has methods:
- generate(prompt, max_new_tokens=..., temperature=...)
- generate_with_activations(prompt, max_new_tokens=...) -> returns dict with keys:
    "prompt", "hidden_states", "attentions", "model_output"
"""

import os
import time
import json
from pathlib import Path
import numpy as np
import networkx as nx

# adapt imports to your repo layout
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt
from visualize import draw_graph as old_draw_graph  # optional fallback

RAW_DIR = Path("data/raw")
FIG_DIR = Path("results/figures")
RAW_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _tensor_tuple_to_numpy(hidden_states):
    """
    Convert hidden_states (tuple of torch tensors) to numpy arrays.
    Expects each layer tensor shape: (batch, seq_len, hidden_dim)
    Returns dict: { "layer0": array (seq_len, hidden_dim), ... }
    """
    import torch
    out = {}
    for i, layer in enumerate(hidden_states):
        # convert to cpu numpy; handle tuple/list
        if hasattr(layer, "detach"):
            arr = layer.detach().cpu().numpy()
        else:
            # sometimes hidden_states might be tuples of numpy already
            arr = np.array(layer)
        # if batch dim present, take batch 0
        if arr.ndim == 3:
            arr = arr[0]  # seq_len x hidden_dim
        out[f"layer_{i}"] = arr
    return out


def _attn_tuple_to_numpy(attentions):
    """
    Convert attentions (tuple of torch tensors) to numpy arrays.
    Expects per-layer tensor shapes: (batch, num_heads, seq_len, seq_len)
    Returns dict: { "layer0": array (num_heads, seq_len, seq_len), ... }
    """
    import torch
    out = {}
    for i, layer in enumerate(attentions):
        if hasattr(layer, "detach"):
            arr = layer.detach().cpu().numpy()
        else:
            arr = np.array(layer)
        # take batch 0 if batch dim exists
        if arr.ndim == 4:
            arr = arr[0]  # heads x seq x seq
        out[f"layer_{i}"] = arr
    return out


def save_npz_dict(np_dict, path: Path):
    """Save a dict of numpy arrays to a compressed npz (preserves keys)."""
    np.savez_compressed(path, **np_dict)


def default_graph_image_save(G, symbolic_path, trial_id):
    """
    Save a graph image to FIG_DIR/<trial_id>_graph.png
    Uses minimal plotting (networkx + matplotlib).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color="lightblue", node_size=900)
    if symbolic_path and len(symbolic_path) > 1:
        edges = list(zip(symbolic_path, symbolic_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=3, edge_color='red')
    plt.title(f"{trial_id} Graph")
    out = FIG_DIR / f"{trial_id}_graph.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    return str(out)


def save_trial(llm, G, model_id: str, graph_name: str, method_name: str,
               start_node: str, trial_id: str, max_new_tokens: int = 20,
               temperature: float = 0.0, save_attn_and_hidden: bool = True):
    """
    Run one trial and save raw data.

    Parameters:
    - llm: TransformersLLM instance
    - G: networkx graph
    - model_id: string id
    - graph_name: "n7line" / "n7tree" / ...
    - method_name: "Scratchpad"/"Hybrid"/"DynamicRSA"/"Attention"
    - start_node: e.g., "Room 3"
    - trial_id: unique string used for filenames
    - save_attn_and_hidden: True captures activations (may be heavy)
    """
    # build description and prompt
    desc = base_description_text(G)
    prompt = scratchpad_prompt(desc, "valuePath")

    # symbolic planner (ground truth)
    symbolic_path = bfs_optimal_path_to_max_reward(G, start_node)

    # prepare the result container
    meta = {
        "trial_id": trial_id,
        "model_id": model_id,
        "model_short": model_id.split("/")[-1],
        "graph_name": graph_name,
        "start_node": start_node,
        "method": method_name,
        "timestamp": int(time.time()),
        "symbolic_path": symbolic_path,
        "prompt": prompt,
        "room_names": list(G.nodes()),  # allow metrics to build positions_map from prompt/text
    }

    print(f"[capture] Trial {trial_id}: model={model_id} graph={graph_name} start={start_node} method={method_name}")

    # run depending on method
    if method_name in ("DynamicRSA", "Attention"):
        # capture activations
        data = llm.generate_with_activations(prompt, max_new_tokens=max_new_tokens)
        # store textual model output if available
        meta["llm_output"] = data.get("model_output_text", "") if isinstance(data.get("model_output_text", ""), str) else ""
        # many model wrappers return raw outputs in 'model_output'
        # convert hidden & attentions to numpy dicts
        hidden = data.get("hidden_states", None)
        attn = data.get("attentions", None)
        if save_attn_and_hidden:
            if hidden is not None:
                hidden_np = _tensor_tuple_to_numpy(hidden)
                save_npz_dict(hidden_np, RAW_DIR / f"{trial_id}_hidden.npz")
            if attn is not None:
                attn_np = _attn_tuple_to_numpy(attn)
                save_npz_dict(attn_np, RAW_DIR / f"{trial_id}_attn.npz")
    else:
        # scratchpad/hybrid: just generate text (faster)
        llm_text = llm.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        meta["llm_output"] = llm_text
        # For consistency, still call generate_with_activations with use_cache=False if the user wants to keep activations
        # but by default we don't for speed.

    # attempt to parse predicted path (best-effort). If you have utils.parse_final_json_path, use it.
    try:
        # avoid heavy imports at top-level
        from utils import parse_final_json_path
        pred_path = parse_final_json_path(meta.get("llm_output", ""))
    except Exception:
        pred_path = []

    meta["pred_path"] = pred_path

    # save metadata JSON
    meta_path = RAW_DIR / f"{trial_id}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # save graph image
    graph_img_path = default_graph_image_save(G, symbolic_path, trial_id)
    print(f"[capture] Saved graph image to {graph_img_path}")
    print(f"[capture] Saved meta to {meta_path}")

    return {
        "trial_id": trial_id,
        "meta_path": str(meta_path),
        "hidden_path": str(RAW_DIR / f"{trial_id}_hidden.npz") if (RAW_DIR / f"{trial_id}_hidden.npz").exists() else None,
        "attn_path": str(RAW_DIR / f"{trial_id}_attn.npz") if (RAW_DIR / f"{trial_id}_attn.npz").exists() else None,
        "graph_image": graph_img_path
    }
