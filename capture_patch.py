from pathlib import Path
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ---------------- Helper Functions ----------------
def build_positions_map_from_prompt(tokenizer, prompt_text, room_names):
    positions = {r: [] for r in room_names}
    tokens = tokenizer(prompt_text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    max_window = 6
    token_text_cache = {}
    for i in range(len(tokens)):
        for w in range(1, max_window + 1):
            j = i + w
            if j > len(tokens):
                break
            key = (i, j)
            if key in token_text_cache:
                s = token_text_cache[key]
            else:
                try:
                    s = tokenizer.decode(tokens[i:j], clean_up_tokenization_spaces=True)
                except Exception:
                    s = ""
                token_text_cache[key] = s
            for r in room_names:
                if r in s:
                    positions[r].extend(list(range(i, j)))
    positions = {r: sorted(list(set(v))) for r, v in positions.items()}
    return positions

# ---------------- Main Trial Saver ----------------
def save_trial(llm, prompt, meta, meta_path, G=None, model_id=None, graph_name=None, method_name=None, start_node=None, trial_id=None, max_new_tokens=40, save_attn_and_hidden=False):
    """
    Saves trial metadata, optionally with graph info and trial parameters.
    """
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary.")
    if "room_names" not in meta:
        raise KeyError("meta must contain 'room_names'.")

    # Include trial info in meta
    if model_id: meta["model_id"] = model_id
    if graph_name: meta["graph_name"] = graph_name
    if method_name: meta["method_name"] = method_name
    if start_node: meta["start_node"] = start_node
    if trial_id: meta["trial_id"] = trial_id

    # Include graph node info
    if G is not None:
        meta["graph_nodes"] = list(G.nodes())

    # Build positions map
    try:
        if hasattr(llm, "tokenizer") and llm.tokenizer is not None and prompt is not None:
            positions_map = build_positions_map_from_prompt(llm.tokenizer, prompt, meta["room_names"])
        else:
            positions_map = {r: [] for r in meta["room_names"]}
    except Exception as e:
        print(f"Warning: failed to build positions_map: {e}")
        positions_map = {r: [] for r in meta["room_names"]}

    meta["positions_map"] = positions_map

    # Save meta.json
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved trial metadata to {meta_path}")
