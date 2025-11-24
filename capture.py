from pathlib import Path
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define directories if not already defined
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

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
def save_trial(llm, prompt, meta, meta_path):
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary.")
    if "room_names" not in meta:
        raise KeyError("meta must contain 'room_names'.")

    # Build positions map
    try:
        if hasattr(llm, "tokenizer") and llm.tokenizer is not None:
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

# ---------------- Optimized Model Loader ----------------
def load_model(model_name: str, device: str = 'cuda'):
    """
    Loads the model efficiently with caching, dtype optimization, and automatic device mapping.
    """
    cache_dir = Path("./cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_name}... this may take a while the first time")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Use float16 for GPU if available to save memory and speed up
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map='auto' if torch.cuda.is_available() else None,
        torch_dtype=dtype
    )

    print(f"Model {model_name} loaded successfully")
    return model, tokenizer
