# capture.py (update / patch - add these helper funcs and ensure meta["positions_map"] is saved)

from pathlib import Path
import json
import numpy as np

# Define directories if not already defined
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helper Functions ----------------
def build_positions_map_from_prompt(tokenizer, prompt_text, room_names):
    """
    Build a positions_map: room -> list of token indices in tokenized prompt.
    """
    positions = {r: [] for r in room_names}

    # Tokenize prompt
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

    # Deduplicate & sort
    positions = {r: sorted(list(set(v))) for r, v in positions.items()}
    return positions

# ---------------- Main Trial Saver ----------------
def save_trial(llm, prompt, meta, meta_path):
    """
    Saves the trial metadata including positions_map.
    llm: Language model wrapper exposing tokenizer
    prompt: str, the text prompt for the model
    meta: dict, trial metadata including room_names
    meta_path: Path to save meta.json
    """
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
