# capture.py (update / patch - add these helper funcs and ensure meta["positions_map"] is saved)

# ... (keep your existing imports and helper functions)
from pathlib import Path
import json
import numpy as np

# assume RAW_DIR and FIG_DIR are already defined in your capture.py; if not define:
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Add this function near top of file
def build_positions_map_from_prompt(tokenizer, prompt_text, room_names):
    """
    Build a positions_map: room -> list of token indices in tokenized prompt.
    Strategy:
      - Use unique markers for rooms if present, otherwise search for exact 'Room 1' tokens.
      - Tokenize prompt and then find token spans that decode to each room string.
    Returns: { "Room 1": [3,4], ... }
    """
    positions = {r: [] for r in room_names}
    # token-level approach: tokenize and decode tokens incrementally and search substrings
    tokens = tokenizer(prompt_text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    # convert tokens to strings with tokenizer.convert_tokens_to_string on slices
    # naive approach: for each index i, attempt to decode tokens[i:i+max_len] and check contains room name
    max_window = 6
    token_text_cache = {}
    for i in range(len(tokens)):
        for w in range(1, max_window+1):
            j = i + w
            if j > len(tokens): break
            key = (i,j)
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
                    # mark all token indices i..j-1 for that room
                    positions[r].extend(list(range(i, j)))
    # deduplicate & sort
    positions = {r: sorted(list(set(v))) for r, v in positions.items()}
    return positions

# Then, inside your save_trial(...) before writing meta.json, add:

try:
    # only attempt if llm wrapper exposes tokenizer
    if hasattr(llm, "tokenizer"):
        positions_map = build_positions_map_from_prompt(llm.tokenizer, prompt, meta["room_names"])
    else:
        # fallback: approximate by token indices per room name if not tokenizable
        positions_map = {r: [] for r in meta["room_names"]}
except Exception:
    positions_map = {r: [] for r in meta["room_names"]}

meta["positions_map"] = positions_map

# Now save meta json as usual
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
