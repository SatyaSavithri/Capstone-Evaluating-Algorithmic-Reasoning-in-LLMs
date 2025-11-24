from pathlib import Path
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Optional directory used by experiments
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Helper Functions ----------------
def build_positions_map_from_prompt(tokenizer, prompt_text, room_names, max_window: int = 6):
    """
    Build positions_map: mapping room_name -> list of token indices in the tokenized prompt.
    Uses a sliding window decode approach (robust, but not super-fast).
    """
    positions = {r: [] for r in room_names}
    if prompt_text is None or tokenizer is None:
        return positions

    # Tokenize once; use numpy tensors if tokenizer supports it.
    try:
        toks = tokenizer(prompt_text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    except Exception:
        # Fallback to pytorch tensors if needed
        toks = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].cpu().numpy()

    token_text_cache = {}
    n_tokens = len(toks)

    for i in range(n_tokens):
        for w in range(1, max_window + 1):
            j = i + w
            if j > n_tokens:
                break
            key = (i, j)
            if key in token_text_cache:
                s = token_text_cache[key]
            else:
                try:
                    s = tokenizer.decode(toks[i:j], clean_up_tokenization_spaces=True)
                except Exception:
                    s = ""
                token_text_cache[key] = s
            for r in room_names:
                if r in s:
                    positions[r].extend(list(range(i, j)))

    # dedupe & sort indices
    positions = {r: sorted(set(idx_list)) for r, idx_list in positions.items()}
    return positions

# ---------------- Main Trial Saver ----------------
def save_trial(
    llm,
    prompt,
    meta,
    meta_path,
    G=None,
    model_id=None,
    graph_name=None,
    method_name=None,
    start_node=None,
    trial_id=None,
    max_new_tokens: int = 40,
    save_attn_and_hidden: bool = False,
):
    """
    Save trial metadata to meta_path (JSON). Augments meta with:
      - model_id, graph_name, method_name, start_node, trial_id
      - graph_nodes (if G provided)
      - positions_map (if tokenizer available and prompt provided)
    llm: wrapper with .tokenizer attribute (optional)
    prompt: original prompt string used for the trial (optional)
    meta: dict, must include 'room_names' key
    meta_path: path to write meta JSON
    """
    # Validate meta
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dictionary.")
    if "room_names" not in meta:
        raise KeyError("meta must contain 'room_names'.")

    # Add trial params into meta where provided
    if model_id:
        meta["model_id"] = model_id
    if graph_name:
        meta["graph_name"] = graph_name
    if method_name:
        meta["method_name"] = method_name
    if start_node:
        meta["start_node"] = start_node
    if trial_id:
        meta["trial_id"] = trial_id
    meta["max_new_tokens"] = int(max_new_tokens)
    meta["save_attn_and_hidden"] = bool(save_attn_and_hidden)

    # Graph info
    if G is not None:
        try:
            meta["graph_nodes"] = list(G.nodes())
            # optionally store edges if desired:
            meta["graph_edges"] = [list(e) for e in G.edges()]
        except Exception:
            # safe fallback if G is not a networkx graph
            pass

    # Build positions map (best-effort)
    try:
        if hasattr(llm, "tokenizer") and llm.tokenizer is not None and prompt is not None:
            positions_map = build_positions_map_from_prompt(llm.tokenizer, prompt, meta["room_names"])
        else:
            positions_map = {r: [] for r in meta["room_names"]}
    except Exception as e:
        print(f"[capture_patch] Warning: failed to build positions_map: {e}")
        positions_map = {r: [] for r in meta["room_names"]}

    meta["positions_map"] = positions_map

    # Ensure parent dir and write JSON
    meta_path = Path(meta_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[capture_patch] Saved trial metadata to {meta_path}")

# ---------------- Optimized Model Loader ----------------
def load_model(model_name: str, device: str = "cuda"):
    """
    Loads tokenizer + model using huggingface transformers with caching.
    Returns: (model, tokenizer)
    Notes:
      - Uses float16 when CUDA is available to reduce memory.
      - Uses device_map='auto' to place weights on GPU(s) automatically.
    """
    cache_dir = Path("./cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[capture_patch] Loading model {model_name}... this may take a while the first time")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype,
    )

    print(f"[capture_patch] Model {model_name} loaded successfully")
    return model, tokenizer
PY
