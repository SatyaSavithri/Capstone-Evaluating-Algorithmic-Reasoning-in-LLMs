# attention_analysis.py
"""
Patched attention utilities: robust numpy conversion and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def attention_to_room_ratio(attn_matrix, positions_map):
    """
    attn_matrix: (seq, seq) numpy array or torch tensor
    positions_map: dict room -> list of token indices
    """
    att = _to_numpy(attn_matrix)
    all_room_positions = [p for v in positions_map.values() for p in v]
    if len(all_room_positions) == 0:
        return None
    seq_len = att.shape[0]
    idx = min(seq_len - 1, seq_len - 1)
    mean_to_rooms = float(att[idx, all_room_positions].mean())
    mean_to_all = float(att[idx, :].mean())
    if mean_to_all == 0:
        return None
    return float(mean_to_rooms / (mean_to_all + 1e-12))


def save_attention_heatmap_from_tensor(attn_matrix, tokens, save_path):
    """
    attn_matrix: (seq, seq) tensor/array
    tokens: list[str]
    save_path: filename
    """
    arr = _to_numpy(attn_matrix)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(arr, cmap="viridis")
    plt.title("Attention heatmap (avg heads)")
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    try:
        if len(tokens) <= 60:
            xticks = list(range(len(tokens)))
            plt.xticks(xticks, tokens, rotation=90, fontsize=6)
            plt.yticks(xticks, tokens, rotation=0, fontsize=6)
    except Exception:
        pass
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()
    return str(save_path)
