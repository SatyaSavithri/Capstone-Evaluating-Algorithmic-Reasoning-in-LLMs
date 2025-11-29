# attention_analysis.py
"""
Attention analysis helpers (final).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def attention_to_room_ratio(attn_matrix: np.ndarray, positions_map: dict):
    """
    attn_matrix: (seq, seq) numpy array
    positions_map: dict room -> list[int] token positions
    Returns float ratio of attention to room tokens vs overall
    """
    all_positions = [p for v in positions_map.values() for p in v]
    if len(all_positions) == 0:
        return None
    seq_len = attn_matrix.shape[0]
    idx = min(seq_len - 1, seq_len - 1)
    mean_to_rooms = float(np.mean(attn_matrix[idx, all_positions]))
    mean_to_all = float(np.mean(attn_matrix[idx, :]))
    if mean_to_all == 0:
        return None
    return float(mean_to_rooms / (mean_to_all + 1e-12))


def save_attention_heatmap_from_tensor(attn_matrix, tokens, save_path):
    """
    attn_matrix: (seq, seq) numpy array
    tokens: list of token strings (length seq)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap="viridis")
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
