# attention_analysis.py
import numpy as np

def attention_to_room_ratio(attn_matrix, positions_map):
    """
    attn_matrix: (seq_len, seq_len) average attention
    positions_map: dict room -> list of token positions
    """
    all_room_positions = [p for v in positions_map.values() for p in v]
    if len(all_room_positions) == 0:
        return None
    seq_len = attn_matrix.shape[0]
    idx = min(seq_len-1, attn_matrix.shape[0]-1)
    mean_to_rooms = attn_matrix[idx, all_room_positions].mean()
    mean_to_all = attn_matrix[idx, :].mean()
    return float(mean_to_rooms / (mean_to_all + 1e-12))
