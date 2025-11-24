# metrics.py
import os
import json
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from typing import List, Dict, Tuple, Optional
import itertools
import random

# -----------------------------
# Behavioral: Path comparisons
# -----------------------------
def path_accuracy(pred_path: List[str], gold_path: List[str]) -> int:
    """Exact match: 1 if identical, else 0"""
    return int(pred_path == gold_path)

def levenshtein_distance(a: List[str], b: List[str]) -> int:
    """Classic edit distance between two sequences."""
    # dynamic programming
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,    # deletion
                           dp[i][j-1] + 1,    # insertion
                           dp[i-1][j-1] + cost)  # substitution
    return dp[n][m]

def normalized_edit_distance(pred_path: List[str], gold_path: List[str]) -> float:
    d = levenshtein_distance(pred_path, gold_path)
    denom = max(len(pred_path), len(gold_path), 1)
    return d / denom

def total_reward_of_path(G: nx.Graph, path: List[str]) -> float:
    rewards = nx.get_node_attributes(G, "reward")
    return sum(rewards.get(n, 0) for n in path)

def reward_difference(G: nx.Graph, pred_path: List[str], gold_path: List[str]) -> float:
    return total_reward_of_path(G, pred_path) - total_reward_of_path(G, gold_path)

# -----------------------------
# RSA / Representational metrics
# -----------------------------
def rsm_from_embeddings(embs: np.ndarray) -> np.ndarray:
    """
    embs: (n_items, dim)
    returns similarity matrix (n_items, n_items) using cosine similarity.
    """
    if embs.shape[0] < 2:
        return np.eye(embs.shape[0])
    d = pdist(embs, metric="cosine")
    sim = 1 - squareform(d)
    sim = np.nan_to_num(sim)
    return sim

def theoretical_rsm_from_graph(G: nx.Graph, rooms: List[str]) -> np.ndarray:
    n = len(rooms)
    mat = np.zeros((n, n), dtype=float)
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms):
            if a == b:
                mat[i, j] = 1.0
            else:
                try:
                    d = nx.shortest_path_length(G, source=a, target=b)
                    mat[i, j] = 1.0 / (1.0 + d)
                except Exception:
                    mat[i, j] = 0.0
    return mat

def rsa_correlation(empirical_rsm: np.ndarray, theoretical_rsm: np.ndarray) -> Tuple[float, float]:
    iu = np.triu_indices_from(empirical_rsm, k=1)
    e = empirical_rsm[iu]
    t = theoretical_rsm[iu]
    r, p = spearmanr(e, t)
    return float(r), float(p if p is not None else np.nan)

def permutation_test_rsa(empirical_rsm: np.ndarray, theoretical_rsm: np.ndarray, n_perm: int = 1000, seed: Optional[int] = None) -> Tuple[float, float]:
    """Return (observed_r, p_value) where p_value is fraction of perms >= observed."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    observed_r, _ = rsa_correlation(empirical_rsm, theoretical_rsm)
    n = empirical_rsm.shape[0]
    iu = np.triu_indices_from(empirical_rsm, k=1)
    emp_vals = empirical_rsm[iu]
    theo_vals = theoretical_rsm[iu]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        perm_mat = theoretical_rsm[perm][:, perm]
        perm_vals = perm_mat[iu]
        r, _ = spearmanr(emp_vals, perm_vals)
        if r >= observed_r:
            count += 1
    pval = (count + 1) / (n_perm + 1)
    return observed_r, pval

# -----------------------------
# Attention metrics
# -----------------------------
def attention_to_room_ratio(attn_matrix: np.ndarray, positions_map: Dict[str, List[int]]) -> Optional[float]:
    """
    attn_matrix: (seq_len, seq_len) average attention for generating token t (or per head averaged)
    positions_map: room -> list of token positions for that room
    """
    all_room_positions = [p for v in positions_map.values() for p in v]
    if len(all_room_positions) == 0:
        return None
    seq_len = attn_matrix.shape[0]
    # use last row (attention from new token to inputs), but if attn_matrix is (heads, seq, seq) handle outside
    last_row = attn_matrix[min(seq_len - 1, attn_matrix.shape[0] - 1), :]
    mean_to_rooms = last_row[all_room_positions].mean()
    mean_to_all = last_row.mean()
    return float(mean_to_rooms / (mean_to_all + 1e-12))

def attention_entropy(attn_vector: np.ndarray) -> float:
    """Entropy of attention distribution (single vector)."""
    a = attn_vector + 1e-12
    a = a / a.sum()
    return -float((a * np.log(a)).sum())

def next_step_attention_predictivity(attn_vector: np.ndarray, positions_map: Dict[str, List[int]], next_room: str) -> bool:
    """Return True if the highest-attended room corresponds to next_room"""
    room_means = {}
    for room, pos in positions_map.items():
        if len(pos) == 0:
            room_means[room] = 0.0
        else:
            room_means[room] = float(attn_vector[pos].mean())
    predicted = max(room_means.items(), key=lambda x: x[1])[0]
    return predicted == next_room

# -----------------------------
# Plotting utilities
# -----------------------------
def plot_rsa_timeseries(steps: List[int], rsa_values: List[float], ci_lower: Optional[List[float]] = None, ci_upper: Optional[List[float]] = None, title: str = "RSA over generation steps", save_path: Optional[str] = None):
    plt.figure(figsize=(7,4))
    plt.plot(steps, rsa_values, marker='o', label='RSA (Spearman)')
    if ci_lower is not None and ci_upper is not None:
        plt.fill_between(steps, ci_lower, ci_upper, alpha=0.2, label='95% CI')
    plt.xlabel("Generation step (token index)")
    plt.ylabel("RSA correlation (Spearman rho)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved RSA timeseries to {save_path}")
    plt.close()

def plot_attention_heatmap(attn_seq: np.ndarray, room_labels: List[str], title: str = "Attention to rooms over time", save_path: Optional[str] = None):
    """
    attn_seq: (time_steps, n_rooms) average attention to each room at each step
    """
    plt.figure(figsize=(8, 5))
    plt.imshow(attn_seq, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label='Attention')
    plt.yticks(range(attn_seq.shape[0]), [f"t={i}" for i in range(attn_seq.shape[0])])
    plt.xticks(range(len(room_labels)), room_labels, rotation=45)
    plt.xlabel("Room")
    plt.ylabel("Generation step")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved attention heatmap to {save_path}")
    plt.close()

def plot_behavioral_summary(metrics: Dict[str, float], title: str = "Behavioral metrics", save_path: Optional[str] = None):
    """metrics = {'accuracy':0.8, 'norm_edit':0.2, 'reward_diff':-5.0}"""
    keys = list(metrics.keys())
    vals = [metrics[k] for k in keys]
    plt.figure(figsize=(6,3))
    plt.bar(keys, vals)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved behavioral summary to {save_path}")
    plt.close()

# -----------------------------
# Helper: map token-based hidden states into room embeddings
# -----------------------------
def pool_positions(hidden_states: np.ndarray, positions: List[int], method: str = "mean"):
    """
    hidden_states: (seq_len, dim) or (dim, ) convert to (len(positions), dim) pooled vector
    """
    if len(positions) == 0:
        return None
    toks = hidden_states[positions, :]
    if method == "mean":
        return toks.mean(axis=0)
    else:
        return toks.sum(axis=0)

def build_room_embeddings_from_hidden_states(hidden_states: np.ndarray, positions_map: Dict[str, List[int]], method: str = "mean"):
    """
    hidden_states: (seq_len, dim) - for a given generation step
    positions_map: room -> token positions in that hidden_states indexing
    returns (n_rooms, dim)
    """
    rooms = list(positions_map.keys())
    embs = []
    for r in rooms:
        emb = pool_positions(hidden_states, positions_map.get(r, []), method=method)
        if emb is None:
            embs.append(np.zeros(hidden_states.shape[1], dtype=float))
        else:
            embs.append(emb)
    return np.stack(embs, axis=0), rooms

# -----------------------------
# End of metrics.py
# -----------------------------
