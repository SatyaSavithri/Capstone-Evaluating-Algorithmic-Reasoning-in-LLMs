# rsa_analysis.py
"""
RSA utilities (final).
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from pathlib import Path

def build_theoretical_rsm(G: nx.Graph, rooms: list):
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

def compute_room_embeddings_from_hidden_states(hidden_states: np.ndarray, positions_map: dict, method="mean"):
    """
    hidden_states: (seq, hidden_dim) numpy array
    positions_map: dict room->list[token_positions]
    returns array (n_rooms, hidden_dim)
    """
    rooms = list(positions_map.keys())
    embs = []
    for r in rooms:
        pos = positions_map.get(r, [])
        if not pos:
            embs.append(np.zeros(hidden_states.shape[1], dtype=float))
            continue
        toks = hidden_states[pos, :]
        if method == "mean":
            embs.append(np.mean(toks, axis=0))
        else:
            embs.append(np.sum(toks, axis=0))
    return np.stack(embs, axis=0)

def rsm_from_embeddings(embs: np.ndarray):
    d = pdist(embs, metric='cosine')
    sim = 1 - squareform(d)
    sim = np.nan_to_num(sim)
    return sim

def rsa_correlation(empirical_rsm: np.ndarray, theoretical_rsm: np.ndarray):
    iu = np.triu_indices_from(empirical_rsm, k=1)
    e = empirical_rsm[iu]
    t = theoretical_rsm[iu]
    # if constant arrays, spearmanr returns nan; handle gracefully
    try:
        r, p = spearmanr(e, t)
        if np.isnan(r):
            return None, None
        return float(r), float(p)
    except Exception:
        return None, None

def save_rsm_plot(rsm_mat, save_path, title="RSM"):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,5))
    plt.imshow(rsm_mat, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=200)
    plt.close()
    return str(save_path)
