# rsa_analysis.py
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import networkx as nx

def build_theoretical_rsm(G, rooms):
    n = len(rooms)
    mat = np.zeros((n,n), dtype=float)
    for i,a in enumerate(rooms):
        for j,b in enumerate(rooms):
            if a == b:
                mat[i,j] = 1.0
            else:
                try:
                    d = nx.shortest_path_length(G, source=a, target=b)
                    mat[i,j] = 1.0/(1.0 + d)
                except Exception:
                    mat[i,j] = 0.0
    return mat

def compute_room_embeddings_from_hidden_states(hidden_states, positions_map, method="mean"):
    rooms = list(positions_map.keys())
    embs = []
    for r in rooms:
        pos = positions_map.get(r, [])
        if not pos:
            embs.append(np.zeros(hidden_states.shape[1], dtype=float))
            continue
        toks = hidden_states[pos, :]
        if method == "mean":
            embs.append(toks.mean(axis=0))
        else:
            embs.append(toks.sum(axis=0))
    return np.stack(embs, axis=0)

def rsm_from_embeddings(embs):
    d = pdist(embs, metric='cosine')
    sim = 1 - squareform(d)
    sim = np.nan_to_num(sim)
    return sim

def rsa_correlation(empirical_rsm, theoretical_rsm):
    iu = np.triu_indices_from(empirical_rsm, k=1)
    e = empirical_rsm[iu]
    t = theoretical_rsm[iu]
    r, p = spearmanr(e, t)
    return float(r), float(p)
