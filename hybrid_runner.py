# hybrid_runner.py
from prompts import base_description_text, validator_prompt
from planner import bfs_optimal_path_to_max_reward
import networkx as nx

def pick_k_candidates(G, k=3):
    gt = bfs_optimal_path_to_max_reward(G)
    candidates = [gt]
    start = "Room 1"
    if start in G:
        for n in G.neighbors(start):
            if [start, n] not in candidates:
                candidates.append([start, n])
            if len(candidates) >= k:
                break
    rewards = {n: G.nodes[n].get("reward", 0) for n in G.nodes()}
    sorted_nodes = sorted(rewards.items(), key=lambda x: -x[1])
    for node, r in sorted_nodes:
        if len(candidates) >= k:
            break
        if node != gt[-1] and node != start:
            try:
                p = nx.shortest_path(G, source=start, target=node)
                if p not in candidates:
                    candidates.append(p)
            except Exception:
                continue
    return candidates[:k]

def run_hybrid(model_wrapper, G, task="valuePath", max_new_tokens=128):
    base = base_description_text(G)
    candidates = pick_k_candidates(G, k=3)
    validator = validator_prompt(base, candidates)
    validator_text = model_wrapper.text_generate(validator, max_new_tokens=max_new_tokens, temperature=0.0)
    best = None
    for line in validator_text.splitlines():
        if line.strip().upper().startswith("BEST:"):
            best = line.split(":",1)[1].strip()
            break
    return {"validator_text": validator_text, "candidates": candidates, "best": best}
