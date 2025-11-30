# hybrid_runner.py
"""
Patched hybrid runner: returns dict with keys 'candidates', 'best', 'validator_text'.
Does not expect hidden_states from llm.generate().
"""

import networkx as nx
from prompts import validator_prompt
from typing import List, Dict, Any
import logging

logger = logging.getLogger("hybrid_runner")


def pick_k_candidates(G: nx.Graph, start="Room 1", k=3) -> List[List[str]]:
    candidates = []
    rewards = nx.get_node_attributes(G, "reward")
    if not rewards:
        return candidates

    # optimal by shortest path to max reward
    max_node = max(rewards, key=lambda n: rewards[n])
    try:
        gt = nx.shortest_path(G, source=start, target=max_node)
        candidates.append(gt)
    except Exception:
        pass

    # add neighbor short paths
    if start in G:
        for n in G.neighbors(start):
            p = [start, n]
            if p not in candidates:
                candidates.append(p)
            if len(candidates) >= k:
                return candidates[:k]

    # add more candidate paths by reward descending
    sorted_nodes = sorted(rewards.items(), key=lambda x: -x[1])
    for node, _ in sorted_nodes:
        if node == max_node or node == start:
            continue
        try:
            p = nx.shortest_path(G, source=start, target=node)
            if p not in candidates:
                candidates.append(p)
        except Exception:
            continue
        if len(candidates) >= k:
            break

    # pad if needed (loop or trivial)
    if len(candidates) < k:
        candidates.append([start, start])

    return candidates[:k]


def run_hybrid(llm, G: nx.Graph, start: str = "Room 1", task: str = "valuePath",
               k: int = 3, max_new_tokens: int = 128) -> Dict[str, Any]:
    """
    llm: instance of TransformersLLM (must implement .generate(prompt, ...))
    G: networkx graph
    Returns: dict {"candidates": [...], "best": "P1", "validator_text": "..."}
    """
    candidates = pick_k_candidates(G, start=start, k=k)
    base_text = f"Facility with {G.number_of_nodes()} rooms starting at {start}.\n"
    validator = validator_prompt(base_text, candidates)
    try:
        validator_text = llm.generate(validator, max_new_tokens=max_new_tokens)
    except Exception as e:
        logger.warning(f"LLM generate failed in hybrid validator: {e}")
        validator_text = ""

    # parse BEST line
    best = None
    for line in validator_text.splitlines():
        line = line.strip()
        if line.upper().startswith("BEST:"):
            best = line.split(":", 1)[1].strip()
            break

    return {"candidates": candidates, "best": best, "validator_text": validator_text}
