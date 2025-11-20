# prompts.py
import networkx as nx
from typing import List

def base_description_text(G: nx.Graph, start_node: str = "Room 1") -> str:
    n = G.number_of_nodes()
    edges = "; ".join([f"{u}-{v}" for u, v in G.edges()])
    rewards = nx.get_node_attributes(G, "reward")
    reward_text = ", ".join([f"{k}=${v}" for k, v in rewards.items() if v > 0])
    return f"Facility with {n} rooms starting at {start_node}.\nConnections: {edges}.\nRewards: {reward_text}."

def scratchpad_prompt(base_text: str, task: str) -> str:
    sp = base_text + "\n\n"
    if task == "valuePath":
        sp += "TASK: Find the shortest path from Room 1 to the room with the highest reward.\n"
    elif task == "rewardReval":
        sp += "TASK: Re-evaluate after a reward update.\n"
    sp += (
        "\n***SCRATCHPAD METHOD***\n"
        "For each step: Step i: Current room: [Room X]; Choices: [Room A, Room B]; Chosen: [Room Y]; Rationale: [brief].\n"
        "FINAL_JSON: {\"path\": [\"Room 1\", \"Room 2\", ...]}\n"
        "Important: final JSON line must be valid JSON with key 'path'."
    )
    return sp

def validator_prompt(base_text: str, candidates: List[List[str]]) -> str:
    cand_lines = "\n".join([f"P{i+1}: {' -> '.join(p)}" for i,p in enumerate(candidates)])
    return f"{base_text}\n\nProvided candidate paths:\n{cand_lines}\n\nTASK: Evaluate which candidate reaches the highest reward. Answer exactly 'BEST: P#' on a single line. If tie, list comma-separated (e.g., BEST: P1,P2)."
