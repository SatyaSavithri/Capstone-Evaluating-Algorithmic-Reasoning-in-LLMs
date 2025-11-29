# hybrid_runner_eval.py
import torch
import networkx as nx
from typing import Tuple, List

def run_hybrid(model_wrapper, G: nx.Graph, max_new_tokens: int = 20) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Run the model on the given graph G and return activations and attention matrices.

    Args:
        model_wrapper (dict): Dictionary containing 'model', 'tokenizer', 'device'
        G (nx.Graph): Graph to traverse
        max_new_tokens (int): Number of tokens to generate for each node

    Returns:
        activations (List[torch.Tensor]): List of hidden states for each node
        attention_matrices (List[torch.Tensor]): List of attention matrices for each node
    """
    model = model_wrapper["model"]
    tokenizer = model_wrapper["tokenizer"]
    device = model_wrapper["device"]

    activations = []
    attention_matrices = []

    # Example traversal: BFS from "Room 1"
    start_node = "Room 1"
    queue = [start_node]
    visited = set()

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        # Prepare input prompt (can be customized per node)
        prompt = f"Navigate to {node}."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Run model forward pass with attention outputs
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1][:, 0, :]  # CLS-like representation
            attentions = torch.stack(outputs.attentions)        # Stack all layers

        activations.append(hidden_states.squeeze(0).cpu())
        attention_matrices.append(attentions.cpu())

        # Add neighbors to queue for BFS
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)

        # Limit generation to max_new_tokens (optional)
        if len(activations) >= max_new_tokens:
            break

    return activations, attention_matrices
