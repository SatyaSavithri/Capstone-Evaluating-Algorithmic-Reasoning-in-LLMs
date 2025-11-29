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
        max_new_tokens (int): Max nodes to process (acts as cutoff)

    Returns:
        activations (List[torch.Tensor]): List of hidden states for each node
        attention_matrices (List[torch.Tensor]): List of attention matrices for each node
    """
    model = model_wrapper["model"]
    tokenizer = model_wrapper["tokenizer"]
    device = model_wrapper["device"]

    activations = []
    attention_matrices = []

    start_node = "Room 1"
    queue = [start_node]
    visited = set()

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        prompt = f"Navigate to {node}."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
            # Extract last hidden state (CLS token-like)
            if outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states[-1][:, 0, :]
                activations.append(hidden_states.squeeze(0).cpu())
            else:
                activations.append(torch.zeros(model.config.hidden_size))  # fallback

            # Extract attentions safely
            if outputs.attentions is not None:
                attention_matrices.append(torch.stack(outputs.attentions).cpu())
            else:
                # Append empty tensor if attentions are None
                attention_matrices.append(torch.zeros(1, 1, 1, 1))

        # BFS neighbors
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)

        if len(activations) >= max_new_tokens:
            break

    return activations, attention_matrices
