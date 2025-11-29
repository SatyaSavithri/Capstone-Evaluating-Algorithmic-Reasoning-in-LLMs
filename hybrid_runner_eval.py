# hybrid_runner_eval.py
import torch
from torch import nn

# --- Example LLM wrapper ---
class SimpleLLMWrapper(nn.Module):
    """
    Minimal LLM-like wrapper for generating embeddings/activations
    compatible with evaluation_runner.py
    """
    def __init__(self, embed_dim=16):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple linear layer to simulate embedding generation
        self.embed_layer = nn.Linear(1, embed_dim)

    def forward(self, node_index_tensor):
        """
        node_index_tensor: tensor of shape [num_nodes, 1]
        returns: embeddings tensor of shape [num_nodes, embed_dim]
        """
        return self.embed_layer(node_index_tensor.float())

def run_hybrid_eval(G, start_node="Room 1", device="cuda"):
    """
    Run hybrid evaluation for a given graph G starting from start_node.
    Returns activations (tensor) and success flag (bool)
    """
    try:
        num_nodes = len(G.nodes)
        # Assign integer indices to nodes for embeddings
        node_to_idx = {node: i for i, node in enumerate(G.nodes)}
        idx_tensor = torch.arange(num_nodes).unsqueeze(1).to(device)  # shape [num_nodes, 1]

        # Create the simple LLM wrapper
        llm = SimpleLLMWrapper(embed_dim=16).to(device)

        # Forward pass to generate activations
        with torch.no_grad():
            activations = llm(idx_tensor)  # shape [num_nodes, embed_dim]

        success = True
        return activations, success

    except Exception as e:
        print(f"[ERROR] run_hybrid_eval failed: {e}")
        # Return dummy activations to avoid breaking evaluation_runner
        dummy_activations = torch.zeros(len(G.nodes), 16)
        return dummy_activations, False
