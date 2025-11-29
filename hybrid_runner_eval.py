# hybrid_runner_eval.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import networkx as nx
from collections import deque
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Transformers LLM wrapper ---
class TransformersLLM:
    def __init__(self, model_id: str = "microsoft/phi-3-mini-4k-instruct", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Note: Do not call .to(device) if device_map="auto" is used to avoid accelerate errors
        self.model.eval()
        logger.info(f"Model {model_id} loaded.")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded

    @torch.no_grad()
    def get_activations(self, prompt: str, max_new_tokens: int = 20):
        # returns last hidden state as embedding
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        output = self.model(**inputs, output_hidden_states=True)
        # last hidden state for the generated tokens
        hidden_states = output.hidden_states[-1]  # [batch, seq_len, hidden_dim]
        return hidden_states.squeeze(0).cpu()  # remove batch dim

# --- BFS path to max reward ---
def bfs_optimal_path_to_max_reward(G: nx.Graph, start_node: str) -> List[str]:
    """Compute BFS path from start node to node with maximum reward."""
    visited = set()
    queue = deque([[start_node]])
    max_reward_node = max(G.nodes, key=lambda n: G.nodes[n].get("reward", 0))

    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == max_reward_node:
            return path
        if node not in visited:
            visited.add(node)
            for neighbor in G.neighbors(node):
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    return [start_node]  # fallback

# --- Pick k candidate nodes (example heuristic) ---
def pick_k_candidates(G: nx.Graph, k: int = 3) -> List[str]:
    """Return top-k nodes with highest reward."""
    sorted_nodes = sorted(G.nodes, key=lambda n: G.nodes[n].get("reward", 0), reverse=True)
    return sorted_nodes[:k]

# --- Run hybrid experiment ---
def run_hybrid(model_wrapper: TransformersLLM, G: nx.Graph, max_new_tokens: int = 20):
    """
    Run hybrid experiment:
    - Compute BFS path
    - Generate LLM activations along the path
    - Return embeddings for RSM analysis
    """
    start_node = list(G.nodes)[0]  # pick first node as start
    gt_path = bfs_optimal_path_to_max_reward(G, start_node)
    logger.info(f"[INFO] Ground-truth BFS path: {gt_path}")

    llm_embs = []
    for node in gt_path:
        prompt = f"You are at {node}. What is the next room to move to?"
        emb = model_wrapper.get_activations(prompt, max_new_tokens=max_new_tokens)
        if emb is not None:
            llm_embs.append(emb)
        else:
            logger.warning(f"Embedding for {node} is None, skipping...")

    if not llm_embs:
        raise RuntimeError("No embeddings collected. Check model or prompts.")

    # Stack embeddings into tensor for RSM analysis
    embeddings_tensor = torch.stack(llm_embs)
    return embeddings_tensor, gt_path
