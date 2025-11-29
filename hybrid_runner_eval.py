# hybrid_runner_eval.py
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class TransformersLLM:
    """
    Wrapper around a HuggingFace transformers causal LM
    """
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if device=="cuda" else torch.float32
        )
        self.model.eval()
        logger.info(f"Model {model_name} loaded.")

    def generate_embedding(self, prompt: str):
        """
        Generate embeddings for a given prompt
        Returns: torch.Tensor of shape (seq_len, hidden_size)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        # Take last hidden state
        hidden_states = outputs.hidden_states[-1].squeeze(0)  # (seq_len, hidden_size)
        return hidden_states

def run_hybrid(model_wrapper: TransformersLLM, G, start_node):
    """
    Run LLM activations for each node in graph G
    Returns:
        activations: dict[node_name] = tensor
        attentions: dict[node_name] = tensor (dummy here, can extend later)
    """
    activations = {}
    attentions = {}

    for node in G.nodes:
        try:
            # Generate node prompt (example: node name)
            prompt = f"Generate features for {node}"
            emb = model_wrapper.generate_embedding(prompt)
            activations[node] = emb
            # Dummy attentions as zeros (since real attention handling may vary)
            attentions[node] = torch.zeros(emb.size(0), emb.size(1))
        except Exception as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations[node] = torch.zeros(1, model_wrapper.model.config.hidden_size)
            attentions[node] = torch.zeros(1, model_wrapper.model.config.hidden_size)

    return activations, attentions
