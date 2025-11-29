import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TransformersLLM:
    """Wrapper for HuggingFace causal LM with activations and attentions extraction."""

    def __init__(self, model_id, device="cuda"):
        self.device = device
        logger.info(f"Loading model {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            output_attentions=True,
            output_hidden_states=True,
        )
        self.model.eval()
        logger.info(f"Model {model_id} loaded.")

    def get_activations(self, prompt):
        """Forward pass through model to get hidden states and attentions."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)

        # Hidden states: tuple of (layer_count+1) tensors of shape [batch, seq_len, hidden]
        hidden_states = output.hidden_states
        # Take mean over sequence length to get fixed-size vector
        last_hidden = hidden_states[-1]  # (batch, seq_len, hidden)
        pooled = last_hidden.mean(dim=1).squeeze(0)  # (hidden,)
        attentions = output.attentions  # tuple of tensors
        return pooled, attentions


def run_hybrid(model_wrapper, G):
    """
    Run a hybrid experiment on a graph with LLM.
    Returns activations and attentions dictionaries for all nodes.
    """
    activations = {}
    attentions = {}

    for node in G.nodes():
        prompt = f"Navigate to {node}."
        try:
            pooled, node_attentions = model_wrapper.get_activations(prompt)
            activations[node] = pooled
            attentions[node] = node_attentions
        except Exception as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations[node] = torch.zeros(model_wrapper.model.config.hidden_size, device=model_wrapper.device)
            attentions[node] = []

    return activations, attentions
