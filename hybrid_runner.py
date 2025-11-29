# hybrid_runner_eval.py
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

class TransformersLLM:
    """Wrapper for transformers LLM for generating outputs and extracting activations."""
    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device = device

        logger.info(f"Loading model {model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model {model_id} loaded.")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20):
        """Generate output and capture hidden states."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True, output_attentions=True)
        return outputs


def run_hybrid(model_wrapper, G, max_new_tokens=20):
    """
    Run a hybrid experiment on a graph with LLM.
    Returns activations and attentions dictionaries for all nodes.
    """
    activations = {}
    attentions = {}

    for node in G.nodes():
        prompt = f"Navigate to {node}."
        try:
            output = model_wrapper.generate(prompt, max_new_tokens=max_new_tokens)
            # Take last hidden state of last token
            hidden_state = output.hidden_states[-1][:, -1, :].detach().cpu()
            activations[node] = hidden_state.squeeze(0)  # shape: (hidden_dim,)
            attentions[node] = output.attentions  # list of attention matrices
        except Exception as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations[node] = torch.zeros(model_wrapper.model.config.hidden_size)
            attentions[node] = []

    return activations, attentions
