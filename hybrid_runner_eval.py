import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TransformersLLM:
    """Wrapper for HuggingFace causal LM to extract hidden states and attentions."""

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

    def get_activations(self, prompt: str):
        """
        Forward pass through the model to get pooled activations and attentions.
        Ensures always returns a 1D tensor for activations.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        # output.hidden_states: tuple(layer_count+1) of [batch, seq_len, hidden]
        if hasattr(output, "hidden_states") and output.hidden_states is not None:
            # Take last layer hidden states
            last_hidden = output.hidden_states[-1]  # [batch, seq_len, hidden]
            # Pool over sequence
            pooled = last_hidden.mean(dim=1).squeeze(0)  # [hidden]
            if pooled.dim() == 0:
                # fallback to zeros
                pooled = torch.zeros(self.model.config.hidden_size, device=self.device)
        else:
            pooled = torch.zeros(self.model.config.hidden_size, device=self.device)

        # output.attentions: tuple(layer_count) of [batch, head, seq_len, seq_len]
        attentions = output.attentions if hasattr(output, "attentions") else []

        return pooled, attentions


def run_hybrid(model_wrapper, G):
    """
    Run hybrid experiment on the graph G.
    Returns:
        activations: dict[node_name] -> tensor (hidden_size)
        attentions: dict[node_name] -> list of tensors
    """
    activations = {}
    attentions = {}

    for node in G.nodes():
        prompt = f"Navigate to {node}."
        try:
            pooled, node_attentions = model_wrapper.get_activations(prompt)
            # Ensure tensor is always 1D
            if pooled is None or not torch.is_tensor(pooled):
                pooled = torch.zeros(model_wrapper.model.config.hidden_size, device=model_wrapper.device)
            activations[node] = pooled
            attentions[node] = node_attentions
        except Exception as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations[node] = torch.zeros(model_wrapper.model.config.hidden_size, device=model_wrapper.device)
            attentions[node] = []

    return activations, attentions
