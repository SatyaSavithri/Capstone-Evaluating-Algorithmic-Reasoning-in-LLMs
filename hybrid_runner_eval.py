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

    def generate(self, prompt, max_new_tokens=20):
        """Generates text with model and returns hidden states and attentions."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )
        # Extract hidden states and attentions from the model output
        hidden_states = output.decoder_hidden_states if hasattr(output, "decoder_hidden_states") else output.hidden_states
        attentions = output.decoder_attentions if hasattr(output, "decoder_attentions") else output.attentions
        # Return a simple namespace-like object
        class Output:
            pass
        out = Output()
        out.hidden_states = hidden_states
        out.attentions = attentions
        return out


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
            # Mean-pool over sequence length to get fixed-size representation
            last_hidden = output.hidden_states[-1]  # (batch, seq_len, hidden)
            if last_hidden.dim() == 3:
                pooled = last_hidden.mean(dim=1).squeeze(0)  # (hidden,)
            else:
                pooled = last_hidden.squeeze(0)
            activations[node] = pooled
            attentions[node] = output.attentions
        except Exception as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations[node] = torch.zeros(model_wrapper.model.config.hidden_size)
            attentions[node] = []

    return activations, attentions
