# hybrid_runner_eval.py
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformersLLM:
    """A simple wrapper around a HuggingFace causal LM for generating tokens
    and capturing hidden states (activations) and attentions.
    """

    def __init__(self, model_name: str = "microsoft/phi-3-mini-4k-instruct", device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        self.model.to(device)
        self.model.eval()
        logger.info(f"Model {model_name} loaded.")

    def generate(self, prompt: str, max_new_tokens: int = 20):
        """Generates output tokens while capturing hidden states and attentions."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # Collect hidden states and attentions
        hidden_states = output.hidden_states if hasattr(output, "hidden_states") else None
        attentions = output.attentions if hasattr(output, "attentions") else None

        return self.tokenizer.decode(output.sequences[0], skip_special_tokens=True), hidden_states, attentions


def run_hybrid(model_wrapper: TransformersLLM, G, max_new_tokens: int = 20):
    """
    Simulates a hybrid experiment: uses the model to 'walk' the graph and
    collects activations & attentions for each node.
    """
    from graphs import NODE_PREFIX
    import torch

    # Get node names
    nodes = list(G.nodes())
    activations = []
    attentions = []

    # Simple simulation: feed each node name as a prompt to the model
    for node in nodes:
        prompt = f"Navigate to {node}:"
        try:
            _, hidden_states, attn = model_wrapper.generate(prompt, max_new_tokens=max_new_tokens)
        except Exception as e:
            logger.error(f"Failed generating for node {node}: {e}")
            hidden_states, attn = None, None

        # Convert hidden states to a tensor stack if possible
        if hidden_states:
            try:
                node_emb = torch.stack([h[:, 0, :] for h in hidden_states], dim=0)  # layer x batch x hidden
                activations.append(node_emb.cpu())
            except Exception as e:
                logger.warning(f"Failed stacking hidden states for {node}: {e}")
                activations.append(torch.zeros(1, 1, model_wrapper.model.config.hidden_size))
        else:
            activations.append(torch.zeros(1, 1, model_wrapper.model.config.hidden_size))

        if attn:
            attentions.append(attn)
        else:
            attentions.append(None)

    return {"activations": activations, "attentions": attentions}
