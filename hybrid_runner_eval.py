# hybrid_runner_eval.py
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TransformersLLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        logger.info(f"Model {model_name} loaded.")

    def get_activations_and_attention(self, prompts: list):
        """
        Generate outputs and return activations and attention matrices
        """
        activations = []
        attentions = []

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            try:
                outputs = self.model(**inputs, output_attentions=True)
                # Take last hidden state as activations
                last_hidden_state = outputs.last_hidden_state.detach().cpu()
                attention = torch.stack([att.detach().cpu() for att in outputs.attentions])
                
                activations.append(last_hidden_state)
                attentions.append(attention)
            except Exception as e:
                logger.warning(f"Failed to generate for prompt '{prompt}': {e}")
        
        if not activations:
            return None, None

        # Stack activations and attentions
        try:
            activations = torch.stack(activations)
        except:
            # fallback for variable-length sequences
            activations = [act for act in activations]

        try:
            attentions = torch.stack(attentions)
        except:
            attentions = [att for att in attentions]

        return activations, attentions


def run_hybrid(graph, start_node, model_wrapper):
    """
    Generate activations and attentions for each node in BFS order.
    """
    nodes = list(graph.nodes)
    prompts = [f"Navigate from {start_node} to {node}" for node in nodes]
    activations, attentions = model_wrapper.get_activations_and_attention(prompts)
    return activations, attentions
