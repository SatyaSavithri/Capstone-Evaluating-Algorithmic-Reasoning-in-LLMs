# hybrid_runner_eval.py
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TransformersLLM:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",           # safe GPU/CPU offload
            torch_dtype=torch.float16,   # reduce memory
        )
        self.model.eval()
        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")  # enable attentions if supported
        logger.info(f"Model {model_name} loaded.")

    @torch.no_grad()
    def generate_activations(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        # Use last hidden layer
        last_hidden_state = outputs.hidden_states[-1].detach().cpu()
        attentions = None
        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            attentions = [a.detach().cpu() for a in outputs.attentions]
        return last_hidden_state, attentions

def run_hybrid(llm, path_nodes):
    """
    Generate activations for a sequence of nodes (path).
    llm: TransformersLLM instance
    path_nodes: list of node names
    Returns:
        activations_list: list of tensors [seq_len, hidden_dim]
        attentions_list: list of list of tensors or None
    """
    activations_list = []
    attentions_list = []

    for node in path_nodes:
        prompt = f"Navigate from {path_nodes[0]} to {node}"
        try:
            act, att = llm.generate_activations(prompt)
            activations_list.append(act)
            attentions_list.append(att)
        except RuntimeError as e:
            logger.warning(f"Failed to generate for node {node}: {e}")
            activations_list.append(torch.zeros(1, llm.model.config.hidden_size))
            attentions_list.append(None)

    return activations_list, attentions_list
