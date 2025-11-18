# models_transformers.py
from dotenv import load_dotenv
load_dotenv()
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

def load_transformers_model(model_id: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {}
    # For CPU we use low_cpu_mem_usage to reduce peak memory
    if device == "cpu":
        kwargs = {"device_map": "cpu", "low_cpu_mem_usage": True}
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer

class TransformersModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # model device detection
        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = torch.device("cpu")

    def text_generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=(temperature>0.0), temperature=temperature)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text

    def generate_with_activations(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.0):
        """
        Token-by-token generation capturing:
         - step 0: full prompt last-layer hidden states (seq_len, hidden_dim)
         - subsequent steps: last token hidden vector and averaged attentions (seq_len x seq_len)
        Returns: {'text': str, 'activations': [ {step, hidden_states, attentions}, ... ], 'input_ids': list}
        """
        import torch
        tokenizer = self.tokenizer
        model = self.model
        device = self.device

        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True, use_cache=True)
        past = outputs.past_key_values
        hidden_states = outputs.hidden_states
        activations = []
        # prompt-level last layer hidden states
        prompt_last = hidden_states[-1][0].cpu().numpy() if hidden_states is not None and hidden_states[-1] is not None else None
        activations.append({"step": 0, "hidden_states": prompt_last, "attentions": None})

        generated_ids = []
        eos_id = tokenizer.eos_token_id

        for step in range(max_new_tokens):
            if step == 0:
                cur_input = input_ids[:, -1:].to(device)
            else:
                cur_input = torch.tensor([[generated_ids[-1]]], device=device)
            with torch.no_grad():
                out_step = model(input_ids=cur_input, past_key_values=past, use_cache=True, output_attentions=True, output_hidden_states=True)
            past = out_step.past_key_values
            logits = out_step.logits
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).cpu().numpy()[0])
            generated_ids.append(next_id)
            # last-token hidden state
            hs = out_step.hidden_states[-1][0, -1, :].cpu().numpy() if out_step.hidden_states is not None else None
            # attentions average across heads & layers if present
            attn_mean = None
            if out_step.attentions:
                try:
                    att_stack = [a[0].mean(axis=0).cpu().numpy() for a in out_step.attentions]
                    attn_mean = np.mean(np.stack(att_stack, axis=0), axis=0)
                except Exception:
                    attn_mean = None
            activations.append({"step": step+1, "hidden_states": hs, "attentions": attn_mean})
            if eos_id is not None and next_id == eos_id:
                break

        full_ids = input_ids[0].cpu().numpy().tolist() + generated_ids
        text = tokenizer.decode(full_ids, skip_special_tokens=True)
        return {"text": text, "activations": activations, "input_ids": full_ids}
