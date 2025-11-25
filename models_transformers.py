import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersLLM:
    def __init__(self, model_id: str, device: str = "cpu"):
        print(f"Loading model {model_id} (this may take time)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Always use CPU-friendly settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            output_hidden_states=True,
            output_attentions=True
        )

        # IMPORTANT: Enable attention extraction
        try:
            self.model.set_attn_implementation("eager")
        except Exception:
            print("⚠ Attention implementation setting not supported; continuing.")

        self.device = device
        print(f"Model {model_id} loaded.\n")

    # -------------------------------------------------
    # Standard generation (Scratchpad / Hybrid)
    # -------------------------------------------------
    def generate(self, prompt: str, max_new_tokens=200, temperature=0.1):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    # -------------------------------------------------
    # Full forward pass with activations (RSA & Attention)
    # -------------------------------------------------
    def generate_with_activations(self, prompt: str, max_new_tokens=128):
        print("Generating with activations (may take time)...")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False
        )

        return {
            "prompt": prompt,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "model_output": outputs
        }
