# ============================================================
# models_transformers.py
# Professional LLM Wrapper with Activation Capture
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersLLM:
    """
    Wrapper around a HuggingFace casual LM that supports:
      - Standard text generation
      - Activation capture (hidden states + attention)
      - Loading from HF hub OR using externally passed model/tokenizer
    """

    def __init__(self, model_id=None, model=None, tokenizer=None, device=None):
        """
        Initialize the LLM wrapper.

        Parameters:
        - model_id: HuggingFace model ID (string)
        - model: Preloaded model instance (optional)
        - tokenizer: Preloaded tokenizer instance (optional)
        - device: "cuda", "cpu", or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load externally supplied model + tokenizer
        if model is not None and tokenizer is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer

        # Load from HuggingFace Hub
        elif model_id is not None:
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )

        else:
            raise ValueError("Either provide model_id OR both model and tokenizer.")

        # Ensure the model returns hidden states & attention when requested
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

    # ------------------------------------------------------------
    # Standard Generation
    # ------------------------------------------------------------
    def generate(self, prompt: str, max_new_tokens: int = 50):
        """
        Simple wrapper for model.generate()
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------
    # Dynamic RSA + Attention Analysis
    # ------------------------------------------------------------
    def generate_with_activations(self, prompt: str, max_new_tokens: int = 30):
        """
        Manual token-by-token generation that records:
            - hidden_states for each layer at each step
            - attention matrices at each step
            - generated token IDs

        Returns:
            {
                "text": decoded_output,
                "tokens": tensor(shape=[1, seq_len]),
                "hidden_states": list[step][layers][batch, seq, dim],
                "attentions": list[step][layers][batch, heads, seq, seq]
            }
        """

        # Enable hidden states + attention
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs["input_ids"]

        hidden_states_over_time = []
        attentions_over_time = []

        for _ in range(max_new_tokens):

            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )

            # Store activations
            hidden_states_over_time.append(outputs.hidden_states)
            attentions_over_time.append(outputs.attentions)

            # Greedy decode
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # End if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode result
        decoded_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        return {
            "text": decoded_text,
            "tokens": generated_ids,
            "hidden_states": hidden_states_over_time,
            "attentions": attentions_over_time,
        }
