# ============================================================
# models_transformers.py
# Professional LLM Wrapper with Activation Capture (RSA + Attn)
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class TransformersLLM:
    """
    Wrapper around HuggingFace causal LMs supporting:
      • Standard generation
      • Hidden-state capture (Dynamic RSA)
      • Attention-weight capture (requires eager attention)
      • Loading via HF hub or preloaded instances
    """

    def __init__(self, model_id=None, model=None, tokenizer=None, device=None):
        """
        Initialize the LLM wrapper.

        Args:
            model_id: HuggingFace model ID (optional if model + tokenizer provided)
            model: Preloaded model instance
            tokenizer: Preloaded tokenizer instance
            device: "cuda", "cpu", or None → auto-detect
        """

        # -------------------------------
        # Device selection
        # -------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # -------------------------------
        # Load provided model/tokenizer
        # -------------------------------
        if model is not None and tokenizer is not None:
            self.model = model.to(self.device)
            self.tokenizer = tokenizer

        # -------------------------------
        # Load from HuggingFace Hub
        # -------------------------------
        elif model_id is not None:
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

            # Load model using GPU if possible
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
            )

            # --------------------------------------------------------
            # IMPORTANT:
            # SDPA attention backend CANNOT return attention weights.
            # To use attention analysis, switch to eager attention.
            # --------------------------------------------------------
            if hasattr(self.model.config, "attn_implementation"):
                try:
                    self.model.config.attn_implementation = "eager"
                except Exception:
                    pass

        else:
            raise ValueError("Either provide model_id OR both model and tokenizer.")

        # Disable hidden states + attentions by default
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

    # ============================================================
    # Standard Generation
    # ============================================================
    def generate(self, prompt: str, max_new_tokens: int = 50):
        """
        Wrapper around model.generate() for normal LLM output.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ============================================================
    # Dynamic RSA + Attention Analysis
    # ============================================================
    def generate_with_activations(self, prompt: str, max_new_tokens: int = 30):
        """
        Token-by-token generation that returns:
            • text output
            • generated token IDs
            • hidden states for every step & layer
            • attention matrices for every step & layer

        Returns a dictionary:
            {
                "text": ...,
                "tokens": ...,
                "hidden_states": [...],
                "attentions": [...]
            }
        """

        # Enable activation capture
        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        # Tokenize input prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated_ids = inputs["input_ids"]

        hidden_states_over_time = []
        attentions_over_time = []

        # ------------------------------------------
        # Manual autoregressive generation loop
        # ------------------------------------------
        for _ in range(max_new_tokens):

            outputs = self.model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )

            # Save activations
            hidden_states_over_time.append(outputs.hidden_states)
            attentions_over_time.append(outputs.attentions)

            # Greedy decoding (deterministic)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # Stop if EOS emitted
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        # Decode final text
        decoded_text = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )

        return {
            "text": decoded_text,
            "tokens": generated_ids,
            "hidden_states": hidden_states_over_time,
            "attentions": attentions_over_time,
        }
