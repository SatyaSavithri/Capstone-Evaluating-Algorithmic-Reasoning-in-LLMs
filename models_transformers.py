# ============================================================
# models_transformers.py  (Updated to accept external model & tokenizer)
# ============================================================

from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformersLLM:
    def __init__(self, model_id=None, model=None, tokenizer=None):
        """
        Wrapper for Transformers LLM.
        model_id: str, Hugging Face model ID (optional if model is provided)
        model: pre-loaded model instance (optional)
        tokenizer: pre-loaded tokenizer instance (optional)
        """
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        elif model_id is not None:
            self.model_id = model_id
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id)
        else:
            raise ValueError("Either provide model_id or both model and tokenizer")

    def generate(self, prompt, max_new_tokens=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
