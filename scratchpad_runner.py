# scratchpad_runner.py
from prompts import base_description_text, scratchpad_prompt
from utils import parse_final_json_path

def run_scratchpad(model_wrapper, G, task="valuePath", max_new_tokens=128):
    base = base_description_text(G)
    prompt = scratchpad_prompt(base, task)
    llm_text = model_wrapper.text_generate(prompt, max_new_tokens=max_new_tokens, temperature=0.0)
    parsed_path = parse_final_json_path(llm_text)
    return {"prompt": prompt, "llm_text": llm_text, "parsed_path": parsed_path}
