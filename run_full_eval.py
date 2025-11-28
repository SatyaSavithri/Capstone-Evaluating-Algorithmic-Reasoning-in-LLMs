# run_full_eval.py
from evaluation_pipeline import evaluate_dataset, summarize_results
from my_model_wrapper import MyLLM
from tasks_loader import load_tasks


model = MyLLM()

# Load all graph tasks
tasks = load_tasks()

# 1. Standard
evaluate_dataset(model, tasks, mode="standard", save_path="standard.jsonl")

# 2. Scratchpad / CoT
evaluate_dataset(model, tasks, mode="scratchpad", save_path="scratchpad.jsonl")

# 3. Hybrid
evaluate_dataset(model, tasks, mode="hybrid", save_path="hybrid.jsonl")

# Summaries
print("Standard:", summarize_results("standard.jsonl"))
print("Scratchpad:", summarize_results("scratchpad.jsonl"))
print("Hybrid:", summarize_results("hybrid.jsonl"))
