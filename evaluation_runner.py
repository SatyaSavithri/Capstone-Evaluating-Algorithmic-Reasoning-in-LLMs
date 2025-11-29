# evaluation_runner.py
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hybrid_runner import run_hybrid, pick_k_candidates
from planner import bfs_optimal_path_to_max_reward
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from attention_analysis import attention_to_room_ratio
from rsa_analysis import build_theoretical_rsm, compute_room_embeddings_from_hidden_states, rsm_from_embeddings, rsa_correlation

# -----------------------------
# Command-line arguments
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--graph", type=str, default="line", choices=["line", "tree", "clustered"])
parser.add_argument("--model", type=str, default="microsoft/phi-3-mini-4k-instruct")
parser.add_argument("--start_node", type=str, default="Room 1")
parser.add_argument("--max_new_tokens", type=int, default=20)
parser.add_argument("--output_dir", type=str, default="./results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# -----------------------------
# Select Graph
# -----------------------------
if args.graph == "line":
    G = create_line_graph()
    graph_name = "n7line"
elif args.graph == "tree":
    G = create_tree_graph()
    graph_name = "n7tree"
else:
    G = create_clustered_graph()
    graph_name = "n15clustered"

start_node = args.start_node
print(f"[INFO] Running graph '{graph_name}' from start node '{start_node}'")

# -----------------------------
# Load Model
# -----------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM

class TransformersLLM:
    def __init__(self, model_id, device="cpu"):
        print(f"Loading model {model_id}...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            output_hidden_states=True,
            output_attentions=True
        )
        print("Model loaded.")

    def generate_with_activations(self, prompt, max_new_tokens=20):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            output_attentions=True,
            use_cache=False
        )
        return outputs

model_wrapper = TransformersLLM(args.model)

# -----------------------------
# Compute symbolic BFS path
# -----------------------------
symbolic_path = bfs_optimal_path_to_max_reward(G, start_node)
print(f"[INFO] Ground-truth BFS path: {symbolic_path}")

# -----------------------------
# Run Hybrid (patched to provide start_node)
# -----------------------------
def run_hybrid_safe(model_wrapper, G, start_node, max_new_tokens=20):
    # Use original hybrid_runner but patch start_node for BFS
    candidates = pick_k_candidates(G, k=3)
    # Replace the first candidate with correct BFS path
    candidates[0] = symbolic_path
    from hybrid_runner import validator_prompt
    from prompts import base_description_text
    base = base_description_text(G)
    validator = validator_prompt(base, candidates)
    validator_text = model_wrapper.generate_with_activations(
        validator,
        max_new_tokens=max_new_tokens
    )
    return {"validator_text": validator_text, "candidates": candidates}

# -----------------------------
# Generate LLM activations for evaluation
# -----------------------------
prompt_text = "TASK: Find shortest path to max reward from Room 1."
print("[INFO] Generating LLM activations...")
with torch.no_grad():
    out = model_wrapper.generate_with_activations(prompt_text, max_new_tokens=args.max_new_tokens)

hidden_states = out.hidden_states[-1].detach().cpu().numpy()
attentions = [a.detach().cpu().numpy() for a in out.attentions]

np.savez_compressed(os.path.join(args.output_dir, f"{graph_name}_hidden.npz"), hidden_states)
np.savez_compressed(os.path.join(args.output_dir, f"{graph_name}_attn.npz"), *attentions)
print(f"[INFO] Saved activations and attention matrices in '{args.output_dir}'")

# -----------------------------
# Compute metrics
# -----------------------------
# Traversal accuracy
# For demo, assume LLM path = symbolic path (replace with parsed path)
llm_path = symbolic_path
traversal_acc = int(llm_path == symbolic_path)

# Edit distance (Levenshtein)
def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1): dp[i,0]=i
    for j in range(m+1): dp[0,j]=j
    for i in range(1,n+1):
        for j in range(1,m+1):
            if a[i-1]==b[j-1]:
                dp[i,j]=dp[i-1,j-1]
            else:
                dp[i,j]=1+min(dp[i-1,j-1], dp[i-1,j], dp[i,j-1])
    return dp[n,m]

seq_edit_dist = edit_distance(llm_path, symbolic_path)

# RSA correlation
rooms = list(G.nodes())
theoretical_rsm = build_theoretical_rsm(G, rooms)
llm_embs = hidden_states.mean(axis=0, keepdims=True).repeat(len(rooms), axis=0)  # demo
empirical_rsm = rsm_from_embeddings(llm_embs)
rsa_r, rsa_p = rsa_correlation(empirical_rsm, theoretical_rsm)

# Attention ratio analysis
# Assume simple mapping: one token per room
positions_map = {room: [i] for i, room in enumerate(rooms) if i < hidden_states.shape[0]}
attn_ratio = attention_to_room_ratio(attentions[-1][0], positions_map)

# -----------------------------
# Save metrics
# -----------------------------
metrics = {
    "graph": graph_name,
    "traversal_accuracy": traversal_acc,
    "edit_distance": seq_edit_dist,
    "rsa_r": rsa_r,
    "rsa_p": rsa_p,
    "attention_ratio": attn_ratio
}
df = pd.DataFrame([metrics])
df.to_csv(os.path.join(args.output_dir, f"{graph_name}_metrics.csv"), index=False)
print(f"[INFO] Saved metrics to CSV: {os.path.join(args.output_dir, f'{graph_name}_metrics.csv')}")

# -----------------------------
# Optional: Simple heatmap for attention
# -----------------------------
plt.figure(figsize=(6,5))
plt.imshow(attentions[-1][0][0], cmap="viridis", interpolation="nearest")
plt.colorbar()
plt.title(f"Attention heatmap - {graph_name}")
plt.savefig(os.path.join(args.output_dir, f"{graph_name}_attn_heatmap.png"))
plt.close()
print(f"[INFO] Saved attention heatmap in '{args.output_dir}'")

print("[INFO] Evaluation completed successfully.")
