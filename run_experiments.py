# ============================================================
# run_experiments.py  (Interactive Batch Experiment Runner - Optimized)
# ============================================================

import time
import random
from pathlib import Path
import networkx as nx

from graphs import (
    create_line_graph,
    create_tree_graph,
    create_clustered_graph
)
from capture import save_trial
from capture_patch import load_model  # Optimized loader

# -------------------------------
# GRAPH FACTORY
# -------------------------------
GRAPH_FACTORY = {
    "1": ("n7line", create_line_graph),
    "2": ("n7tree", create_tree_graph),
    "3": ("n15clustered", create_clustered_graph),
}

METHOD_SELECTION = {
    "1": "Scratchpad",
    "2": "Hybrid",
    "3": "DynamicRSA",
    "4": "Attention",
}

# ============================================================
# INTERACTIVE MENUS
# ============================================================

def choose_model_interactive():
    print("\nSelect model:")
    print("1) microsoft/phi-3-mini-4k-instruct")
    print("2) google/gemma-2b")
    print("3) Custom HuggingFace model ID")

    while True:
        c = input("Choice: ").strip()
        if c == "1":
            return "microsoft/phi-3-mini-4k-instruct"
        elif c == "2":
            return "google/gemma-2b"
        elif c == "3":
            return input("Enter full HF model ID: ").strip()
        else:
            print("Invalid. Try again.")


def choose_graph_interactive():
    print("\nSelect graph:")
    print("1) Line Graph (n7line)")
    print("2) Tree Graph (n7tree)")
    print("3) Clustered Graph (n15clustered)")

    while True:
        c = input("Choice: ").strip()
        if c in GRAPH_FACTORY:
            return GRAPH_FACTORY[c]
        print("Invalid. Try again.")


def choose_methods_interactive():
    print("\nSelect methods (comma-separated):")
    print("1) Scratchpad")
    print("2) Hybrid")
    print("3) DynamicRSA")
    print("4) Attention")

    while True:
        raw = input("Enter method numbers (e.g., 1,3,4): ").strip()
        nums = [n.strip() for n in raw.split(",") if n.strip()]

        methods = []
        for n in nums:
            if n in METHOD_SELECTION:
                methods.append(METHOD_SELECTION[n])

        if methods:
            return methods
        print("Invalid. Try again.")


def choose_seeds_interactive():
    while True:
        raw = input("\nEnter number of seeds (e.g., 3): ").strip()
        if raw.isdigit():
            n = int(raw)
            if n >= 1:
                return list(range(1, n + 1))
        print("Invalid. Try again.")


def confirm_start():
    c = input("\nStart running experiments? (y/n): ").lower().strip()
    return c == "y"

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=========================================")
    print("        INTERACTIVE EXPERIMENT RUNNER")
    print("=========================================\n")

    model_id = choose_model_interactive()
    graph_name, graph_fn = choose_graph_interactive()
    methods = choose_methods_interactive()
    seeds = choose_seeds_interactive()

    # ----------------- Optimized model loading -----------------
    print("\nLoading model efficiently...\n")
    model, tokenizer = load_model(model_id)  # Use float16 & GPU if available

    # Wrap in your TransformersLLM wrapper if needed
    from capture_patch import load_model
    from models_transformers import TransformersLLM

    model, tokenizer = load_model(model_id)
    llm = TransformersLLM(model=model, tokenizer=tokenizer)


    model_short = model_id.split("/")[-1]

    # create graph
    G = graph_fn()
    start_nodes = list(G.nodes())

    print("\nThe following start nodes will be used:")
    for s in start_nodes:
        print(" -", s)

    if not confirm_start():
        print("\nExperiment cancelled.\n")
        return

    print("\n=========================================")
    print("          RUNNING EXPERIMENTS")
    print("=========================================\n")

    for start_node in start_nodes:
        for method_name in methods:
            for seed in seeds:

                random.seed(seed)
                trial_id = f"{model_short}_{graph_name}_{method_name}_{start_node.replace(' ','_')}_s{seed}"

                print(f"\n[Running Trial] {trial_id}")

                save_trial(
                    llm=llm,
                    G=G,
                    model_id=model_id,
                    graph_name=graph_name,
                    method_name=method_name,
                    start_node=start_node,
                    trial_id=trial_id,
                    max_new_tokens=40,
                    save_attn_and_hidden=(method_name in ("DynamicRSA", "Attention"))
                )

    print("\n=========================================")
    print("     ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=========================================\n")


if __name__ == "__main__":
    main()
