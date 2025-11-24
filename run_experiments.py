# run_experiments.py
"""
run_experiments.py

Runs a batch of trials using capture.save_trial.

By default runs:
- models: ['microsoft/phi-3-mini-4k-instruct', 'mistralai/Mistral-7B' (or google/gemma-2b if available)]
- graphs: line/tree/clustered
- methods: Scratchpad, Hybrid, DynamicRSA, Attention
- seeds: 1..N
- start_nodes: all nodes in graph (or a random subset if you set max_start_nodes)

Usage:
    python run_experiments.py --models microsoft/phi-3-mini-4k-instruct --graphs n7line,n7tree --methods Scratchpad --seeds 1
"""

import argparse
import time
from pathlib import Path
import random

from models_transformers import TransformersLLM
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from capture import save_trial

GRAPH_FACTORY = {
    "n7line": create_line_graph,
    "n7tree": create_tree_graph,
    "n15clustered": create_clustered_graph
}

METHODS = {
    "Scratchpad": {"save_activations": False, "max_new_tokens": 40},
    "Hybrid": {"save_activations": False, "max_new_tokens": 40},
    "DynamicRSA": {"save_activations": True, "max_new_tokens": 40},
    "Attention": {"save_activations": True, "max_new_tokens": 40},
}


def build_trial_id(model_short, graph_name, method_name, start_node, seed):
    ts = int(time.time())
    safe_start = start_node.replace(" ", "_")
    return f"{model_short}_{graph_name}_{method_name}_{safe_start}_s{seed}_{ts}"


def parse_csv_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="microsoft/phi-3-mini-4k-instruct", help="comma-separated hf model ids")
    parser.add_argument("--graphs", default="n7line,n7tree,n15clustered", help="comma-separated graph keys")
    parser.add_argument("--methods", default="Scratchpad,Hybrid,DynamicRSA,Attention", help="comma-separated methods")
    parser.add_argument("--seeds", default="1", help="comma-separated seeds (e.g., 1,2,3)")
    parser.add_argument("--max-start-nodes", default=None, type=int, help="Limit number of start nodes to run per graph (optional)")
    parser.add_argument("--output-dir", default="data/raw", help="Where trial outputs go (captures invoked write here)")
    args = parser.parse_args()

    models = parse_csv_list(args.models)
    graph_keys = parse_csv_list(args.graphs)
    methods = parse_csv_list(args.methods)
    seeds = [int(s) for s in parse_csv_list(args.seeds)]

    # ensure output dir exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # iterate
    for model_id in models:
        print(f"\n=== Loading model: {model_id} ===")
        llm = TransformersLLM(model_id=model_id)
        model_short = model_id.split("/")[-1]

        for graph_key in graph_keys:
            if graph_key not in GRAPH_FACTORY:
                print(f"Unknown graph key: {graph_key}, skipping.")
                continue
            G = GRAPH_FACTORY[graph_key]()
            nodes = list(G.nodes())

            # choose start nodes: either all or a random subset
            if args.max_start_nodes is not None and args.max_start_nodes < len(nodes):
                start_nodes = random.sample(nodes, args.max_start_nodes)
            else:
                start_nodes = nodes

            for start_node in start_nodes:
                for method_name in methods:
                    if method_name not in METHODS:
                        print(f"Unknown method: {method_name}, skipping.")
                        continue
                    cfg = METHODS[method_name]
                    for seed in seeds:
                        # set random seeds for reproducibility as best as possible
                        random.seed(seed)
                        try:
                            import numpy as np
                            import torch
                            np.random.seed(seed)
                            torch.manual_seed(seed)
                        except Exception:
                            pass

                        trial_id = build_trial_id(model_short, graph_key, method_name, start_node, seed)
                        print(f"\nStarting trial {trial_id}")
                        # call capture
                        try:
                            result = save_trial(
                                llm=llm,
                                G=G,
                                model_id=model_id,
                                graph_name=graph_key,
                                method_name=method_name,
                                start_node=start_node,
                                trial_id=trial_id,
                                max_new_tokens=cfg["max_new_tokens"],
                                save_attn_and_hidden=cfg["save_activations"]
                            )
                            print(f"Saved trial: {result}")
                        except Exception as e:
                            print(f"ERROR running trial {trial_id}: {e}")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
