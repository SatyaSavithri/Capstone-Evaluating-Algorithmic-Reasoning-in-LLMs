# ================================================================
# run_capstone_transformers.py  (Final Updated Version)
# ================================================================

import matplotlib
matplotlib.use("Agg")   # REQUIRED for CyVerse / Jupyter non-GUI environment

import matplotlib.pyplot as plt
import networkx as nx

from models_transformers import TransformersLLM
from graphs import (
    create_line_graph,
    create_tree_graph,
    create_clustered_graph
)
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt


# ================================================================
#               USER INPUT SELECTION MENUS
# ================================================================

def choose_model():
    print("\nSelect model to load:")
    print("1) microsoft/phi-3-mini-4k-instruct (Phi-3 Mini 3.8B)")
    print("2) google/gemma-2b (Gemma 2B)")
    print("3) Custom HF Model ID")

    c = input("Enter choice: ").strip()

    if c == "1":
        return "microsoft/phi-3-mini-4k-instruct"
    elif c == "2":
        return "google/gemma-2b"
    else:
        return input("Enter full HuggingFace model ID: ")


def choose_graph():
    print("\nSelect graph:")
    print("1) Line graph")
    print("2) Tree graph")
    print("3) Clustered graph")

    c = input("Enter choice: ").strip()

    if c == "1":
        return "n7line", create_line_graph()
    elif c == "2":
        return "n7tree", create_tree_graph()
    else:
        return "n15clustered", create_clustered_graph()


def choose_method():
    print("\nSelect method:")
    print("1) Scratchpad Reasoning")
    print("2) Hybrid (Symbolic + LLM)")
    print("3) Dynamic RSA (hidden states)")
    print("4) Attention Analysis")
    return input("Enter choice: ").strip()


def choose_start_node(G):
    print("\nAvailable rooms:")
    for n in G.nodes():
        print(" -", n)

    while True:
        start = input("\nEnter START room exactly as listed: ").strip()
        if start in G.nodes():
            return start
        else:
            print("Invalid room. Try again.")


# ================================================================
#                 GRAPH VISUALIZATION (WORKS ON CYVERSE)
# ================================================================

def draw_graph(G, symbolic_path):
    """Draw graph, highlight path, save PNG, show inline in Jupyter."""
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))

    # draw all nodes
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=900,
        node_color="lightblue",
        font_size=10
    )

    # highlight edges of symbolic path
    if symbolic_path and len(symbolic_path) > 1:
        edges = list(zip(symbolic_path, symbolic_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               width=3, edge_color='red')

    plt.title("Graph with Symbolic Optimal Path")

    # save output
    save_path = "graph_output.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\nGraph saved to: {save_path}")

    # Show inline if inside Jupyter Lab (CyVerse)
    try:
        from IPython.display import Image, display
        display(Image(filename=save_path))
    except:
        pass    # no Jupyter available


# ================================================================
#                            MAIN
# ================================================================

def main():
    print("\n=== CAPSTONE: Phi-3 Mini & Gemma 2B Runner ===\n")

    # ------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------
    model_id = choose_model()
    llm = TransformersLLM(model_id=model_id)

    # ------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------
    graph_name, G = choose_graph()

    # ------------------------------------------------------------
    # Interactive start node
    # ------------------------------------------------------------
    start_node = choose_start_node(G)

    # ------------------------------------------------------------
    # Choose method
    # ------------------------------------------------------------
    method = choose_method()

    # ------------------------------------------------------------
    # Compute symbolic path (from chosen START)
    # ------------------------------------------------------------
    symbolic_path = bfs_optimal_path_to_max_reward(G, start_node)
    print("\nSymbolic optimal path:")
    print(" → ".join(symbolic_path))

    # ------------------------------------------------------------
    # Display graph
    # ------------------------------------------------------------
    draw_graph(G, symbolic_path)

    # ------------------------------------------------------------
    # Build LLM prompt
    # ------------------------------------------------------------
    description = base_description_text(G)
    prompt = scratchpad_prompt(description, "valuePath")

    print("\n=== Running Method ===\n")

    # ------------------------------------------------------------
    # Scratchpad
    # ------------------------------------------------------------
    if method == "1":
        print("Method: Scratchpad Reasoning\n")
        out = llm.generate(prompt, max_new_tokens=50)
        print("\nLLM Output:\n", out)

    # ------------------------------------------------------------
    # Hybrid
    # ------------------------------------------------------------
    elif method == "2":
        print("Method: Hybrid Reasoning\n")
        out = llm.generate(prompt, max_new_tokens=50)
        print("\nLLM Output:\n", out)

    # ------------------------------------------------------------
    # RSA
    # ------------------------------------------------------------
    elif method == "3":
        print("Method: Dynamic RSA\n")
        data = llm.generate_with_activations(prompt, max_new_tokens=20)
        print("\nHidden state layers:", len(data["hidden_states"]))
        print("Attention layers:", len(data["attentions"]))

    # ------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------
    elif method == "4":
        print("Method: Attention Analysis\n")
        data = llm.generate_with_activations(prompt, max_new_tokens=20)
        print("Attention layers:", len(data["attentions"]))
        for i, att in enumerate(data["attentions"]):
            print(f"Layer {i}: {att.shape}")

    print("\n=== Completed Successfully ===\n")


if __name__ == "__main__":
    main()
