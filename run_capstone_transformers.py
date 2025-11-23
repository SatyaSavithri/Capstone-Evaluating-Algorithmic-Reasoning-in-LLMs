# run_capstone_transformers.py

import matplotlib.pyplot as plt
import networkx as nx

from models_transformers import TransformersLLM
from graphs import create_line_graph, create_tree_graph, create_clustered_graph
from planner import bfs_optimal_path_to_max_reward
from prompts import base_description_text, scratchpad_prompt


# ============================================================
#                    USER SELECTION MENUS
# ============================================================

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
        return input("Enter the full Hugging Face model ID: ")


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
    print("1) Scratchpad")
    print("2) Hybrid")
    print("3) Dynamic RSA (hidden states)")
    print("4) Attention Analysis")
    return input("Enter choice: ").strip()


def choose_start_node(G):
    print("\nAvailable rooms:")
    for node in sorted(G.nodes()):
        print(" -", node)

    while True:
        start = input("\nEnter START room exactly as shown above: ").strip()
        if start in G.nodes():
            return start
        print("Invalid room. Try again.")


# ============================================================
#                 GRAPH VISUALIZATION FUNCTION
# ============================================================

def draw_graph(G, symbolic_path):
    """Draw the graph and highlight the chosen BFS path."""
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))

    nx.draw(G, pos, with_labels=True,
            node_color="lightblue",
            node_size=900,
            font_size=10)

    if symbolic_path and len(symbolic_path) > 1:
        edges = list(zip(symbolic_path, symbolic_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges,
                               width=3, edge_color="red")

    plt.title("Graph with Symbolic Optimal Path")
    plt.tight_layout()
    plt.show()


# ============================================================
#                           MAIN
# ============================================================

def main():
    print("\n=== CAPSTONE: Phi-3 Mini & Gemma 2B Runner ===\n")

    # -------------------- Load model --------------------
    model_id = choose_model()
    llm = TransformersLLM(model_id=model_id)

    # -------------------- Load graph --------------------
    graph_name, G = choose_graph()

    # -------------------- USER chooses start node --------------------
    start_node = choose_start_node(G)

    # -------------------- Choose method --------------------
    method = choose_method()

    # -------------------- Compute symbolic path --------------------
    symbolic_path = bfs_optimal_path_to_max_reward(G, start_node)
    print("\nSymbolic optimal path:")
    print(" → ".join(symbolic_path))

    # -------------------- Draw graph --------------------
    draw_graph(G, symbolic_path)

    # -------------------- Build prompt --------------------
    description = base_description_text(G)
    prompt = scratchpad_prompt(description, "valuePath")

    print("\n=== Running Method ===\n")

    # -------------------- METHOD EXECUTION --------------------
    if method == "1":
        print("Method: Scratchpad reasoning\n")
        answer = llm.generate(prompt, max_new_tokens=10)
        print("\nLLM Output:\n", answer)

    elif method == "2":
        print("Method: Hybrid reasoning\n")
        answer = llm.generate(prompt, max_new_tokens=10)
        print("\nLLM Output:\n", answer)

    elif method == "3":
        print("Method: Dynamic RSA (Hidden States)\n")
        data = llm.generate_with_activations(prompt, max_new_tokens=10)
        print("\nHidden state layers:", len(data["hidden_states"]))
        print("Attention layers:", len(data["attentions"]))

    elif method == "4":
        print("Method: Attention Analysis\n")
        data = llm.generate_with_activations(prompt, max_new_tokens=10)
        print("\nAttention layers:", len(data["attentions"]))

    print("\n=== Completed Successfully ===\n")


if __name__ == "__main__":
    main()
