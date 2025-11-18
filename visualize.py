# visualize.py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def draw_graph(G, highlight_path=None, title=""):
    pos = nx.spring_layout(G, seed=42)
    rewards = nx.get_node_attributes(G, "reward")
    node_colors = [rewards.get(n, 0) for n in G.nodes()]
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, with_labels=True, node_size=900, node_color=node_colors, cmap=plt.cm.YlOrRd, font_weight='bold')
    if highlight_path and len(highlight_path) > 1:
        edges = list(zip(highlight_path, highlight_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=4.0, edge_color='blue')
    plt.title(title)
    plt.show()

def plot_time_series(values, title="", xlabel="Step", ylabel="Value"):
    x = list(range(len(values)))
    y = [v if v is not None else float("nan") for v in values]
    plt.figure(figsize=(8,4))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title); plt.grid(True)
    plt.show()

def plot_rsm(mat, title="RSM"):
    plt.figure(figsize=(6,5))
    plt.imshow(mat, cmap='viridis', interpolation='nearest')
    plt.colorbar(); plt.title(title); plt.show()
