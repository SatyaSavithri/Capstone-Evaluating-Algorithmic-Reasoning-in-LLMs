# graphs.py
import networkx as nx

NODE_PREFIX = "Room"


def create_line_graph():
    G = nx.path_graph(7)
    mapping = {i: f"{NODE_PREFIX} {i+1}" for i in range(7)}
    G = nx.relabel_nodes(G, mapping)

    rewards = {node: 0 for node in G.nodes()}
    rewards["Room 7"] = 50
    rewards["Room 5"] = 20
    nx.set_node_attributes(G, rewards, "reward")

    return G


def create_tree_graph():
    G = nx.Graph()
    nodes = [f"{NODE_PREFIX} {i}" for i in range(1, 8)]
    G.add_nodes_from(nodes)

    edges = [
        ("Room 1", "Room 2"), ("Room 1", "Room 3"),
        ("Room 2", "Room 4"), ("Room 2", "Room 5"),
        ("Room 3", "Room 6"), ("Room 3", "Room 7")
    ]
    G.add_edges_from(edges)

    rewards = {n: 0 for n in nodes}
    rewards["Room 7"] = 50
    rewards["Room 4"] = 15
    nx.set_node_attributes(G, rewards, "reward")

    return G


def create_clustered_graph():
    G = nx.Graph()
    cluster_size = 5

    nodes = []
    for i in range(3):
        cluster = nx.complete_graph(cluster_size)
        mapping = {j: f"{NODE_PREFIX} {i*cluster_size + j + 1}" for j in range(cluster_size)}
        cluster = nx.relabel_nodes(cluster, mapping)
        G = nx.union(G, cluster)
        nodes.extend(cluster.nodes())

        if i > 0:
            G.add_edge(nodes[i * cluster_size - 1], nodes[i * cluster_size])

    rewards = {n: 0 for n in G.nodes()}
    rewards[nodes[-1]] = 75
    rewards[nodes[3]] = 30
    nx.set_node_attributes(G, rewards, "reward")

    return G
