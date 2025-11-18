# graphs.py
import networkx as nx
NODE_PREFIX = "Room"

def _room(i: int) -> str:
    return f"{NODE_PREFIX} {i}"

def create_line_graph(n_nodes: int = 7) -> nx.Graph:
    G = nx.path_graph(n_nodes)
    mapping = {i: _room(i+1) for i in range(n_nodes)}
    G = nx.relabel_nodes(G, mapping)
    rewards = {node: 0 for node in G.nodes()}
    rewards[_room(n_nodes)] = 50
    if n_nodes >= 3:
        rewards[_room(n_nodes-2)] = 20
    nx.set_node_attributes(G, rewards, "reward")
    return G

def create_tree_graph() -> nx.Graph:
    G = nx.Graph()
    edges = [
        (_room(1), _room(2)), (_room(1), _room(3)),
        (_room(2), _room(4)), (_room(2), _room(5)),
        (_room(3), _room(6)), (_room(3), _room(7))
    ]
    G.add_edges_from(edges)
    rewards = {node: 0 for node in G.nodes()}
    rewards[_room(7)] = 50
    rewards[_room(4)] = 15
    nx.set_node_attributes(G, rewards, "reward")
    return G

def create_clustered_graph(n_clusters: int = 3, cluster_size: int = 5) -> nx.Graph:
    G = nx.Graph()
    clusters = []
    for i in range(n_clusters):
        base = i * cluster_size
        clique = nx.complete_graph(cluster_size)
        mapping = {node: f"{NODE_PREFIX} {base + node + 1}" for node in clique.nodes()}
        clique = nx.relabel_nodes(clique, mapping)
        G = nx.union(G, clique, rename=(None, None))
        clusters.append(list(clique.nodes()))
    for i in range(len(clusters)-1):
        G.add_edge(clusters[i][-1], clusters[i+1][0])
    rewards = {node: 0 for node in G.nodes()}
    rewards[clusters[-1][-1]] = 75
    if len(clusters[0]) >= 4:
        rewards[clusters[0][3]] = 30
    nx.set_node_attributes(G, rewards, "reward")
    return G
