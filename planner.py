# planner.py
import networkx as nx
START_NODE = "Room 1"

def node_with_max_reward(G: nx.Graph) -> str:
    rewards = nx.get_node_attributes(G, "reward")
    if not rewards:
        return START_NODE
    return max(rewards.items(), key=lambda x: x[1])[0]

def bfs_optimal_path_to_max_reward(G: nx.Graph, start: str = START_NODE):
    try:
        target = node_with_max_reward(G)
        return nx.shortest_path(G, source=start, target=target)
    except Exception:
        return []
