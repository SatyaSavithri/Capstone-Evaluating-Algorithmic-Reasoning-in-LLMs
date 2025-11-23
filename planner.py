# planner.py
import networkx as nx

def bfs_optimal_path_to_max_reward(G, start_node):
    """
    Compute BFS shortest path from a user-selected START node
    to the highest reward node.
    """
    rewards = nx.get_node_attributes(G, "reward")
    max_node = max(rewards, key=lambda n: rewards[n])

    try:
        return nx.shortest_path(G, source=start_node, target=max_node)
    except:
        return ["NO PATH"]
