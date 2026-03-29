"""
dashboard/graph_view.py — Network graph component using pyvis.

Renders the social graph with nodes coloured by their true state and the
recent actions taken by the agent.

Colors:
    Green: Real user (allow/no action)
    Red: Confirmed bot (allow/no action)
    Yellow: Under review (warn, reduce_reach, escalate)
    Gray: Removed
"""

import networkx as nx
from pyvis.network import Network
from env.spaces import ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE, ACTION_ESCALATE


def generate_graph_html(nx_graph: nx.Graph, decision_log: dict[int, dict] = None) -> str:
    """Generate Pyvis HTML representation of the network graph.
    
    Args:
        nx_graph: NetworkX graph from sim.social_graph
        decision_log: Dict mapping node_id to decision info (action_taken, ground_truth, etc.)
            
    Returns:
        HTML string of the rendered pyvis network.
    """
    if decision_log is None:
        decision_log = {}

    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")
    
    # Disable physics for large graphs to prevent infinite freezing in browser
    if len(nx_graph.nodes) > 100:
        net.toggle_physics(False)

    for node_id, data in nx_graph.nodes(data=True):
        is_bot = data.get("is_bot", False)
        
        # Default colours based on ground truth
        color = "red" if is_bot else "green"
        title = f"Node: {node_id}\nBot: {is_bot}"
        
        # Override with decision log if available
        if node_id in decision_log:
            action = decision_log[node_id].get("action", ACTION_ALLOW)
            if action == ACTION_REMOVE:
                color = "gray"
            elif action in (ACTION_WARN, ACTION_REDUCE_REACH, ACTION_ESCALATE):
                color = "yellow"
            
            title += f"\nAction: {action}"
            
        net.add_node(
            node_id,
            label=str(node_id),
            color=color,
            title=title,
            size=15 if color != "gray" else 5
        )

    for u, v in nx_graph.edges:
        net.add_edge(u, v, color="#cccccc")

    return net.generate_html()
