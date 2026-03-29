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

import json
import re

import networkx as nx
from pyvis.network import Network
from env.spaces import ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE, ACTION_ESCALATE


_NETWORK_INIT_RE = re.compile(r"(var\\s+network\\s*=\\s*new\\s+vis\\.Network\\(container,\\s*data,\\s*options\\);)")
_DECISION_MARKER = "// __SOCIALGUARD_DECISION_UPDATE__"


def generate_graph_base_html(nx_graph: nx.Graph) -> str:
    """Generate a base Pyvis HTML blob once per episode."""
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="black")

    # Disable physics for large graphs to prevent infinite freezing in browser
    if len(nx_graph.nodes) > 100:
        net.toggle_physics(False)

    for node_id, data in nx_graph.nodes(data=True):
        is_bot = data.get("is_bot", False)
        color = "red" if is_bot else "green"
        title = f"Node: {node_id}\nBot: {is_bot}"
        net.add_node(
            node_id,
            label=str(node_id),
            color=color,
            title=title,
            size=15,
        )

    for u, v in nx_graph.edges:
        net.add_edge(u, v, color="#cccccc")

    html = net.generate_html()
    m = _NETWORK_INIT_RE.search(html)
    if not m:
        return html
    insert_at = m.end(1)
    return html[:insert_at] + "\n" + _DECISION_MARKER + "\n" + html[insert_at:]


def apply_decision_log(base_html: str, decision_log: dict[int, dict] | None) -> str:
    """Inject decision-log node recoloring JS into a cached base HTML string."""
    if not decision_log:
        return base_html.replace(_DECISION_MARKER, "")

    payload = json.dumps({str(int(k)): v for k, v in decision_log.items()})
    js = f"""
// Apply decision-log node styling (injected by SocialGuard-RL)
try {{
  const decisionLog = {payload};
  const updates = [];
  for (const [nodeIdStr, info] of Object.entries(decisionLog)) {{
    const nodeId = parseInt(nodeIdStr, 10);
    const action = info && typeof info.action === "number" ? info.action : {ACTION_ALLOW};
    let color = null;
    let size = 15;
    if (action === {ACTION_REMOVE}) {{
      color = "gray";
      size = 5;
    }} else if (action === {ACTION_WARN} || action === {ACTION_REDUCE_REACH} || action === {ACTION_ESCALATE}) {{
      color = "yellow";
    }}
    if (color !== null) {{
      updates.push({{ id: nodeId, color: color, size: size }});
    }}
  }}
  if (typeof nodes !== "undefined" && updates.length) {{
    nodes.update(updates);
  }}
}} catch (e) {{
  // ignore
}}
"""
    return base_html.replace(_DECISION_MARKER, js)

def generate_graph_html(nx_graph: nx.Graph, decision_log: dict[int, dict] = None) -> str:
    """Generate Pyvis HTML representation of the network graph."""
    base = generate_graph_base_html(nx_graph)
    return apply_decision_log(base, decision_log)
