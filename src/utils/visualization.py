"""
visualization.py
----------------
Graph visualization utilities for the supply chain knowledge graph.

Provides two rendering modes:
  1. Static matplotlib plot     — for notebook inline display
  2. Interactive pyvis HTML     — for the demo (nodes color-coded by
                                   disruption score, hoverable metadata)

Color scheme:
  Node color reflects EntityType (consistent across all visualizations)
  Node border / glow reflects disruption score (red → orange → yellow → gray)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.graph.schema import (
    NODE_ATTR_TYPE,
    EDGE_ATTR_RELATION,
    EntityType,
)


# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------

# One color per EntityType for consistent node coloring
ENTITY_COLOR_MAP: dict[str, str] = {
    EntityType.SUPPLIER.value:         "#6366f1",   # indigo
    EntityType.MANUFACTURER.value:     "#8b5cf6",   # purple
    EntityType.PART.value:             "#06b6d4",   # cyan
    EntityType.PORT.value:             "#f59e0b",   # amber
    EntityType.REGION.value:           "#10b981",   # emerald
    EntityType.LOGISTICS_ROUTE.value:  "#64748b",   # slate
    EntityType.DISRUPTION.value:       "#ef4444",   # red
    EntityType.CUSTOMER.value:         "#3b82f6",   # blue
    "Unknown":                         "#94a3b8",   # gray fallback
}

# Disruption score → border/highlight color
def disruption_color(score: float) -> str:
    """Map a [0, 1] disruption score to a hex color string."""
    if score >= 0.8:   return "#dc2626"   # red — critical
    if score >= 0.5:   return "#ea580c"   # orange — high
    if score >= 0.25:  return "#d97706"   # amber — moderate
    if score > 0.05:   return "#65a30d"   # yellow-green — low
    return "#94a3b8"                      # gray — unaffected


# ---------------------------------------------------------------------------
# Static matplotlib plot
# ---------------------------------------------------------------------------

def plot_graph(
    G: nx.DiGraph,
    title: str = "Supply Chain Knowledge Graph",
    disruption_scores: Optional[dict[str, float]] = None,
    highlight_nodes: Optional[list[str]] = None,
    figsize: tuple[int, int] = (16, 10),
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Render the supply chain graph using matplotlib.

    Parameters
    ----------
    G                  : nx.DiGraph  — the knowledge graph
    title              : str         — plot title
    disruption_scores  : dict        — {node: score} for disruption overlay
    highlight_nodes    : list[str]   — nodes to draw with a thick border
    figsize            : tuple       — matplotlib figure size
    save_path          : str | Path  — if given, save PNG to this path
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.axis("off")

    # Layout — spring layout with fixed seed for reproducibility
    pos = nx.spring_layout(G, seed=42, k=2.5)

    # ---- Node appearance ----
    node_colors  = []
    node_sizes   = []
    node_borders = []

    for node in G.nodes():
        etype  = G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
        color  = ENTITY_COLOR_MAP.get(etype, ENTITY_COLOR_MAP["Unknown"])
        node_colors.append(color)

        if disruption_scores and node in disruption_scores:
            score = disruption_scores[node]
            node_sizes.append(800 + int(score * 1200))
            node_borders.append(disruption_color(score))
        else:
            node_sizes.append(600)
            node_borders.append("#1e293b")

    # ---- Draw nodes ----
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_borders,
        linewidths=2.5,
        alpha=0.9,
    )

    # ---- Draw edges ----
    edge_colors = []
    for u, v, data in G.edges(data=True):
        rel = data.get(EDGE_ATTR_RELATION, "")
        if rel in {"supplies", "depends_on"}:
            edge_colors.append("#475569")
        elif rel == "affected_by":
            edge_colors.append("#ef4444")
        elif rel == "alternative_to":
            edge_colors.append("#10b981")
        else:
            edge_colors.append("#94a3b8")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color=edge_colors,
        arrows=True,
        arrowsize=15,
        arrowstyle="-|>",
        width=1.2,
        alpha=0.7,
        connectionstyle="arc3,rad=0.1",
    )

    # ---- Labels ----
    # Only show labels for nodes with high degree or high disruption score
    important_nodes = {
        n for n in G.nodes()
        if G.degree(n) >= 2
        or (disruption_scores and disruption_scores.get(n, 0) >= 0.3)
    }
    label_dict = {n: n for n in important_nodes}
    nx.draw_networkx_labels(
        G, pos, labels=label_dict, ax=ax,
        font_size=8, font_color="#0f172a",
    )

    # ---- Legend — entity types ----
    legend_handles = [
        mpatches.Patch(color=c, label=t)
        for t, c in ENTITY_COLOR_MAP.items()
        if t != "Unknown"
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=8,
        title="Entity type",
        framealpha=0.85,
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[Viz] Plot saved → {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Interactive pyvis HTML
# ---------------------------------------------------------------------------

def plot_graph_interactive(
    G: nx.DiGraph,
    disruption_scores: Optional[dict[str, float]] = None,
    title: str = "Supply Chain Knowledge Graph",
    save_path: str | Path = "outputs/graphs/interactive.html",
    height: str = "750px",
) -> str:
    """
    Generate an interactive pyvis HTML graph with:
      - Node colors by EntityType
      - Node border colors by disruption score
      - Hover tooltips showing all node attributes
      - Edge labels showing relation type

    Parameters
    ----------
    G                 : nx.DiGraph
    disruption_scores : dict {node: float}  — if provided, colors borders
    title             : str
    save_path         : path to write the HTML file
    height            : pyvis canvas height

    Returns
    -------
    str : absolute path to the generated HTML file
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis not installed. Run: pip install pyvis")

    net = Network(
        height=height,
        width="100%",
        directed=True,
        notebook=True,
        heading=title,
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 200
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      },
      "edges": {
        "smooth": { "type": "continuous" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }
      }
    }
    """)

    for node, attrs in G.nodes(data=True):
        etype  = attrs.get(NODE_ATTR_TYPE, "Unknown")
        color  = ENTITY_COLOR_MAP.get(etype, ENTITY_COLOR_MAP["Unknown"])
        score  = (disruption_scores or {}).get(node, 0.0)
        border = disruption_color(score)
        size   = 20 + int(score * 30)

        tooltip = (
            f"<b>{node}</b><br>"
            f"Type: {etype}<br>"
            f"Disruption score: {score:.2f}<br>"
            f"Degree: {G.degree(node)}"
        )

        net.add_node(
            node,
            label=node,
            color={"background": color, "border": border},
            size=size,
            title=tooltip,
        )

    for u, v, data in G.edges(data=True):
        rel   = data.get(EDGE_ATTR_RELATION, "")
        wt    = data.get("weight", 1.0)
        width = 1.0 + wt * 2.5
        net.add_edge(u, v, label=rel, width=width, title=f"{rel} (w={wt:.2f})")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(save_path))
    print(f"[Viz] Interactive graph saved → {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# Subgraph helper — extract and display a neighbourhood
# ---------------------------------------------------------------------------

def plot_ego_subgraph(
    G: nx.DiGraph,
    center_node: str,
    radius: int = 2,
    disruption_scores: Optional[dict[str, float]] = None,
    **kwargs,
) -> None:
    """
    Extract and plot the ego subgraph around a given node.
    Useful for inspecting a supplier's immediate neighbourhood.
    """
    if center_node not in G:
        print(f"[Viz] Node '{center_node}' not found in graph.")
        return

    ego = nx.ego_graph(G.to_undirected(), center_node, radius=radius)
    subgraph = G.subgraph(ego.nodes()).copy()

    title = f"Ego subgraph: {center_node} (radius={radius})"
    plot_graph(subgraph, title=title, disruption_scores=disruption_scores, **kwargs)
