"""
visualizer.py
-------------
SimulationVisualizer — renders disruption propagation results as visual outputs.

Three visualization modes:

  1. disruption_map()   — full graph with nodes colored by exposure severity.
                          The demo-ready visualization: nodes glow red→orange→
                          yellow→gray as the disruption propagates outward.

  2. propagation_tree() — directed tree showing only affected nodes and the
                          paths through which the disruption traveled.
                          Best for explaining "why is this node affected?"

  3. risk_dashboard()   — summary matplotlib figure with four panels:
                          - Bar chart of top-10 most exposed nodes
                          - Severity tier pie chart
                          - Hop-distance distribution histogram
                          - Entity-type exposure heatmap

All functions save to outputs/simulation/ and also render inline in Jupyter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from src.graph.schema import NODE_ATTR_TYPE, EDGE_ATTR_RELATION
from src.simulation.events import SeverityLevel
from src.simulation.propagator import PropagationResult


# ---------------------------------------------------------------------------
# Color maps — consistent across all visualizations
# ---------------------------------------------------------------------------

# Severity → node fill color
SEVERITY_COLORS: dict[str, str] = {
    SeverityLevel.CRITICAL.value: "#dc2626",   # red
    SeverityLevel.HIGH.value:     "#ea580c",   # orange
    SeverityLevel.MODERATE.value: "#d97706",   # amber
    SeverityLevel.LOW.value:      "#65a30d",   # yellow-green
    "Unaffected":                 "#d1d5db",   # gray
}

# Entity type → node border color (same palette as visualization.py Module 1)
ENTITY_BORDER_COLORS: dict[str, str] = {
    "Supplier":         "#6366f1",
    "Manufacturer":     "#8b5cf6",
    "Part":             "#06b6d4",
    "Port":             "#f59e0b",
    "Region":           "#10b981",
    "LogisticsRoute":   "#64748b",
    "DisruptionEvent":  "#ef4444",
    "Customer":         "#3b82f6",
    "Unknown":          "#94a3b8",
}

OUTPUT_DIR = Path("outputs/simulation")


# ---------------------------------------------------------------------------
# SimulationVisualizer
# ---------------------------------------------------------------------------

class SimulationVisualizer:
    """
    Renders disruption propagation results as matplotlib figures.

    Usage
    -----
    >>> viz = SimulationVisualizer(G)
    >>> viz.disruption_map(result, save=True)
    >>> viz.risk_dashboard(result, save=True)
    >>> viz.propagation_tree(result, target_node="Apple", save=True)
    """

    def __init__(self, graph: nx.DiGraph):
        self.G = graph
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Disruption map — full graph colored by severity
    # ------------------------------------------------------------------

    def disruption_map(
        self,
        result:    PropagationResult,
        figsize:   tuple = (18, 11),
        save:      bool  = True,
        save_path: Optional[Path] = None,
    ) -> None:
        """
        Render the full supply chain graph with nodes colored by disruption
        exposure severity. This is the primary demo visualization.

        Color scheme:
          Red    = Critical (score ≥ 0.8) — direct or near-direct exposure
          Orange = High     (score ≥ 0.5) — 1–2 hop dependency
          Amber  = Moderate (score ≥ 0.25) — 2–3 hop dependency
          Green  = Low      (score > 0.03) — peripheral exposure
          Gray   = Unaffected              — no propagation reached
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")

        ax.set_title(
            f"Disruption Exposure Map — {result.event.name}\n"
            f"(Initial shock: {result.event.initial_shock:.0%}  |  "
            f"Affected nodes: {len(result.scores)})",
            fontsize=13, fontweight="bold", pad=14,
        )
        ax.axis("off")

        pos = nx.spring_layout(self.G, seed=42, k=2.5)

        # ---- Node styling ----
        node_fills   = []
        node_sizes   = []
        node_borders = []

        for node in self.G.nodes():
            etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")

            if node in result.scores:
                score    = result.scores[node]
                severity = SeverityLevel.from_score(score).value
                fill     = SEVERITY_COLORS[severity]
                size     = int(400 + score * 1400)   # bigger = more affected
            else:
                fill = SEVERITY_COLORS["Unaffected"]
                size = 350

            node_fills.append(fill)
            node_sizes.append(size)
            node_borders.append(ENTITY_BORDER_COLORS.get(etype, "#94a3b8"))

        # ---- Draw nodes ----
        nx.draw_networkx_nodes(
            self.G, pos, ax=ax,
            node_color=node_fills,
            node_size=node_sizes,
            edgecolors=node_borders,
            linewidths=2.0,
            alpha=0.92,
        )

        # ---- Draw edges ----
        # Highlight edges that carry the propagation (affected→affected)
        prop_edges, prop_colors = [], []
        other_edges, other_colors = [], []

        for u, v, data in self.G.edges(data=True):
            rel = data.get(EDGE_ATTR_RELATION, "")
            if u in result.scores and v in result.scores:
                prop_edges.append((u, v))
                prop_colors.append("#dc2626")
            else:
                other_edges.append((u, v))
                other_colors.append("#e2e8f0")

        if other_edges:
            nx.draw_networkx_edges(
                self.G, pos, edgelist=other_edges,
                edge_color=other_colors, ax=ax,
                arrows=True, arrowsize=10,
                width=0.8, alpha=0.4,
                connectionstyle="arc3,rad=0.1",
            )
        if prop_edges:
            nx.draw_networkx_edges(
                self.G, pos, edgelist=prop_edges,
                edge_color=prop_colors, ax=ax,
                arrows=True, arrowsize=14,
                width=1.8, alpha=0.75,
                connectionstyle="arc3,rad=0.1",
            )

        # ---- Labels — only show affected nodes ----
        label_dict = {
            n: n for n in self.G.nodes()
            if n in result.scores or self.G.degree(n) >= 3
        }
        nx.draw_networkx_labels(
            self.G, pos, labels=label_dict, ax=ax,
            font_size=8, font_color="#0f172a", font_weight="500",
        )

        # ---- Legend ----
        severity_handles = [
            mpatches.Patch(
                color=SEVERITY_COLORS[sl.value],
                label=f"{sl.value} exposure"
            )
            for sl in SeverityLevel
        ]
        severity_handles.append(
            mpatches.Patch(color=SEVERITY_COLORS["Unaffected"], label="Unaffected")
        )

        legend1 = ax.legend(
            handles=severity_handles,
            loc="upper left", fontsize=9,
            title="Exposure severity", title_fontsize=9,
            framealpha=0.9,
        )
        ax.add_artist(legend1)

        # Disruption stats box
        stats = result.stats
        stats_text = (
            f"Disruption stats\n"
            f"Category : {result.event.category.value}\n"
            f"Critical : {stats.get('critical_count', 0)} nodes\n"
            f"High     : {stats.get('high_count', 0)} nodes\n"
            f"Moderate : {stats.get('moderate_count', 0)} nodes\n"
            f"Max hops : {stats.get('max_hop_reached', 0)}"
        )
        ax.text(
            0.99, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#94a3b8", alpha=0.9),
        )

        plt.tight_layout()
        if save:
            p = save_path or (
                OUTPUT_DIR / f"disruption_map_{result.event.name.replace(' ', '_')}.png"
            )
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"[Viz] Disruption map saved → {p}")
        plt.show()

    # ------------------------------------------------------------------
    # 2. Propagation tree — path traces for a target node
    # ------------------------------------------------------------------

    def propagation_tree(
        self,
        result:      PropagationResult,
        target_node: Optional[str] = None,
        max_paths:   int  = 8,
        figsize:     tuple = (14, 7),
        save:        bool  = True,
    ) -> None:
        """
        Show only the subgraph of nodes and edges that form the
        disruption propagation paths, highlighting how the shock
        traveled from ground-zero to affected nodes.

        If target_node is given, only the path to that node is shown.
        Otherwise, the top max_paths most-affected nodes are shown.
        """
        if target_node:
            paths_to_show = {target_node: result.path_traces.get(target_node, [])}
        else:
            top_nodes = result.top_n(max_paths)
            paths_to_show = {
                n: result.path_traces.get(n, []) for n, _ in top_nodes
            }

        # Build a tree graph from the path nodes and edges
        tree = nx.DiGraph()
        for node, path in paths_to_show.items():
            for i, n in enumerate(path):
                etype = self.G.nodes[n].get(NODE_ATTR_TYPE, "Unknown") if n in self.G else "Unknown"
                tree.add_node(n, entity_type=etype)
                if i > 0:
                    tree.add_edge(path[i - 1], n)

        if tree.number_of_nodes() == 0:
            print("[Viz] No propagation paths to visualize.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis("off")
        title = (
            f"Propagation path to '{target_node}'"
            if target_node
            else f"Top-{max_paths} propagation paths — {result.event.name}"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

        try:
            pos = nx.nx_agraph.graphviz_layout(tree, prog="dot")
        except Exception:
            pos = nx.spring_layout(tree, seed=42, k=3)

        node_colors = []
        node_sizes  = []
        for node in tree.nodes():
            if node in result.scores:
                sev   = SeverityLevel.from_score(result.scores[node]).value
                color = SEVERITY_COLORS[sev]
                size  = int(600 + result.scores[node] * 800)
            else:
                color = SEVERITY_COLORS["Unaffected"]
                size  = 500
            node_colors.append(color)
            node_sizes.append(size)

        nx.draw(
            tree, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            with_labels=True, font_size=9, font_weight="500",
            edge_color="#475569", arrows=True, arrowsize=18,
            width=1.5, connectionstyle="arc3,rad=0.05",
        )

        # Score annotations
        score_labels = {
            n: f"{result.scores[n]:.2f}"
            for n in tree.nodes() if n in result.scores
        }
        offset_pos = {n: (x, y - 18) for n, (x, y) in pos.items()}
        nx.draw_networkx_labels(
            tree, offset_pos, labels=score_labels, ax=ax,
            font_size=7, font_color="#64748b",
        )

        plt.tight_layout()
        if save:
            label = target_node or "top_paths"
            p = OUTPUT_DIR / f"prop_tree_{result.event.name.replace(' ','_')}_{label}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"[Viz] Propagation tree saved → {p}")
        plt.show()

    # ------------------------------------------------------------------
    # 3. Risk dashboard — four-panel summary figure
    # ------------------------------------------------------------------

    def risk_dashboard(
        self,
        result:  PropagationResult,
        top_n:   int   = 10,
        figsize: tuple = (16, 10),
        save:    bool  = True,
    ) -> None:
        """
        Four-panel summary dashboard for a propagation result:
          Panel A (top-left)  : Horizontal bar chart — top-N most exposed nodes
          Panel B (top-right) : Pie chart — severity tier distribution
          Panel C (bottom-left): Histogram — score distribution across all affected nodes
          Panel D (bottom-right): Grouped bar — exposure by entity type
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(
            f"Risk Dashboard — {result.event.name}  "
            f"({result.event.category.value})",
            fontsize=14, fontweight="bold", y=1.01,
        )

        # ---- Panel A: Top-N bar chart ----
        ax = axes[0][0]
        top_nodes = result.top_n(top_n)
        names  = [n for n, _ in top_nodes]
        scores = [s for _, s in top_nodes]
        colors = [
            SEVERITY_COLORS[SeverityLevel.from_score(s).value]
            for s in scores
        ]
        bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1],
                       edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Disruption score", fontsize=9)
        ax.set_title(f"Top {top_n} most exposed nodes", fontsize=10, fontweight="500")
        ax.set_xlim(0, 1.05)
        ax.axvline(0.8, color="#dc2626", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(0.5, color="#ea580c", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.tick_params(axis="y", labelsize=8)
        for bar, score in zip(bars[::-1], scores):
            ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va="center", fontsize=7, color="#374151")

        # ---- Panel B: Severity pie chart ----
        ax = axes[0][1]
        tier_counts = {
            sl.value: len(result.tiers.get(sl.value, []))
            for sl in SeverityLevel
        }
        sizes  = [v for v in tier_counts.values() if v > 0]
        labels = [k for k, v in tier_counts.items() if v > 0]
        pie_colors = [SEVERITY_COLORS[l] for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_colors,
            autopct="%1.0f%%", startangle=90,
            textprops={"fontsize": 9},
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(8)
        ax.set_title("Severity tier distribution", fontsize=10, fontweight="500")

        # ---- Panel C: Score distribution histogram ----
        ax = axes[1][0]
        all_scores = list(result.scores.values())
        if not all_scores:
            ax.text(0.5, 0.5, "No nodes affected", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10,
                    color="#94a3b8")
        else:
            n_unique = len(set(round(s, 3) for s in all_scores))
            n_bins   = max(1, min(20, len(all_scores), n_unique))
            ax.hist(all_scores, bins=n_bins, color="#7F77DD", edgecolor="white",
                    linewidth=0.5, alpha=0.85)

        # ---- Panel D: Exposure by entity type ----
        ax = axes[1][1]
        type_scores: dict[str, list[float]] = {}
        for node, score in result.scores.items():
            if node in self.G:
                etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
                type_scores.setdefault(etype, []).append(score)

        if type_scores:
            types     = list(type_scores.keys())
            avg_scores = [np.mean(v) for v in type_scores.values()]
            max_scores = [max(v) for v in type_scores.values()]
            counts     = [len(v) for v in type_scores.values()]

            x = np.arange(len(types))
            w = 0.35
            bars1 = ax.bar(x - w/2, avg_scores, w, label="Avg score",
                           color="#7F77DD", alpha=0.85, edgecolor="white")
            bars2 = ax.bar(x + w/2, max_scores, w, label="Max score",
                           color="#D85A30", alpha=0.85, edgecolor="white")
            ax.set_xticks(x)
            ax.set_xticklabels(types, rotation=25, ha="right", fontsize=8)
            ax.set_ylabel("Disruption score", fontsize=9)
            ax.set_ylim(0, 1.1)
            ax.set_title("Exposure by entity type", fontsize=10, fontweight="500")
            ax.legend(fontsize=8)
            ax.tick_params(axis="y", labelsize=8)

            # Annotate with node counts
            for i, (bar, cnt) in enumerate(zip(bars1, counts)):
                ax.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        f"n={cnt}", ha="center", va="bottom",
                        fontsize=6.5, color="#374151")

        plt.tight_layout()
        if save:
            p = OUTPUT_DIR / f"risk_dashboard_{result.event.name.replace(' ','_')}.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"[Viz] Risk dashboard saved → {p}")
        plt.show()

    # ------------------------------------------------------------------
    # 4. Scenario comparison — side-by-side bar charts
    # ------------------------------------------------------------------

    def compare_scenarios(
        self,
        results:  dict[str, PropagationResult],
        top_n:    int   = 8,
        figsize:  tuple = (16, 6),
        save:     bool  = True,
    ) -> None:
        """
        Side-by-side bar charts comparing exposure scores across
        multiple disruption scenarios for the same set of nodes.
        Useful for the evaluation section of the project.
        """
        # Collect the union of top-N affected nodes across all scenarios
        all_top_nodes: set[str] = set()
        for r in results.values():
            all_top_nodes.update(n for n, _ in r.top_n(top_n))
        nodes = sorted(all_top_nodes)

        n_scenarios = len(results)
        fig, ax = plt.subplots(figsize=figsize)
        x  = np.arange(len(nodes))
        w  = 0.8 / n_scenarios

        scenario_colors = [
            "#7F77DD", "#D85A30", "#1D9E75",
            "#378ADD", "#BA7517", "#D4537E",
        ]

        for i, (name, result) in enumerate(results.items()):
            scores_for_nodes = [result.scores.get(n, 0.0) for n in nodes]
            offset = (i - n_scenarios / 2 + 0.5) * w
            ax.bar(
                x + offset, scores_for_nodes, w,
                label=name,
                color=scenario_colors[i % len(scenario_colors)],
                alpha=0.85, edgecolor="white", linewidth=0.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Disruption score", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.axhline(0.8, color="#dc2626", linestyle="--",
                   linewidth=0.8, alpha=0.5, label="Critical threshold")
        ax.set_title(
            "Scenario comparison — node exposure across disruption events",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=9, loc="upper right")
        ax.tick_params(axis="y", labelsize=9)

        plt.tight_layout()
        if save:
            p = OUTPUT_DIR / "scenario_comparison.png"
            plt.savefig(p, dpi=150, bbox_inches="tight")
            print(f"[Viz] Scenario comparison saved → {p}")
        plt.show()
