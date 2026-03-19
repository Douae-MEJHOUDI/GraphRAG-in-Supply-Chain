"""
propagator.py
-------------
DisruptionPropagator — the heart of Module 3 and the project's novel contribution.

This module models how a supply chain shock radiates outward from its origin
through the dependency graph, assigning every reachable node a numeric
"disruption score" that reflects its exposure to the original event.

Core algorithm
--------------
Weighted BFS from ground-zero nodes, following directed dependency edges
(supplies, depends_on, ships_through, located_in) with multiplicative decay:

    score(child) = score(parent) × decay × edge_weight

Where:
  - decay       : configurable [0, 1], default 0.6. Models the real-world
                  observation that disruptions lose force with each tier.
                  A value of 0.6 means a tier-2 supplier feels 60% of the
                  shock that the tier-1 supplier feels.
  - edge_weight : assigned by builder.assign_edge_weights() in Module 1.
                  Edges with no alternative supplier → weight ≈ 1.0 (critical).
                  Edges with many alternatives → weight → 0.25 (redundant).
                  ALTERNATIVE_TO edges → weight = 0.0 (excluded from propagation).

Why this is supply-chain-aware (not generic graph diffusion)
------------------------------------------------------------
Generic graph diffusion would give every edge the same weight.
Our propagator is different in two important ways:

  1. Edge direction matters: shock flows FORWARD along dependency edges
     (supplier → manufacturer → customer). It does not flow backward.
     This reflects physical reality: if your chip supplier is hit, you
     are affected; the miners who supply the chip fab's raw materials
     are not (directly) affected by the same shock.

  2. Edge weight encodes supply redundancy: a tier-1 supplier with zero
     alternatives gets full propagation weight. A supplier with 3
     alternatives gets 25% weight — the buyer can partially substitute.
     This makes the simulation quantitatively supply-chain-aware.

Propagation follows these edge types (in priority order):
  supplies, depends_on → direct dependency (full propagation)
  ships_through        → logistics dependency (partial, configurable)
  located_in           → geographic co-location risk (partial)
  affected_by          → direct event association (already ground-zero)

ALTERNATIVE_TO edges are NEVER followed for propagation — they represent
risk mitigation, not risk transmission.

Output
------
PropagationResult — contains:
  - scores dict {node: disruption_score}
  - exposure_tiers {CRITICAL/HIGH/MODERATE/LOW: [nodes]}
  - path_traces {node: [path from ground-zero]}  ← explainability
  - summary statistics
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from src.graph.schema import (
    EDGE_ATTR_RELATION,
    EDGE_ATTR_WEIGHT,
    NODE_ATTR_TYPE,
    RelationType,
    EntityType,
)
from src.simulation.events import (
    DisruptionEvent,
    SeverityLevel,
    DisruptionCategory,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DECAY          = 0.60   # score multiplier per hop
DEFAULT_MAX_HOPS       = 5      # maximum propagation depth
PRUNING_THRESHOLD      = 0.03   # stop propagating below this score
LOGISTICS_DECAY_FACTOR = 0.70   # extra damping for ships_through edges
LOCATION_DECAY_FACTOR  = 0.50   # extra damping for located_in edges

# Edge types that carry the disruption forward
PROPAGATION_EDGES = {
    RelationType.SUPPLIES.value,
    RelationType.DEPENDS_ON.value,
    RelationType.SHIPS_THROUGH.value,
    RelationType.LOCATED_IN.value,
}

# Edge types that are NEVER followed
BLOCKED_EDGES = {
    RelationType.ALTERNATIVE_TO.value,
    RelationType.SELLS_TO.value,      # downstream commercial → not a dependency
}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeExposure:
    """
    Full exposure record for a single graph node post-propagation.

    Attributes
    ----------
    node          : node name
    score         : disruption score [0, 1]
    severity      : SeverityLevel derived from score
    hop_distance  : minimum hops from any ground-zero node
    entity_type   : EntityType of this node
    path          : list of node names from ground-zero to this node
    has_alternative: whether this node has alternative_to edges
                     (reduces real-world impact despite high score)
    """
    node:            str
    score:           float
    severity:        SeverityLevel
    hop_distance:    int
    entity_type:     str
    path:            list[str]        = field(default_factory=list)
    has_alternative: bool             = False

    def __repr__(self) -> str:
        return (
            f"NodeExposure({self.node!r}, "
            f"score={self.score:.3f}, "
            f"severity={self.severity.value}, "
            f"hop={self.hop_distance})"
        )


@dataclass
class PropagationResult:
    """
    Complete output of a disruption propagation run.

    Attributes
    ----------
    event          : the DisruptionEvent that triggered this simulation
    scores         : {node_name: disruption_score}  — all affected nodes
    exposures      : {node_name: NodeExposure}       — full per-node detail
    tiers          : {SeverityLevel: [node_names]}   — grouped by severity
    path_traces    : {node_name: [path]}              — explainability traces
    stats          : summary statistics dict
    """
    event:       DisruptionEvent
    scores:      dict[str, float]
    exposures:   dict[str, NodeExposure]
    tiers:       dict[str, list[str]]
    path_traces: dict[str, list[str]]
    stats:       dict                  = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def critical_nodes(self) -> list[str]:
        return self.tiers.get(SeverityLevel.CRITICAL.value, [])

    def high_nodes(self) -> list[str]:
        return self.tiers.get(SeverityLevel.HIGH.value, [])

    def affected_nodes(self) -> list[str]:
        """All nodes with score above the pruning threshold."""
        return list(self.scores.keys())

    def top_n(self, n: int = 10) -> list[tuple[str, float]]:
        """Top-N most exposed nodes as (name, score) pairs."""
        return sorted(self.scores.items(), key=lambda x: -x[1])[:n]

    def path_explanation(self, node: str) -> str:
        """
        Human-readable dependency chain from ground-zero to a given node.
        Example: "Taiwan Earthquake → TSMC → Apple → Port of Los Angeles"
        """
        path = self.path_traces.get(node, [])
        if not path:
            return f"{node} (direct ground-zero)"
        return " → ".join(path)

    def exposure_summary(self) -> str:
        """Short summary string for logging and display."""
        return (
            f"Event: {self.event.name} "
            f"(shock={self.event.initial_shock:.2f})\n"
            f"  Affected nodes  : {len(self.scores)}\n"
            f"  Critical        : {len(self.critical_nodes())}\n"
            f"  High            : {len(self.high_nodes())}\n"
            f"  Max hops reached: {self.stats.get('max_hop_reached', '?')}\n"
        )


# ---------------------------------------------------------------------------
# DisruptionPropagator
# ---------------------------------------------------------------------------

class DisruptionPropagator:
    """
    Propagates a DisruptionEvent through the supply chain knowledge graph
    using weighted BFS with supply-redundancy-aware edge weights.

    Usage
    -----
    >>> propagator = DisruptionPropagator(G)
    >>> result = propagator.propagate(SCENARIO_LIBRARY["taiwan_earthquake"])
    >>> print(result.exposure_summary())
    >>> print(result.path_explanation("Apple"))
    """

    def __init__(
        self,
        graph:          nx.DiGraph,
        decay:          float = DEFAULT_DECAY,
        max_hops:       int   = DEFAULT_MAX_HOPS,
        prune_threshold: float = PRUNING_THRESHOLD,
    ):
        self.G               = graph
        self.decay           = decay
        self.max_hops        = max_hops
        self.prune_threshold = prune_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, event: DisruptionEvent) -> PropagationResult:
        """
        Run the full propagation simulation for a disruption event.

        Parameters
        ----------
        event : DisruptionEvent
            The disruption to simulate. Contains ground-zero nodes and
            initial shock severity.

        Returns
        -------
        PropagationResult
            Full exposure map with scores, severity tiers, path traces,
            and summary statistics.
        """
        # Resolve ground-zero: expand by affected_region if specified
        ground_zero_nodes = self._resolve_ground_zero(event)

        # Run weighted BFS
        scores, path_traces, hop_distances = self._weighted_bfs(
            ground_zero_nodes, event.initial_shock
        )

        # Build per-node exposure records
        exposures = self._build_exposures(scores, path_traces, hop_distances)

        # Group into severity tiers
        tiers = self._build_tiers(scores)

        # Compute summary statistics
        stats = self._compute_stats(scores, hop_distances, event)

        result = PropagationResult(
            event=event,
            scores=scores,
            exposures=exposures,
            tiers=tiers,
            path_traces=path_traces,
            stats=stats,
        )

        print(result.exposure_summary())
        return result

    def compare_scenarios(
        self, events: list[DisruptionEvent]
    ) -> dict[str, PropagationResult]:
        """
        Run multiple disruption scenarios and return all results keyed
        by event name. Useful for the evaluation section.
        """
        results = {}
        for event in events:
            print(f"[Propagator] Running scenario: {event.name}")
            results[event.name] = self.propagate(event)
        return results

    # ------------------------------------------------------------------
    # Ground-zero resolution
    # ------------------------------------------------------------------

    def _resolve_ground_zero(self, event: DisruptionEvent) -> list[str]:
        """
        Build the complete list of initially disrupted nodes.

        Steps:
        1. Start with event.ground_zero (explicit list)
        2. If event.affected_region is set, add all nodes that have a
           located_in edge pointing to that region (geographic co-location)
        3. Filter to nodes that actually exist in the graph
        """
        gz_set = set(event.ground_zero)

        if event.affected_region:
            region = event.affected_region
            for node in self.G.nodes():
                for _, target, data in self.G.out_edges(node, data=True):
                    if (target == region
                            and data.get(EDGE_ATTR_RELATION) ==
                            RelationType.LOCATED_IN.value):
                        gz_set.add(node)

        # Filter to nodes that exist in graph
        valid = [n for n in gz_set if n in self.G]
        missing = gz_set - set(valid)
        if missing:
            print(f"[Propagator] Warning: ground-zero nodes not in graph: {missing}")

        return valid

    # ------------------------------------------------------------------
    # Core weighted BFS
    # ------------------------------------------------------------------

    def _weighted_bfs(
        self,
        ground_zero: list[str],
        initial_shock: float,
    ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, int]]:
        """
        Weighted BFS propagation from ground-zero nodes.

        Queue entries: (node, current_score, hop_distance, path_so_far)

        At each step, for every outgoing edge of the current node that is
        in PROPAGATION_EDGES and not in BLOCKED_EDGES, compute:

            child_score = current_score
                          × self.decay
                          × edge_weight
                          × relation_type_factor

        where relation_type_factor dampens logistics and location edges
        relative to direct supply/dependency edges.

        Returns
        -------
        scores        : {node: highest_score_seen}
        path_traces   : {node: path_from_any_ground_zero}
        hop_distances : {node: minimum_hop_distance}
        """
        # Initialise with ground-zero nodes at full shock
        scores:        dict[str, float]      = {}
        path_traces:   dict[str, list[str]]  = {}
        hop_distances: dict[str, int]        = {}

        for gz_node in ground_zero:
            scores[gz_node]        = initial_shock
            path_traces[gz_node]   = [gz_node]
            hop_distances[gz_node] = 0

        # BFS queue: (node, score, hop, path)
        queue: list[tuple[str, float, int, list[str]]] = [
            (node, initial_shock, 0, [node])
            for node in ground_zero
        ]

        visited_at_score: dict[str, float] = dict(scores)  # prune revisits

        while queue:
            current_node, current_score, current_hop, current_path = (
                queue.pop(0)
            )

            # Depth limit
            if current_hop >= self.max_hops:
                continue

            # Follow every outgoing edge from current node
            for _, neighbor, edge_data in self.G.out_edges(
                current_node, data=True
            ):
                relation = edge_data.get(EDGE_ATTR_RELATION, "")

                # Skip blocked or irrelevant edge types
                if relation in BLOCKED_EDGES:
                    continue
                if relation not in PROPAGATION_EDGES:
                    continue

                # Retrieve criticality weight set by Module 1
                edge_weight = edge_data.get(EDGE_ATTR_WEIGHT, 1.0)
                if edge_weight == 0.0:
                    continue  # alternative_to edges (already 0 from builder)

                # Apply relation-type-specific damping
                rel_factor = self._relation_factor(relation)

                # Compute child score
                child_score = (
                    current_score
                    * self.decay
                    * edge_weight
                    * rel_factor
                )

                # Prune negligible scores
                if child_score < self.prune_threshold:
                    continue

                # Only update if we found a higher score path
                if child_score <= visited_at_score.get(neighbor, 0.0):
                    continue

                visited_at_score[neighbor] = child_score
                scores[neighbor]           = child_score
                new_path                   = current_path + [neighbor]
                path_traces[neighbor]      = new_path
                hop_distances[neighbor]    = current_hop + 1

                queue.append(
                    (neighbor, child_score, current_hop + 1, new_path)
                )

        return scores, path_traces, hop_distances

    def _relation_factor(self, relation: str) -> float:
        """
        Additional damping multiplier per relation type.

        Direct dependencies (supplies, depends_on) propagate at full decay.
        Logistics (ships_through) propagate at 70% — a port closure hurts
        but there are usually alternative routes.
        Geographic co-location (located_in) propagates at 50% — sharing a
        region creates risk but doesn't mean direct operational dependency.
        """
        if relation in {
            RelationType.SUPPLIES.value,
            RelationType.DEPENDS_ON.value,
        }:
            return 1.0
        if relation == RelationType.SHIPS_THROUGH.value:
            return LOGISTICS_DECAY_FACTOR
        if relation == RelationType.LOCATED_IN.value:
            return LOCATION_DECAY_FACTOR
        return 0.8  # default for any other included relation

    # ------------------------------------------------------------------
    # Result construction helpers
    # ------------------------------------------------------------------

    def _build_exposures(
        self,
        scores:        dict[str, float],
        path_traces:   dict[str, list[str]],
        hop_distances: dict[str, int],
    ) -> dict[str, NodeExposure]:
        """Build a NodeExposure record for every affected node."""
        exposures = {}
        for node, score in scores.items():
            etype = (
                self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
                if node in self.G else "Unknown"
            )

            # Check if this node has any alternative_to edges
            has_alt = any(
                data.get(EDGE_ATTR_RELATION) == RelationType.ALTERNATIVE_TO.value
                for _, _, data in self.G.out_edges(node, data=True)
            )

            exposures[node] = NodeExposure(
                node=node,
                score=round(score, 4),
                severity=SeverityLevel.from_score(score),
                hop_distance=hop_distances.get(node, 0),
                entity_type=etype,
                path=path_traces.get(node, [node]),
                has_alternative=has_alt,
            )
        return exposures

    def _build_tiers(
        self, scores: dict[str, float]
    ) -> dict[str, list[str]]:
        """Group nodes into severity tiers by score."""
        tiers: dict[str, list[str]] = {
            sl.value: [] for sl in SeverityLevel
        }
        for node, score in scores.items():
            tier = SeverityLevel.from_score(score).value
            tiers[tier].append(node)
        # Sort each tier by score descending
        for tier in tiers:
            tiers[tier].sort(key=lambda n: -scores[n])
        return tiers

    def _compute_stats(
        self,
        scores:        dict[str, float],
        hop_distances: dict[str, int],
        event:         DisruptionEvent,
    ) -> dict:
        """Compute summary statistics for the propagation result."""
        if not scores:
            return {"total_affected": 0}

        score_values = list(scores.values())
        return {
            "total_affected":    len(scores),
            "mean_score":        round(sum(score_values) / len(score_values), 4),
            "max_score":         round(max(score_values), 4),
            "min_score":         round(min(score_values), 4),
            "max_hop_reached":   max(hop_distances.values(), default=0),
            "critical_count":    sum(1 for s in score_values if s >= 0.80),
            "high_count":        sum(1 for s in score_values if 0.50 <= s < 0.80),
            "moderate_count":    sum(1 for s in score_values if 0.25 <= s < 0.50),
            "low_count":         sum(1 for s in score_values if s < 0.25),
            "event_category":    event.category.value,
            "initial_shock":     event.initial_shock,
        }
