"""
propagator.py
-------------
DisruptionPropagator — the heart of Module 3 and the project's novel contribution.

This module models how a supply chain shock radiates outward from its origin
through the dependency graph, assigning every reachable node a numeric
"disruption score" that reflects its exposure to the original event.

Core algorithm — Model 3: Attenuated Bottleneck Routing
--------------------------------------------------------
Modified Dijkstra's traversal from ground-zero nodes, following directed
dependency edges with the Hybrid Wave Propagation formula:

    F_v = min(F_u, C_{u,v}) · γ

Where:
  - F_u      : flow (disruption score) arriving at the parent node u
  - C_{u,v}  : effective edge capacity = edge_weight × relation_type_factor
                  edge_weight  : 1 / (1 + num_alternatives) — set by Module 1.
                                 No alternatives → weight ≈ 1.0 (critical).
                                 Many alternatives → weight → 0.25 (redundant).
                  rel_factor   : damping for logistics (0.70) / location (0.50)
                                 edges relative to direct supply/dependency (1.0).
  - γ (gamma): global decay factor, default 0.85. Models the real-world
               observation that disruptions lose force with each additional hop.

Why min() instead of multiplication?
-------------------------------------
  - Model 1 (multiplicative): score(child) = score(parent) × decay × weight
    Problem: fractions multiply across hops → score vanishes near-instantly.
    Deep, multi-tier failures are completely missed.

  - Model 2 (simple bottleneck): I_N = min(max_parent_flow, B_final_link)
    Problem: crisis travels with 100% force stopped only by direct bottlenecks,
    ignoring physical distance. Produces false positives far downstream.

  - Model 3 (this implementation): F_v = min(F_u, C_{u,v}) · γ
    The min() caps flow at the weakest link seen so far (bottleneck awareness),
    while γ ensures overall dissipation still happens across hops (wave
    attenuation). Together they give visibility without false positives.

Why Modified Dijkstra's instead of BFS?
-----------------------------------------
BFS visits nodes in arrival order, which means a weak long path can update a
node before the stronger short path is processed. Modified Dijkstra's uses a
max-heap (highest flow first), guaranteeing that the first time a node is
settled it holds the globally-optimal (highest-flow) path — exactly analogous
to Dijkstra's shortest-path guarantee but for maximum flow.

Propagation follows these edge types (in priority order):
  supplies, depends_on → direct dependency (rel_factor = 1.0)
  ships_through        → logistics dependency (rel_factor = 0.70)
  located_in           → geographic co-location risk (rel_factor = 0.50)

ALTERNATIVE_TO and SELLS_TO edges are NEVER followed.

Output
------
PropagationResult — contains:
  - scores dict {node: disruption_score}
  - exposure_tiers {CRITICAL/HIGH/MODERATE/LOW: [nodes]}
  - path_traces {node: [path from ground-zero]}  ← explainability
  - summary statistics

Reference
---------
Pressure Wave Propagation Optimization Models for Supply Chain Risk Mitigation,
MDPI, March 2026.
"""

from __future__ import annotations

import heapq
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

DEFAULT_DECAY          = 0.85   # γ — global attenuation per hop
DEFAULT_MAX_HOPS       = 5      # maximum propagation depth
PRUNING_THRESHOLD      = 0.03   # stop propagating below this score
LOGISTICS_DECAY_FACTOR = 0.70   # rel_factor for ships_through edges
LOCATION_DECAY_FACTOR  = 0.50   # rel_factor for located_in edges

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
    RelationType.SELLS_TO.value,
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
    node           : node name
    score          : disruption score [0, 1]
    severity       : SeverityLevel derived from score
    hop_distance   : minimum hops from any ground-zero node
    entity_type    : EntityType of this node
    path           : list of node names from ground-zero to this node
    has_alternative: whether this node has alternative_to edges
    """
    node:            str
    score:           float
    severity:        SeverityLevel
    hop_distance:    int
    entity_type:     str
    path:            list[str] = field(default_factory=list)
    has_alternative: bool      = False

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
    event       : the DisruptionEvent that triggered this simulation
    scores      : {node_name: disruption_score}  — all affected nodes
    exposures   : {node_name: NodeExposure}       — full per-node detail
    tiers       : {SeverityLevel: [node_names]}   — grouped by severity
    path_traces : {node_name: [path]}              — explainability traces
    stats       : summary statistics dict
    """
    event:       DisruptionEvent
    scores:      dict[str, float]
    exposures:   dict[str, NodeExposure]
    tiers:       dict[str, list[str]]
    path_traces: dict[str, list[str]]
    stats:       dict = field(default_factory=dict)

    def critical_nodes(self) -> list[str]:
        return self.tiers.get(SeverityLevel.CRITICAL.value, [])

    def high_nodes(self) -> list[str]:
        return self.tiers.get(SeverityLevel.HIGH.value, [])

    def affected_nodes(self) -> list[str]:
        return list(self.scores.keys())

    def top_n(self, n: int = 10) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda x: -x[1])[:n]

    def path_explanation(self, node: str) -> str:
        path = self.path_traces.get(node, [])
        if not path:
            return f"{node} (direct ground-zero)"
        return " → ".join(path)

    def exposure_summary(self) -> str:
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
    using Attenuated Bottleneck Routing (Model 3):

        F_v = min(F_u, C_{u,v}) · γ

    traversed via a Modified Dijkstra's algorithm (max-heap on flow).

    Usage
    -----
    >>> propagator = DisruptionPropagator(G)
    >>> result = propagator.propagate(SCENARIO_LIBRARY["taiwan_earthquake"])
    >>> print(result.exposure_summary())
    >>> print(result.path_explanation("Apple"))
    """

    def __init__(
        self,
        graph:           nx.DiGraph,
        decay:           float = DEFAULT_DECAY,
        max_hops:        int   = DEFAULT_MAX_HOPS,
        prune_threshold: float = PRUNING_THRESHOLD,
    ):
        self.G               = graph
        self.decay           = decay       # γ
        self.max_hops        = max_hops
        self.prune_threshold = prune_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, event: DisruptionEvent) -> PropagationResult:
        """
        Run the full Attenuated Bottleneck Routing simulation for one event.

        Parameters
        ----------
        event : DisruptionEvent

        Returns
        -------
        PropagationResult
        """
        ground_zero_nodes = self._resolve_ground_zero(event)

        scores, path_traces, hop_distances = self._attenuated_bottleneck_dijkstra(
            ground_zero_nodes, event.initial_shock
        )

        exposures = self._build_exposures(scores, path_traces, hop_distances)
        tiers     = self._build_tiers(scores)
        stats     = self._compute_stats(scores, hop_distances, event)

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

        1. Start with event.ground_zero — nodes must already be valid graph
           node names (semantic resolution of unrecognised names is done
           upstream by SimulationEngine before propagation starts).
        2. Collect location seeds (affected_region + Region-typed GZ nodes).
        3. Expand each seed to all co-located entities via reverse located_in.
        4. Filter to nodes that exist in the graph; warn on any that don't.
        """
        gz_set: set[str] = set(event.ground_zero)
        location_seeds: set[str] = set()

        if event.affected_region:
            location_seeds.add(event.affected_region)

        for gz_node in list(gz_set):
            if gz_node not in self.G:
                continue
            gz_type = str(self.G.nodes[gz_node].get(NODE_ATTR_TYPE, ""))
            if gz_type == EntityType.REGION.value:
                location_seeds.add(gz_node)
            for _, target, data in self.G.out_edges(gz_node, data=True):
                if data.get(EDGE_ATTR_RELATION) == RelationType.LOCATED_IN.value:
                    location_seeds.add(target)

        if location_seeds:
            gz_set |= self._expand_location_ground_zero(location_seeds)

        valid   = [n for n in gz_set if n in self.G]
        missing = gz_set - set(valid)
        if missing:
            print(f"[Propagator] Warning: ground-zero nodes not in graph: {missing}")
        return valid

    def _expand_location_ground_zero(self, location_seeds: set[str]) -> set[str]:
        """Expand location seeds via reverse located_in edges (recursive)."""
        expanded: set[str] = set()
        queue = list(location_seeds)
        seen:  set[str] = set()

        while queue:
            location = queue.pop(0)
            if location in seen:
                continue
            seen.add(location)
            if location in self.G:
                expanded.add(location)
            for source, _, data in self.G.in_edges(location, data=True):
                if data.get(EDGE_ATTR_RELATION) != RelationType.LOCATED_IN.value:
                    continue
                expanded.add(source)
                if str(self.G.nodes[source].get(NODE_ATTR_TYPE, "")) == EntityType.REGION.value:
                    queue.append(source)

        return expanded

    # ------------------------------------------------------------------
    # Core — Modified Dijkstra's with Attenuated Bottleneck Routing
    # ------------------------------------------------------------------

    def _attenuated_bottleneck_dijkstra(
        self,
        ground_zero:   list[str],
        initial_shock: float,
    ) -> tuple[dict[str, float], dict[str, list[str]], dict[str, int]]:
        """
        Modified Dijkstra's algorithm implementing Attenuated Bottleneck Routing.

        Formula per edge (u → v):
            F_v = min(F_u, C_{u,v}) · γ

        Where:
          F_u     : flow at parent node u (disruption score already settled)
          C_{u,v} : effective edge capacity
                      = edge_weight × relation_type_factor
          γ       : self.decay (global attenuation per hop)

        Algorithm:
          - Max-heap ordered by current flow (negated for Python's min-heap).
          - On each pop, if the stored flow is stale (a better path was already
            found), skip. Otherwise settle the node and relax its neighbours.
          - This guarantees that the first settlement of any node holds the
            globally highest-flow path, identical to Dijkstra's shortest-path
            guarantee but for maximum flow.

        Returns
        -------
        scores        : {node: highest flow seen}
        path_traces   : {node: path from any ground-zero node}
        hop_distances : {node: minimum hop distance}
        """
        scores:        dict[str, float]     = {}
        path_traces:   dict[str, list[str]] = {}
        hop_distances: dict[str, int]       = {}

        # Seed the heap with ground-zero nodes at full initial shock
        for gz in ground_zero:
            scores[gz]        = initial_shock
            path_traces[gz]   = [gz]
            hop_distances[gz] = 0

        # heap entries: (-flow, hop, node, path)
        # Negated flow so heappop gives us the node with HIGHEST flow.
        heap: list[tuple[float, int, str, list[str]]] = [
            (-initial_shock, 0, gz, [gz]) for gz in ground_zero
        ]
        heapq.heapify(heap)

        while heap:
            neg_flow, hop, node, path = heapq.heappop(heap)
            current_flow = -neg_flow

            # Stale entry — a better path has already settled this node
            if current_flow < scores.get(node, 0.0):
                continue

            # Depth limit
            if hop >= self.max_hops:
                continue

            for _, neighbor, edge_data in self.G.out_edges(node, data=True):
                relation = edge_data.get(EDGE_ATTR_RELATION, "")

                if relation in BLOCKED_EDGES:
                    continue
                if relation not in PROPAGATION_EDGES:
                    continue

                edge_weight = edge_data.get(EDGE_ATTR_WEIGHT, 1.0)
                if edge_weight == 0.0:
                    continue

                # Effective capacity: edge criticality × relation-type damping
                rel_factor = self._relation_factor(relation)
                capacity   = edge_weight * rel_factor

                # Attenuated Bottleneck: F_v = min(F_u, C_{u,v}) · γ
                child_flow = min(current_flow, capacity) * self.decay

                if child_flow < self.prune_threshold:
                    continue

                # Only update if this path delivers a strictly higher flow
                if child_flow <= scores.get(neighbor, 0.0):
                    continue

                scores[neighbor]        = child_flow
                path_traces[neighbor]   = path + [neighbor]
                hop_distances[neighbor] = hop + 1

                heapq.heappush(
                    heap,
                    (-child_flow, hop + 1, neighbor, path + [neighbor]),
                )

        return scores, path_traces, hop_distances

    def _relation_factor(self, relation: str) -> float:
        """
        Relation-type damping applied to edge capacity C_{u,v}.

        supplies / depends_on : 1.00 — direct operational dependency
        ships_through         : 0.70 — logistics link (alternative routes exist)
        located_in            : 0.50 — geographic co-location (indirect risk)
        """
        if relation in {RelationType.SUPPLIES.value, RelationType.DEPENDS_ON.value}:
            return 1.0
        if relation == RelationType.SHIPS_THROUGH.value:
            return LOGISTICS_DECAY_FACTOR
        if relation == RelationType.LOCATED_IN.value:
            return LOCATION_DECAY_FACTOR
        return 0.8

    # ------------------------------------------------------------------
    # Result construction helpers
    # ------------------------------------------------------------------

    def _build_exposures(
        self,
        scores:        dict[str, float],
        path_traces:   dict[str, list[str]],
        hop_distances: dict[str, int],
    ) -> dict[str, NodeExposure]:
        exposures = {}
        for node, score in scores.items():
            etype = (
                self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
                if node in self.G else "Unknown"
            )
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

    def _build_tiers(self, scores: dict[str, float]) -> dict[str, list[str]]:
        tiers: dict[str, list[str]] = {sl.value: [] for sl in SeverityLevel}
        for node, score in scores.items():
            tiers[SeverityLevel.from_score(score).value].append(node)
        for tier in tiers:
            tiers[tier].sort(key=lambda n: -scores[n])
        return tiers

    def _compute_stats(
        self,
        scores:        dict[str, float],
        hop_distances: dict[str, int],
        event:         DisruptionEvent,
    ) -> dict:
        if not scores:
            return {"total_affected": 0}
        sv = list(scores.values())
        return {
            "total_affected":  len(scores),
            "mean_score":      round(sum(sv) / len(sv), 4),
            "max_score":       round(max(sv), 4),
            "min_score":       round(min(sv), 4),
            "max_hop_reached": max(hop_distances.values(), default=0),
            "critical_count":  sum(1 for s in sv if s >= 0.80),
            "high_count":      sum(1 for s in sv if 0.50 <= s < 0.80),
            "moderate_count":  sum(1 for s in sv if 0.25 <= s < 0.50),
            "low_count":       sum(1 for s in sv if s < 0.25),
            "event_category":  event.category.value,
            "initial_shock":   event.initial_shock,
        }
