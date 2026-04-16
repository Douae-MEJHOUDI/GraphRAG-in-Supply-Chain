"""
retriever.py
------------
SubgraphRetriever — the core of Module 2.

Given a natural language query, this module finds the most relevant portion
of the supply chain knowledge graph to use as LLM context.

It operates in two complementary stages that together give better coverage
than either alone:

  Stage A — Semantic seed search
    Encode the query with the same sentence-transformer used in Module 1.
    Ask the FAISS index for the top-K most similar nodes.
    These are the "entry points" into the graph.

  Stage B — K-hop graph traversal
    Starting from every seed node, expand outward K hops using NetworkX
    ego_graph. This pulls in all the nodes the seeds depend on or supply,
    capturing multi-hop relational context that pure vector search misses.

  Fusion
    Merge both sets, score every candidate node (semantic + structural
    centrality), and return a ranked, serialized subgraph ready for
    prompt injection.

Why both stages?
  - Vector search alone: finds nodes whose *names* sound relevant, but
    misses e.g. tier-3 suppliers that are relationally critical.
  - Graph traversal alone: has no notion of query relevance — it returns
    everything within K hops regardless of topical fit.
  - Together: semantically relevant entry points + full relational context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from src.embeddings.encoder import NodeEncoder
from src.graph.schema import (
    NODE_ATTR_TYPE,
    NODE_ATTR_METADATA,
    EDGE_ATTR_RELATION,
    EDGE_ATTR_WEIGHT,
    RelationType,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_SEED_K      = 5     # top-K nodes returned by FAISS per query
DEFAULT_HOP_RADIUS  = 2     # ego-graph radius (2 = node + neighbours + their neighbours)
DEFAULT_MAX_NODES   = 40    # cap on subgraph size to keep prompts manageable
MIN_SEMANTIC_SCORE  = 0.20  # discard FAISS results below this similarity

# Fusion score weights (semantic relevance vs structural centrality).
# Must sum to 1.0.
SEM_WEIGHT    = 0.6   # weight of FAISS semantic similarity in node score
STRUCT_WEIGHT = 0.4   # weight of degree centrality in node score

# Per-hop semantic score decay for nodes not directly returned by FAISS.
# A node 1 hop from a seed gets SEM_WEIGHT * HOP_DECAY^1, 2 hops → ^2, etc.
HOP_DECAY = 0.5

# Edge weight threshold above which a dependency is flagged as a single-source
# critical link (no meaningful alternative supplier path exists).
CRITICAL_EDGE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RetrievedSubgraph:
    """
    Everything the generation stage needs about a retrieved subgraph.

    Attributes
    ----------
    graph           : nx.DiGraph  — the subgraph itself
    seed_nodes      : list[str]   — FAISS top-K nodes that seeded retrieval
    node_scores     : dict        — {node_name: combined_relevance_score}
    serialized      : str         — text representation ready for LLM prompt
    query           : str         — the original query string
    """
    graph:         nx.DiGraph
    seed_nodes:    list[str]
    node_scores:   dict[str, float]
    serialized:    str
    query:         str
    stats:         dict           = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SubgraphRetriever
# ---------------------------------------------------------------------------

class SubgraphRetriever:
    """
    Dual-mode retriever: semantic FAISS search + K-hop graph traversal.

    Usage
    -----
    >>> retriever = SubgraphRetriever(G, encoder)
    >>> result = retriever.retrieve("Which suppliers are at risk from the Taiwan earthquake?")
    >>> print(result.serialized)   # ready to inject into an LLM prompt
    """

    def __init__(
        self,
        graph:          nx.DiGraph,
        encoder:        NodeEncoder,
        seed_k:         int   = DEFAULT_SEED_K,
        hop_radius:     int   = DEFAULT_HOP_RADIUS,
        max_nodes:      int   = DEFAULT_MAX_NODES,
        min_sem_score:  float = MIN_SEMANTIC_SCORE,
    ):
        self.G             = graph
        self.encoder       = encoder
        self.seed_k        = seed_k
        self.hop_radius    = hop_radius
        self.max_nodes     = max_nodes
        self.min_sem_score = min_sem_score

        # Pre-compute once — both are graph-invariant and used on every query.
        self._G_undirected:    nx.Graph       = graph.to_undirected()
        self._degree_centrality: dict[str, float] = nx.degree_centrality(graph)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievedSubgraph:
        """
        Main entry point. Run the full dual-retrieval pipeline.

        Parameters
        ----------
        query : str
            Natural language question or topic description.

        Returns
        -------
        RetrievedSubgraph
            Contains the subgraph, scores, serialized text, and metadata.
        """
        # Stage A — semantic seed nodes via FAISS
        seed_nodes, semantic_scores = self._semantic_seeds(query)

        # Stage B — expand each seed outward K hops
        candidate_nodes = self._graph_expansion(seed_nodes)

        # Fusion — score and rank all candidates
        node_scores = self._score_nodes(
            candidate_nodes, seed_nodes, semantic_scores
        )

        # Trim to max_nodes by score
        top_nodes = sorted(node_scores, key=node_scores.get, reverse=True)
        top_nodes = top_nodes[: self.max_nodes]

        # Extract subgraph induced on top_nodes
        subgraph = self.G.subgraph(top_nodes).copy()

        # Serialize for LLM prompt injection
        serialized = self._serialize(subgraph, node_scores, query)

        stats = {
            "seed_nodes":        len(seed_nodes),
            "candidates_before_trim": len(candidate_nodes),
            "subgraph_nodes":    subgraph.number_of_nodes(),
            "subgraph_edges":    subgraph.number_of_edges(),
        }

        return RetrievedSubgraph(
            graph=subgraph,
            seed_nodes=seed_nodes,
            node_scores=node_scores,
            serialized=serialized,
            query=query,
            stats=stats,
        )

    def retrieve_batch(self, queries: list[str]) -> list[RetrievedSubgraph]:
        """Retrieve subgraphs for multiple queries. Returns one result per query."""
        return [self.retrieve(q) for q in queries]

    # ------------------------------------------------------------------
    # Stage A — semantic seed search
    # ------------------------------------------------------------------

    def _semantic_seeds(
        self, query: str
    ) -> tuple[list[str], dict[str, float]]:
        """
        Query the FAISS index for the top-K most semantically similar nodes.

        Returns
        -------
        seed_nodes : list[str]
            Node names above the minimum similarity threshold.
        scores     : dict[str, float]
            {node_name: cosine_similarity_score}
        """
        raw_results = self.encoder.search(query, k=self.seed_k)

        seed_nodes = []
        scores: dict[str, float] = {}

        for node_name, sim_score in raw_results:
            if sim_score < self.min_sem_score:
                continue
            if node_name not in self.G:
                continue
            seed_nodes.append(node_name)
            scores[node_name] = sim_score

        return seed_nodes, scores

    # ------------------------------------------------------------------
    # Stage B — K-hop graph expansion
    # ------------------------------------------------------------------

    def _graph_expansion(self, seed_nodes: list[str]) -> set[str]:
        """
        Expand each seed node outward by hop_radius hops in the graph.

        Uses an undirected ego_graph so expansion follows edges in both
        directions (e.g. a query about a disruption event also pulls in
        entities affected_by it, even though the edge points the other way).

        Returns
        -------
        set[str] : all node names reachable within hop_radius from any seed.
        """
        if not seed_nodes:
            return set()

        all_candidates: set[str] = set()

        for seed in seed_nodes:
            if seed not in self._G_undirected:
                continue
            ego = nx.ego_graph(self._G_undirected, seed, radius=self.hop_radius)
            all_candidates.update(ego.nodes())

        return all_candidates

    # ------------------------------------------------------------------
    # Fusion — score all candidate nodes
    # ------------------------------------------------------------------

    def _score_nodes(
        self,
        candidates:      set[str],
        seed_nodes:      list[str],
        semantic_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Assign a combined relevance score to every candidate node.

        Score = 0.6 × semantic_score  +  0.4 × structural_score

        semantic_score:
          - 1.0  for seed nodes (direct FAISS match)
          - Decays by 0.5 per hop from nearest seed for expanded nodes

        structural_score:
          - Normalized degree centrality — more connected nodes are
            more important in SC context (hubs like major ports or
            tier-1 suppliers dominate the risk landscape).

        Returns
        -------
        dict[str, float] : {node_name: combined_score}
        """
        # Pre-compute hop distances from any seed (BFS on undirected graph)
        hop_distances = self._bfs_from_seeds(seed_nodes)

        # Use pre-computed degree centrality (graph-invariant, cached at init)
        degree_centrality = self._degree_centrality

        scores: dict[str, float] = {}

        for node in candidates:
            # Semantic component
            if node in semantic_scores:
                sem = semantic_scores[node]
            else:
                dist = hop_distances.get(node, 99)
                sem = HOP_DECAY ** dist

            # Structural component — degree centrality, normalized to [0,1]
            struct = degree_centrality.get(node, 0.0)

            combined = round(SEM_WEIGHT * sem + STRUCT_WEIGHT * struct, 4)
            scores[node] = combined

        return scores

    def _bfs_from_seeds(self, seed_nodes: list[str]) -> dict[str, int]:
        """
        BFS from all seed nodes simultaneously.
        Returns {node: minimum_hop_distance_from_any_seed}.
        """
        from collections import deque
        distances: dict[str, int] = {s: 0 for s in seed_nodes if s in self._G_undirected}
        queue: deque[str] = deque(distances.keys())

        while queue:
            current = queue.popleft()
            current_dist = distances[current]
            if current_dist >= self.hop_radius:
                continue
            for nbr in self._G_undirected.neighbors(current):
                if nbr not in distances:
                    distances[nbr] = current_dist + 1
                    queue.append(nbr)

        return distances

    # ------------------------------------------------------------------
    # Serialization — convert subgraph to LLM-readable text
    # ------------------------------------------------------------------

    def _serialize(
        self,
        subgraph:    nx.DiGraph,
        node_scores: dict[str, float],
        query:       str,
    ) -> str:
        """
        Convert a subgraph into a structured text block for LLM prompt injection.

        Format:
          === Supply Chain Subgraph Context ===
          Query: <query>

          [NODE] TSMC | Type: Supplier | Relevance: 0.91
            → supplies Apple (w=1.00)
            → depends_on ASML (w=0.50)
            ← depends_on Tesla

          ...

        This format is intentionally explicit about direction (→/←) and edge
        weights so the LLM can reason about which dependencies are critical.
        """
        lines: list[str] = []
        lines.append("=== Supply Chain Subgraph Context ===")
        lines.append(f"Query: {query}")
        lines.append(f"Nodes: {subgraph.number_of_nodes()}  |  "
                     f"Edges: {subgraph.number_of_edges()}")
        lines.append("")

        # Sort nodes by relevance score descending
        sorted_nodes = sorted(
            subgraph.nodes(),
            key=lambda n: node_scores.get(n, 0),
            reverse=True,
        )

        for node in sorted_nodes:
            attrs    = subgraph.nodes[node]
            etype    = attrs.get(NODE_ATTR_TYPE, "Unknown")
            score    = node_scores.get(node, 0.0)
            metadata = attrs.get(NODE_ATTR_METADATA, {})

            # Node header
            meta_str = ""
            for k in ("country", "capacity", "category"):
                if k in metadata:
                    meta_str += f" | {k}: {metadata[k]}"

            lines.append(
                f"[NODE] {node} | Type: {etype} "
                f"| Relevance: {score:.2f}{meta_str}"
            )

            # Outgoing edges (what this node does TO others)
            for _, target, edata in subgraph.out_edges(node, data=True):
                rel = edata.get(EDGE_ATTR_RELATION, "?")
                wt  = edata.get(EDGE_ATTR_WEIGHT, 1.0)
                crit = " [CRITICAL - no alternative]" if wt >= CRITICAL_EDGE_THRESHOLD else ""
                lines.append(f"  → {rel} {target} (weight={wt:.2f}){crit}")

            # Incoming edges (what others do TO this node)
            for source, _, edata in subgraph.in_edges(node, data=True):
                rel = edata.get(EDGE_ATTR_RELATION, "?")
                lines.append(f"  ← {rel} {source}")

            lines.append("")

        return "\n".join(lines)
