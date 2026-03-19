"""
community.py
------------
CommunitySummarizer — pre-generates thematic summaries of graph communities.

Inspired directly by Microsoft's GraphRAG paper (Edge et al., 2024):
the key insight is that global questions ("what is our overall exposure to
East Asian risk?") cannot be answered by retrieving a local subgraph — they
require reasoning over the entire graph. The solution is to pre-cluster the
graph into communities and summarize each one, so that global queries can be
answered by aggregating community summaries rather than traversing the whole
graph at query time.

Pipeline
--------
1. Detect communities using the Louvain algorithm (python-louvain)
   — groups nodes that are more densely connected to each other than to
   the rest of the graph. In an SC graph, these naturally correspond to
   supplier clusters, regional clusters, and disruption clusters.

2. Summarize each community with a T5/BART encoder-decoder model
   — this is the encoder-decoder concept from the GenAI course. The model
   reads a structured description of the community (nodes + edges) and
   generates a concise natural language summary.

3. Store summaries for lookup at query time.

The summaries act as pre-computed "views" of the graph at different
granularities, enabling both local retrieval (SubgraphRetriever) and
global aggregation (this module) to work together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.schema import (
    NODE_ATTR_TYPE,
    EDGE_ATTR_RELATION,
    EDGE_ATTR_WEIGHT,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SUMMARIZER_MODEL = "facebook/bart-large-cnn"   # enc-dec, good summaries
FALLBACK_SUMMARIZER      = "sshleifer/distilbart-cnn-12-6"  # lighter alternative
MAX_COMMUNITY_TEXT_TOKENS = 512    # truncate community description before encoding
MIN_COMMUNITY_SIZE        = 2      # skip trivial single-node communities


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class Community:
    """
    A single detected community in the supply chain graph.

    Attributes
    ----------
    community_id  : int       — integer label assigned by Louvain
    nodes         : list[str] — node names belonging to this community
    summary       : str       — auto-generated natural language description
    dominant_type : str       — most common EntityType in this community
    size          : int       — number of nodes
    """
    community_id:  int
    nodes:         list[str]
    summary:       str        = ""
    dominant_type: str        = "Unknown"
    size:          int        = 0
    internal_edges: int       = 0

    def __post_init__(self):
        self.size = len(self.nodes)


# ---------------------------------------------------------------------------
# CommunitySummarizer
# ---------------------------------------------------------------------------

class CommunitySummarizer:
    """
    Detects graph communities (Louvain) and generates T5/BART summaries.

    This class has two modes:
      - With a HuggingFace summarizer loaded: generates real model summaries
      - Without (use_llm=False): generates structured rule-based summaries
        that are still useful for the LLM prompt without requiring a GPU.

    Usage
    -----
    >>> cs = CommunitySummarizer(G)
    >>> communities = cs.detect_and_summarize()
    >>> relevant = cs.find_relevant_communities("East Asian semiconductor risk")
    >>> print(cs.format_for_prompt(relevant))
    """

    def __init__(
        self,
        graph:         nx.DiGraph,
        model_name:    str  = DEFAULT_SUMMARIZER_MODEL,
        use_llm:       bool = True,
        min_size:      int  = MIN_COMMUNITY_SIZE,
    ):
        self.G         = graph
        self.model_name = model_name
        self.use_llm   = use_llm
        self.min_size  = min_size

        self.communities:   list[Community] = []
        self._summarizer                    = None   # lazy-loaded

        if use_llm:
            self._load_summarizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_summarize(self) -> list[Community]:
        """
        Full pipeline: detect communities, then summarize each one.

        Returns
        -------
        list[Community]
            All communities above min_size, with summaries populated.
        """
        print("[Community] Detecting communities (Louvain)...")
        raw_partition = self._detect_communities()

        print(f"[Community] Found {len(raw_partition)} raw communities")

        communities = []
        for cid, nodes in raw_partition.items():
            if len(nodes) < self.min_size:
                continue
            community = self._build_community(cid, nodes)
            community.summary = self._summarize_community(community)
            communities.append(community)

        communities.sort(key=lambda c: c.size, reverse=True)
        self.communities = communities

        print(f"[Community] {len(communities)} communities summarized "
              f"(≥ {self.min_size} nodes each)")
        return communities

    def find_relevant_communities(
        self,
        query:   str,
        top_k:   int = 3,
    ) -> list[Community]:
        """
        Return the top-K communities most relevant to a query string.

        Uses simple keyword overlap between query tokens and community
        summaries + node names. For a production system, replace this
        with embedding similarity over summary vectors.

        Parameters
        ----------
        query : str   — natural language query
        top_k : int   — number of communities to return

        Returns
        -------
        list[Community] — sorted by relevance descending
        """
        if not self.communities:
            return []

        query_tokens = set(query.lower().split())

        def relevance_score(c: Community) -> float:
            # Score = fraction of query tokens that appear in summary or node names
            text = (c.summary + " " + " ".join(c.nodes)).lower()
            hits = sum(1 for t in query_tokens if t in text)
            return hits / max(len(query_tokens), 1)

        ranked = sorted(self.communities, key=relevance_score, reverse=True)
        return ranked[:top_k]

    def format_for_prompt(self, communities: list[Community]) -> str:
        """
        Serialize a list of communities into a prompt-ready text block.

        Format:
          === Community Summaries ===

          [Community 1 — Supplier (8 nodes)]
          <summary text>
          Key entities: TSMC, ASML, Apple, NVIDIA...

          ...
        """
        if not communities:
            return "=== No relevant community summaries found ==="

        lines = ["=== Community Summaries (global context) ===", ""]
        for c in communities:
            lines.append(
                f"[Community {c.community_id} — "
                f"{c.dominant_type} cluster ({c.size} nodes)]"
            )
            lines.append(c.summary)
            key_nodes = ", ".join(c.nodes[:8])
            if len(c.nodes) > 8:
                key_nodes += f" ... (+{len(c.nodes)-8} more)"
            lines.append(f"Key entities: {key_nodes}")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Persist communities as JSON for reuse across sessions."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "community_id":   c.community_id,
                "nodes":          c.nodes,
                "summary":        c.summary,
                "dominant_type":  c.dominant_type,
                "size":           c.size,
                "internal_edges": c.internal_edges,
            }
            for c in self.communities
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[Community] Communities saved → {path}")

    def load(self, path: str | Path) -> None:
        """Load previously saved communities from JSON."""
        path = Path(path)
        data = json.loads(path.read_text())
        self.communities = [
            Community(
                community_id=d["community_id"],
                nodes=d["nodes"],
                summary=d["summary"],
                dominant_type=d["dominant_type"],
                size=d["size"],
                internal_edges=d.get("internal_edges", 0),
            )
            for d in data
        ]
        print(f"[Community] Loaded {len(self.communities)} communities from {path}")

    # ------------------------------------------------------------------
    # Community detection
    # ------------------------------------------------------------------

    def _detect_communities(self) -> dict[int, list[str]]:
        """
        Run Louvain community detection on the undirected version of the graph.

        Returns
        -------
        dict[int, list[str]] : {community_id: [node_name, ...]}
        """
        try:
            import community as community_louvain
        except ImportError:
            raise ImportError(
                "python-louvain not installed. "
                "Run: pip install python-louvain"
            )

        G_und   = self.G.to_undirected()
        partition = community_louvain.best_partition(G_und)
        # partition is {node: community_id} — invert to {community_id: [nodes]}
        groups: dict[int, list[str]] = {}
        for node, cid in partition.items():
            groups.setdefault(cid, []).append(node)
        return groups

    # ------------------------------------------------------------------
    # Community building
    # ------------------------------------------------------------------

    def _build_community(self, cid: int, nodes: list[str]) -> Community:
        """
        Build a Community object from a list of node names.
        Computes dominant entity type and internal edge count.
        """
        # Dominant entity type — most common type in this community
        type_counts: dict[str, int] = {}
        for node in nodes:
            if node in self.G:
                etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
                type_counts[etype] = type_counts.get(etype, 0) + 1

        dominant_type = (
            max(type_counts, key=type_counts.get)
            if type_counts else "Unknown"
        )

        # Count edges that are internal to this community
        node_set = set(nodes)
        internal = sum(
            1 for u, v in self.G.edges()
            if u in node_set and v in node_set
        )

        return Community(
            community_id=cid,
            nodes=nodes,
            dominant_type=dominant_type,
            internal_edges=internal,
        )

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------

    def _summarize_community(self, community: Community) -> str:
        """
        Generate a natural language summary of a community.

        With LLM: uses T5/BART encoder-decoder model.
        Without:  uses rule-based structured summary (still useful).
        """
        community_text = self._community_to_text(community)

        if self.use_llm and self._summarizer is not None:
            return self._llm_summary(community_text)
        else:
            return self._rule_based_summary(community)

    def _community_to_text(self, community: Community) -> str:
        """
        Build the input text that the summarizer will compress.

        Format:
          Supply chain community: Supplier cluster with 8 entities.
          Entities: TSMC (Supplier), ASML (Supplier), Apple (Manufacturer) ...
          Relationships: TSMC supplies Apple. TSMC depends_on ASML. ...
        """
        node_set = set(community.nodes)
        lines    = []

        # Header
        lines.append(
            f"Supply chain community: {community.dominant_type} cluster "
            f"with {community.size} entities."
        )

        # Entity list
        entity_descs = []
        for node in community.nodes[:15]:   # cap at 15 to avoid token overflow
            if node in self.G:
                etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "?")
                entity_descs.append(f"{node} ({etype})")
        lines.append("Entities: " + ", ".join(entity_descs) + ".")

        # Internal relationships
        rel_lines = []
        for u, v, data in self.G.edges(data=True):
            if u in node_set and v in node_set:
                rel = data.get(EDGE_ATTR_RELATION, "related_to")
                rel_lines.append(f"{u} {rel} {v}.")

        if rel_lines:
            lines.append("Relationships: " + " ".join(rel_lines[:20]))

        return " ".join(lines)

    def _llm_summary(self, text: str) -> str:
        """Run the T5/BART summarizer on a community text description."""
        try:
            result = self._summarizer(
                text,
                max_length=80,
                min_length=20,
                do_sample=False,
                truncation=True,
            )
            return result[0]["summary_text"]
        except Exception as e:
            print(f"[Community] Summarizer failed ({e}), falling back to rule-based.")
            return text[:200] + "..."

    def _rule_based_summary(self, community: Community) -> str:
        """
        Generate a structured summary without a model.
        Used when use_llm=False or when the model is unavailable.
        """
        node_set = set(community.nodes)

        # Find the most connected node (hub) in the community
        degree_in_community = {
            n: sum(1 for _, v in self.G.out_edges(n) if v in node_set)
               + sum(1 for u, _ in self.G.in_edges(n) if u in node_set)
            for n in community.nodes
            if n in self.G
        }
        hub = max(degree_in_community, key=degree_in_community.get, default="?")

        # Key relation types in this community
        rels_present: set[str] = set()
        for u, v, data in self.G.edges(data=True):
            if u in node_set and v in node_set:
                rels_present.add(data.get(EDGE_ATTR_RELATION, ""))

        rel_str = ", ".join(sorted(rels_present)) if rels_present else "various"

        return (
            f"A {community.dominant_type.lower()} cluster of "
            f"{community.size} supply chain entities. "
            f"Central hub: {hub}. "
            f"Primary relationship types: {rel_str}. "
            f"Contains {community.internal_edges} internal connections."
        )

    # ------------------------------------------------------------------
    # Model loading (lazy)
    # ------------------------------------------------------------------

    def _load_summarizer(self) -> None:
        """Lazy-load the HuggingFace summarization pipeline."""
        try:
            from transformers import pipeline
            print(f"[Community] Loading summarizer: {self.model_name}")
            self._summarizer = pipeline(
                "summarization",
                model=self.model_name,
                tokenizer=self.model_name,
            )
            print("[Community] Summarizer loaded.")
        except Exception as e:
            print(
                f"[Community] Could not load {self.model_name} ({e}). "
                f"Falling back to rule-based summaries."
            )
            self._summarizer = None
            self.use_llm     = False
