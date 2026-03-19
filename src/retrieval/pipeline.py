"""
pipeline.py
-----------
GraphRAGPipeline — the top-level orchestrator for Module 2.

Combines SubgraphRetriever (local, query-specific subgraph) and
CommunitySummarizer (global, pre-computed community context) into a
single interface that any downstream module (3, 4, 5) can call with
one line:

  result = pipeline.query("Which suppliers are exposed to the Taiwan earthquake?")

The result carries both local subgraph context and relevant community
summaries, merged into a single prompt-ready text block.

Query routing
-------------
The pipeline auto-detects whether a query is "local" or "global":

  Local  — asks about a specific entity, event, or dependency chain.
           Examples: "Is TSMC exposed to the earthquake?"
                     "What does Apple depend on?"
  Global — asks about the whole supply chain or broad themes.
           Examples: "What is our overall East Asian risk exposure?"
                     "Which regions create the most concentration risk?"

Local queries use the SubgraphRetriever alone (fast, precise).
Global queries prepend community summaries to the subgraph context.
Both types are useful and the distinction is just about which context
to include in the LLM prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import networkx as nx

from src.embeddings.encoder   import NodeEncoder
from src.retrieval.retriever  import SubgraphRetriever, RetrievedSubgraph
from src.retrieval.community  import CommunitySummarizer, Community


# ---------------------------------------------------------------------------
# Query type detection keywords
# ---------------------------------------------------------------------------

GLOBAL_QUERY_SIGNALS = {
    "overall", "across", "all", "entire", "whole", "total", "every",
    "concentration", "broad", "general", "summary", "overview",
    "most exposed", "biggest risk", "main risks", "key vulnerabilities",
    "which regions", "which countries", "compare",
}


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGResult:
    """
    Full result from a GraphRAG query.

    Attributes
    ----------
    query            : str              — the original query
    query_type       : str              — "local" or "global"
    subgraph_result  : RetrievedSubgraph
    communities      : list[Community]  — relevant communities (global only)
    prompt_context   : str              — merged, prompt-ready text block
    """
    query:           str
    query_type:      str
    subgraph_result: RetrievedSubgraph
    communities:     list[Community]
    prompt_context:  str


# ---------------------------------------------------------------------------
# GraphRAGPipeline
# ---------------------------------------------------------------------------

class GraphRAGPipeline:
    """
    Top-level orchestrator for the GraphRAG retrieval layer.

    Usage
    -----
    >>> pipeline = GraphRAGPipeline(G, encoder)
    >>> pipeline.build_communities()          # run once, saves to disk
    >>> result = pipeline.query("What is Tesla's tier-2 supplier risk?")
    >>> print(result.prompt_context)          # inject into LLM prompt
    """

    def __init__(
        self,
        graph:          nx.DiGraph,
        encoder:        NodeEncoder,
        seed_k:         int  = 5,
        hop_radius:     int  = 2,
        max_nodes:      int  = 40,
        use_llm_summaries: bool = False,   # set True if GPU available
    ):
        self.G       = graph
        self.encoder = encoder

        self.retriever = SubgraphRetriever(
            graph=graph,
            encoder=encoder,
            seed_k=seed_k,
            hop_radius=hop_radius,
            max_nodes=max_nodes,
        )
        self.summarizer = CommunitySummarizer(
            graph=graph,
            use_llm=use_llm_summaries,
        )

        self._communities_built = False

    # ------------------------------------------------------------------
    # Setup — build community summaries (one-time, cacheable)
    # ------------------------------------------------------------------

    def build_communities(
        self, save_path: Optional[str] = None
    ) -> "GraphRAGPipeline":
        """
        Run Louvain + summarization on the full graph.
        Call this once before any queries. Results can be saved and reloaded.

        Parameters
        ----------
        save_path : str, optional
            If given, persist communities to this JSON path.

        Returns self for chaining.
        """
        self.summarizer.detect_and_summarize()
        self._communities_built = True

        if save_path:
            self.summarizer.save(save_path)

        return self

    def load_communities(self, path: str) -> "GraphRAGPipeline":
        """Load pre-computed communities from disk (skip recomputation)."""
        self.summarizer.load(path)
        self._communities_built = True
        return self

    # ------------------------------------------------------------------
    # Query — main entry point
    # ------------------------------------------------------------------

    def query(self, query: str) -> GraphRAGResult:
        """
        Run the full GraphRAG pipeline for a single query.

        Automatically detects query type (local vs global) and builds
        the appropriate context for the LLM.

        Parameters
        ----------
        query : str — natural language question

        Returns
        -------
        GraphRAGResult
        """
        query_type = self._classify_query(query)

        # Always retrieve the local subgraph
        subgraph_result = self.retriever.retrieve(query)

        # For global queries, also pull in community summaries
        communities: list[Community] = []
        community_text = ""

        if query_type == "global" and self._communities_built:
            communities  = self.summarizer.find_relevant_communities(query, top_k=3)
            community_text = self.summarizer.format_for_prompt(communities)

        # Merge into single prompt context block
        prompt_context = self._build_prompt_context(
            query, query_type, subgraph_result, community_text
        )

        return GraphRAGResult(
            query=query,
            query_type=query_type,
            subgraph_result=subgraph_result,
            communities=communities,
            prompt_context=prompt_context,
        )

    def query_batch(self, queries: list[str]) -> list[GraphRAGResult]:
        """Run query() for a list of queries. Returns one result per query."""
        return [self.query(q) for q in queries]

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def _build_prompt_context(
        self,
        query:           str,
        query_type:      str,
        subgraph_result: RetrievedSubgraph,
        community_text:  str,
    ) -> str:
        """
        Assemble the final prompt context block that Module 4 injects
        into the LLM prompt.

        Structure:
          [INSTRUCTIONS]
          [COMMUNITY SUMMARIES]   ← global queries only
          [LOCAL SUBGRAPH]        ← always included
          [QUERY]
        """
        sections: list[str] = []

        sections.append(
            "You are a supply chain risk analyst with access to a knowledge graph.\n"
            "Use ONLY the graph context below to answer. "
            "Cite specific entity names and relationship paths in your answer.\n"
            "If a fact is not in the context, say so explicitly.\n"
        )

        if query_type == "global" and community_text:
            sections.append(community_text)

        sections.append(subgraph_result.serialized)

        sections.append(f"--- Question ---\n{query}")

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Query type classifier
    # ------------------------------------------------------------------

    def _classify_query(self, query: str) -> str:
        """
        Classify a query as "local" or "global" based on keyword signals.

        Local  → asks about a specific entity or event
        Global → asks about the whole supply chain or broad themes
        """
        tokens = set(query.lower().split())
        if tokens & GLOBAL_QUERY_SIGNALS:
            return "global"
        return "local"

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def community_overview(self) -> str:
        """Return a human-readable overview of all detected communities."""
        if not self._communities_built:
            return "Communities not yet built. Call build_communities() first."

        lines = [f"Communities detected: {len(self.summarizer.communities)}\n"]
        for c in self.summarizer.communities:
            lines.append(
                f"  Community {c.community_id:2d} | "
                f"{c.dominant_type:15s} | "
                f"{c.size:3d} nodes | "
                f"{c.internal_edges:3d} internal edges"
            )
            lines.append(f"    → {c.summary[:90]}...")
        return "\n".join(lines)
