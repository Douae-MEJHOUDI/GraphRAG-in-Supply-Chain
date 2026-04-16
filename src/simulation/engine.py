"""
engine.py
---------
SimulationEngine — top-level orchestrator for Module 3.

Combines the DisruptionPropagator with the GraphRAGPipeline from Module 2
so that a single `.run()` call produces:

  1. Propagation result  — who is affected and by how much
  2. Enriched subgraph   — retrieved via GraphRAG, priority-biased toward
                           high-exposure nodes
  3. Enriched prompt context — the subgraph serialization with disruption
                           scores injected, ready for Module 4's LLM

The key integration point between Module 2 and Module 3:

  Module 2 (retriever) returns node_scores based on semantic + structural
  relevance. Module 3 (propagator) returns disruption_scores based on
  graph propagation.

  The engine MERGES these two score signals so the LLM context prioritizes
  nodes that are BOTH semantically relevant to the query AND highly exposed
  to the disruption. This is a stronger signal than either alone.

  merged_score = 0.5 × retrieval_score  +  0.5 × disruption_score
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from src.retrieval.pipeline    import GraphRAGPipeline, GraphRAGResult
from src.retrieval.retriever   import RetrievedSubgraph
from src.simulation.events     import DisruptionEvent, SeverityLevel
from src.simulation.propagator import DisruptionPropagator, PropagationResult, DEFAULT_DECAY, DEFAULT_MAX_HOPS
from src.graph.schema          import NODE_ATTR_TYPE, EDGE_ATTR_RELATION, EDGE_ATTR_WEIGHT


# ---------------------------------------------------------------------------
# Score-merging weights  (retrieval relevance vs disruption exposure)
# Must sum to 1.0.
# ---------------------------------------------------------------------------

RETRIEVAL_WEIGHT   = 0.5   # weight of GraphRAG retrieval score in merged signal
DISRUPTION_WEIGHT  = 0.5   # weight of disruption propagation score in merged signal

# Edge weight threshold above which a dependency is flagged as a single-source
# critical link (no meaningful alternative supplier path exists).
CRITICAL_EDGE_THRESHOLD = 0.95


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """
    Complete output of a disruption simulation + retrieval run.

    Attributes
    ----------
    event              : DisruptionEvent that was simulated
    propagation        : PropagationResult — full exposure map
    graphrag_result    : GraphRAGResult    — retrieved context from Module 2
    enriched_context   : str — prompt-ready text with disruption scores injected
    risk_report_prompt : str — the full prompt string to pass to the LLM
    """
    event:              DisruptionEvent
    propagation:        PropagationResult
    graphrag_result:    GraphRAGResult
    enriched_context:   str
    risk_report_prompt: str
    merged_scores:      dict[str, float] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        """Persist a JSON summary (scores + stats) for Module 5 evaluation."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event_name":       self.event.name,
            "event_category":   self.event.category.value,
            "initial_shock":    self.event.initial_shock,
            "stats":            self.propagation.stats,
            "top_10_exposed":   self.propagation.top_n(10),
            "critical_nodes":   self.propagation.critical_nodes(),
            "high_nodes":       self.propagation.high_nodes(),
            "tiers":            self.propagation.tiers,
            "merged_scores":    self.merged_scores,
        }
        path.write_text(
            json.dumps(payload, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"[Engine] Result saved → {path}")


# ---------------------------------------------------------------------------
# SimulationEngine
# ---------------------------------------------------------------------------

class SimulationEngine:
    """
    Runs the full Module 3 pipeline: propagation → retrieval → context enrichment.

    Usage
    -----
    >>> engine = SimulationEngine(G, pipeline)
    >>> result = engine.run(
    ...     event=SCENARIO_LIBRARY["taiwan_earthquake"],
    ...     query="Which of our suppliers are exposed to the Taiwan earthquake?"
    ... )
    >>> print(result.risk_report_prompt)   # pass directly to Module 4 LLM
    """

    def __init__(
        self,
        graph:    nx.DiGraph,
        pipeline: GraphRAGPipeline,
        decay:    float = DEFAULT_DECAY,
        max_hops: int   = DEFAULT_MAX_HOPS,
    ):
        self.G          = graph
        self.pipeline   = pipeline
        self.propagator = DisruptionPropagator(
            graph=graph,
            decay=decay,
            max_hops=max_hops,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        event: DisruptionEvent,
        query: Optional[str] = None,
    ) -> SimulationResult:
        """
        Full simulation pipeline for one disruption event.

        Parameters
        ----------
        event : DisruptionEvent
        query : str, optional
            Natural language question to answer about this disruption.
            If None, a default query is auto-generated from the event.

        Returns
        -------
        SimulationResult
        """
        if query is None:
            query = (
                f"What are the supply chain impacts of the {event.name}? "
                f"Which entities are most exposed and what are the critical "
                f"dependency paths?"
            )

        print(f"\n[Engine] === Simulation: {event.name} ===")

        # Step 0 — Semantically resolve any ground-zero node names that are
        # not exact matches in the graph (e.g. "ASML" → "ASML Holding NV").
        event = self._resolve_event_nodes(event)

        # Step 1 — Propagate disruption through the graph
        print("[Engine] Step 1: Propagating disruption...")
        propagation = self.propagator.propagate(event)

        # Step 2 — GraphRAG retrieval, biased toward high-exposure nodes
        print("[Engine] Step 2: GraphRAG retrieval...")
        graphrag_result = self.pipeline.query(query)

        # Step 3 — Merge retrieval scores with disruption scores
        print("[Engine] Step 3: Merging retrieval + disruption scores...")
        merged_scores = self._merge_scores(
            graphrag_result.subgraph_result.node_scores,
            propagation.scores,
        )

        # Step 4 — Re-serialize subgraph with disruption scores injected
        print("[Engine] Step 4: Building enriched prompt context...")
        enriched_context = self._build_enriched_context(
            graphrag_result, propagation, merged_scores, query
        )

        # Step 5 — Assemble the final LLM prompt
        risk_report_prompt = self._build_risk_report_prompt(
            event, enriched_context, query
        )

        return SimulationResult(
            event=event,
            propagation=propagation,
            graphrag_result=graphrag_result,
            enriched_context=enriched_context,
            risk_report_prompt=risk_report_prompt,
            merged_scores=merged_scores,
        )

    def run_batch(
        self,
        events:  list[DisruptionEvent],
        queries: Optional[list[str]] = None,
    ) -> list[SimulationResult]:
        """
        Run the full simulation for multiple events.
        queries must match events in order if provided.
        """
        queries = queries or [None] * len(events)
        return [
            self.run(event, query)
            for event, query in zip(events, queries)
        ]

    # ------------------------------------------------------------------
    # Ground-zero semantic resolution
    # ------------------------------------------------------------------

    def _resolve_event_nodes(self, event: DisruptionEvent) -> DisruptionEvent:
        """
        Return a copy of *event* with ground_zero names that are guaranteed
        to exist in the graph.

        For each name in event.ground_zero:
          - If it already exists in the graph → keep it as-is.
          - Otherwise → use the encoder's FAISS index to find the top-1
            semantically closest node and use that instead.

        This uses the same embedding model already loaded for retrieval,
        so there is no extra model cost.
        """
        encoder = self.pipeline.encoder
        resolved: list[str] = []

        for name in event.ground_zero:
            if name in self.G:
                resolved.append(name)
                continue

            # Semantic search: query is the entity name itself
            hits = encoder.search(name, k=1)
            if hits:
                best_node, score = hits[0]
                print(
                    f"[Engine] Ground-zero '{name}' not in graph — "
                    f"resolved to '{best_node}' (similarity={score:.4f})"
                )
                resolved.append(best_node)
            else:
                print(
                    f"[Engine] Warning: '{name}' not in graph and encoder "
                    f"returned no candidates — skipping"
                )

        if resolved == event.ground_zero:
            return event   # nothing changed, reuse original

        from dataclasses import replace
        return replace(event, ground_zero=resolved)

    # ------------------------------------------------------------------
    # Score merging
    # ------------------------------------------------------------------

    def _merge_scores(
        self,
        retrieval_scores:   dict[str, float],
        disruption_scores:  dict[str, float],
    ) -> dict[str, float]:
        """
        Merge retrieval relevance scores (Module 2) with disruption
        exposure scores (Module 3) into a single priority signal.

        merged = 0.5 × retrieval_score  +  0.5 × disruption_score

        Nodes only present in one set get 0.0 for the missing component.
        The merged score drives ordering in the enriched context — nodes
        that are both semantically relevant AND highly exposed appear first.
        """
        all_nodes = set(retrieval_scores) | set(disruption_scores)
        merged    = {}

        for node in all_nodes:
            r = retrieval_scores.get(node, 0.0)
            d = disruption_scores.get(node, 0.0)
            merged[node] = round(RETRIEVAL_WEIGHT * r + DISRUPTION_WEIGHT * d, 4)

        return merged

    # ------------------------------------------------------------------
    # Context enrichment
    # ------------------------------------------------------------------

    def _build_enriched_context(
        self,
        graphrag_result: GraphRAGResult,
        propagation:     PropagationResult,
        merged_scores:   dict[str, float],
        query:           str,
    ) -> str:
        """
        Build the enriched context block that replaces the plain subgraph
        serialization from Module 2.

        Enhancement over Module 2's serialization:
          - Each node now shows its DISRUPTION SCORE alongside relevance score
          - Severity tier badge per node (CRITICAL / HIGH / MODERATE / LOW)
          - Dependency path explanation ("exposed via: A → B → C")
          - Critical edges (no alternative supplier) are flagged inline
          - Ground-zero nodes are clearly marked
        """
        subgraph     = graphrag_result.subgraph_result.graph
        ground_zero  = set(propagation.event.ground_zero)

        lines = [
            "=== Enriched Supply Chain Context (GraphRAG + Disruption Simulation) ===",
            f"Event       : {propagation.event.name}",
            f"Category    : {propagation.event.category.value}",
            f"Initial shock: {propagation.event.initial_shock:.0%}",
            f"Query       : {query}",
            "",
            f"Directly disrupted: {', '.join(propagation.event.ground_zero)}",
            f"Total affected    : {len(propagation.scores)} nodes",
            "",
        ]

        # Sort nodes by merged score
        sorted_nodes = sorted(
            subgraph.nodes(),
            key=lambda n: merged_scores.get(n, 0),
            reverse=True,
        )

        for node in sorted_nodes:
            attrs        = subgraph.nodes[node]
            etype        = attrs.get(NODE_ATTR_TYPE, "Unknown")
            r_score      = graphrag_result.subgraph_result.node_scores.get(node, 0.0)
            d_score      = propagation.scores.get(node, 0.0)
            m_score      = merged_scores.get(node, 0.0)
            is_gz        = " [GROUND ZERO]" if node in ground_zero else ""

            # Severity badge
            if d_score > 0:
                severity = SeverityLevel.from_score(d_score).value
                sev_tag  = f"[{severity.upper()}]"
            else:
                sev_tag  = "[unaffected]"

            lines.append(
                f"{sev_tag} {node}{is_gz}"
                f" | Type: {etype}"
                f" | Disruption: {d_score:.3f}"
                f" | Relevance: {r_score:.3f}"
                f" | Priority: {m_score:.3f}"
            )

            # Dependency path explanation (explainability)
            if node in propagation.path_traces and not (node in ground_zero):
                path_str = propagation.path_explanation(node)
                lines.append(f"  Exposed via: {path_str}")

            # Mitigation — does this node have alternatives?
            exposure = propagation.exposures.get(node)
            if exposure and exposure.has_alternative:
                lines.append(
                    "  Mitigation: alternative suppliers available"
                )

            # Outgoing edges in subgraph
            for _, target, edata in subgraph.out_edges(node, data=True):
                rel = edata.get(EDGE_ATTR_RELATION, "?")
                wt  = edata.get(EDGE_ATTR_WEIGHT, 1.0)
                t_d = propagation.scores.get(target, 0.0)
                crit_flag = " *** CRITICAL - single source ***" if wt >= CRITICAL_EDGE_THRESHOLD else ""
                lines.append(
                    f"  → {rel} {target}"
                    f" (edge_weight={wt:.2f}, target_disruption={t_d:.3f})"
                    f"{crit_flag}"
                )

            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Final LLM prompt assembly
    # ------------------------------------------------------------------

    def _build_risk_report_prompt(
        self,
        event:            DisruptionEvent,
        enriched_context: str,
        query:            str,
    ) -> str:
        """
        Assemble the final prompt string passed to the LLM in Module 4.

        The prompt instructs the LLM to:
          1. Identify the most critically exposed entities (with scores)
          2. Explain the dependency chains that create the exposure
          3. Flag single-source dependencies (no alternatives)
          4. Suggest concrete mitigation actions
          5. Cite specific graph paths in every claim
        """
        return f"""You are a senior supply chain risk analyst.
A disruption event has been simulated through the supply chain knowledge graph.
Your task is to produce a structured risk report based ONLY on the graph context below.

INSTRUCTIONS:
1. Identify the top 5 most critically exposed entities, citing their disruption scores.
2. For each exposed entity, explain the dependency chain that creates the exposure
   (use the "Exposed via" paths provided in the context).
3. Flag any CRITICAL edges (single-source dependencies with no alternatives).
4. Suggest 2–3 concrete mitigation actions based on the graph structure
   (e.g. qualify alternative suppliers, shift logistics routes).
5. Assess the overall supply chain resilience for this event.
6. Every factual claim must cite a specific entity name or graph path from the context.
   Do NOT invent relationships not present in the context.

--- GRAPH CONTEXT ---

{enriched_context}

--- QUESTION ---

{query}

--- RISK REPORT ---"""
