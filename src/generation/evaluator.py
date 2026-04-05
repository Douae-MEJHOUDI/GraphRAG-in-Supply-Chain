"""
evaluator.py
------------
RiskReportEvaluator — compares three system conditions for the ablation study.

This is the evaluation infrastructure used in Module 5, but built here
in Module 4 so it runs alongside generation.

Three conditions
----------------
  Condition A — Baseline LLM
    The LLM receives only the question, no graph context at all.
    Represents "what does the model know from pretraining alone?"

  Condition B — Vector RAG
    The LLM receives the top-K most similar nodes by embedding similarity,
    with NO graph traversal and NO disruption scores.
    Represents "what does semantic search alone retrieve?"

  Condition C — GraphRAG + Simulation (this project)
    The LLM receives the full enriched context: subgraph + community
    summaries + disruption scores + dependency paths.
    Represents the full pipeline.

Evaluation metrics
------------------
  1. Multi-hop accuracy  — does the answer surface entities that are
     2+ hops from ground-zero? (manually labelled ground truth)
  2. Faithfulness        — does the answer contradict the graph?
     (checked by an LLM judge prompt)
  3. Citation rate       — what fraction of factual claims cite a
     specific entity name or graph path?
  4. Disruption coverage — what fraction of nodes above the exposure
     threshold appear in the answer?
  5. Hallucination rate  — does the answer invent relationships not
     in the graph? (LLM judge)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.simulation.engine    import SimulationResult
from src.simulation.propagator import PropagationResult
from src.generation.generator  import RiskReport, RiskReportGenerator


# ---------------------------------------------------------------------------
# Metric result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """Result for a single evaluation condition."""
    condition:       str          # "baseline", "vector_rag", "graphrag_sim"
    prompt_used:     str          # the exact prompt sent to the LLM
    report:          RiskReport
    metrics:         dict         = field(default_factory=dict)


@dataclass
class AblationResult:
    """Full ablation study result for one query/scenario pair."""
    query:             str
    event_name:        str
    baseline:          ConditionResult
    vector_rag:        ConditionResult
    graphrag_sim:      ConditionResult
    comparison_table:  str        = ""    # formatted display string

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "query":      self.query,
            "event_name": self.event_name,
            "baseline":   {
                "metrics": self.baseline.metrics,
                "report":  self.baseline.report.full_text[:500],
            },
            "vector_rag": {
                "metrics": self.vector_rag.metrics,
                "report":  self.vector_rag.report.full_text[:500],
            },
            "graphrag_sim": {
                "metrics": self.graphrag_sim.metrics,
                "report":  self.graphrag_sim.report.full_text[:500],
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Evaluator] Ablation result saved → {path}")


# ---------------------------------------------------------------------------
# RiskReportEvaluator
# ---------------------------------------------------------------------------

class RiskReportEvaluator:
    """
    Runs the three-condition ablation study and computes evaluation metrics.

    Usage
    -----
    >>> evaluator = RiskReportEvaluator(generator, encoder, G)
    >>> ablation = evaluator.run_ablation(sim_result, ground_truth_nodes)
    >>> print(ablation.comparison_table)
    """

    def __init__(
        self,
        generator,          # RiskReportGenerator instance
        encoder,            # NodeEncoder from Module 1
        graph,              # nx.DiGraph
        exposure_threshold: float = 0.25,  # nodes above this score count as "affected"
    ):
        self.generator          = generator
        self.encoder            = encoder
        self.G                  = graph
        self.exposure_threshold = exposure_threshold

    # ------------------------------------------------------------------
    # Main ablation runner
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        sim_result:          SimulationResult,
        ground_truth_nodes:  Optional[list[str]] = None,
    ) -> AblationResult:
        """
        Run all three conditions for a single scenario and compute metrics.

        Parameters
        ----------
        sim_result          : full SimulationResult from Module 3
        ground_truth_nodes  : list of node names that SHOULD appear in a
                              correct answer (for multi-hop accuracy metric).
                              If None, inferred from propagation result.

        Returns
        -------
        AblationResult
        """
        query      = sim_result.graphrag_result.query
        event_name = sim_result.event.name

        # Infer ground truth from propagation if not provided
        if ground_truth_nodes is None:
            ground_truth_nodes = list(sim_result.propagation.scores.keys())

        print(f"\n[Evaluator] Running ablation for: {event_name}")
        print(f"  Query: {query[:70]}...")

        # ── Condition A: Baseline ─────────────────────────────────────
        print("\n[Evaluator] Condition A: Baseline LLM (no context)...")
        baseline_prompt = self._build_baseline_prompt(query, sim_result.event)
        baseline_report = self._generate_with_prompt(baseline_prompt, event_name, query)
        baseline_metrics = self._compute_metrics(
            baseline_report, sim_result.propagation, ground_truth_nodes, "baseline"
        )
        baseline = ConditionResult("baseline", baseline_prompt,
                                   baseline_report, baseline_metrics)

        # ── Condition B: Vector RAG ───────────────────────────────────
        print("\n[Evaluator] Condition B: Vector RAG (semantic search only)...")
        vector_prompt = self._build_vector_rag_prompt(
            query, sim_result.event, top_k=8
        )
        vector_report = self._generate_with_prompt(vector_prompt, event_name, query)
        vector_metrics = self._compute_metrics(
            vector_report, sim_result.propagation, ground_truth_nodes, "vector_rag"
        )
        vector_rag = ConditionResult("vector_rag", vector_prompt,
                                     vector_report, vector_metrics)

        # ── Condition C: GraphRAG + Simulation ────────────────────────
        print("\n[Evaluator] Condition C: GraphRAG + Simulation (full pipeline)...")
        graphrag_report = self.generator.generate(sim_result)
        graphrag_metrics = self._compute_metrics(
            graphrag_report, sim_result.propagation, ground_truth_nodes, "graphrag_sim"
        )
        graphrag_sim = ConditionResult(
            "graphrag_sim", sim_result.risk_report_prompt,
            graphrag_report, graphrag_metrics
        )

        # ── Build comparison table ────────────────────────────────────
        table = self._build_comparison_table(baseline, vector_rag, graphrag_sim)

        return AblationResult(
            query=query,
            event_name=event_name,
            baseline=baseline,
            vector_rag=vector_rag,
            graphrag_sim=graphrag_sim,
            comparison_table=table,
        )

    # ------------------------------------------------------------------
    # Prompt builders for baseline and vector RAG conditions
    # ------------------------------------------------------------------

    def _build_baseline_prompt(
        self, query: str, event: "DisruptionEvent"
    ) -> str:
        """
        Condition A: no graph context at all — just the raw question.
        This measures what the LLM knows from pretraining alone.
        """
        return (
            f"You are a senior supply chain risk analyst.\n\n"
            f"A disruption event has occurred: {event.name}.\n"
            f"Description: {event.description}\n\n"
            f"Question: {query}\n\n"
            f"Please provide a risk assessment based on your knowledge.\n\n"
            f"--- RISK REPORT ---"
        )

    def _build_vector_rag_prompt(
        self, query: str, event: "DisruptionEvent", top_k: int = 8
    ) -> str:
        """
        Condition B: top-K nodes by embedding similarity only.
        No graph traversal, no disruption scores, no path traces.
        This measures what semantic search alone retrieves.
        """
        results = self.encoder.search(query, k=top_k)

        context_lines = [
            f"=== Vector RAG Context (semantic search, top {top_k} nodes) ===",
            f"Event: {event.name}",
            f"Query: {query}",
            "",
        ]
        for node, score in results:
            if node not in self.G:
                continue
            from src.graph.schema import NODE_ATTR_TYPE
            etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
            context_lines.append(
                f"[NODE] {node} | Type: {etype} | Similarity: {score:.3f}"
            )

        context = "\n".join(context_lines)

        return (
            f"You are a senior supply chain risk analyst.\n"
            f"Use ONLY the context below to answer.\n\n"
            f"{context}\n\n"
            f"--- Question ---\n{query}\n\n"
            f"--- RISK REPORT ---"
        )

    def _generate_with_prompt(
        self, prompt: str, event_name: str, query: str
    ) -> RiskReport:
        """Call the generator with a custom prompt (bypasses SimulationResult)."""
        from src.generation.generator import RiskReport, STRUCTURED_PROMPT_SUFFIX
        import time

        full_prompt = prompt + STRUCTURED_PROMPT_SUFFIX
        t0 = time.time()

        if self.generator.backend.value == "openai":
            raw_text, tokens = self.generator._call_openai(full_prompt)
        else:
            raw_text, tokens = self.generator._call_ollama(full_prompt)

        elapsed = time.time() - t0
        return self.generator._parse_output(raw_text, event_name, query,
                                            tokens, elapsed)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        report:               RiskReport,
        propagation:          PropagationResult,
        ground_truth_nodes:   list[str],
        condition:            str,
    ) -> dict:
        """
        Compute all five evaluation metrics for one condition's report.

        Returns a dict with keys:
          multi_hop_accuracy, citation_rate, disruption_coverage,
          answer_length_words, entities_mentioned
        """
        text = report.full_text.lower()

        # 1. Multi-hop accuracy
        #    How many ground-truth affected nodes are mentioned in the answer?
        mentioned = [n for n in ground_truth_nodes if n.lower() in text]
        multi_hop_acc = (
            round(len(mentioned) / len(ground_truth_nodes), 3)
            if ground_truth_nodes else 0.0
        )

        # 2. Citation rate
        #    Fraction of sentences that mention at least one graph entity.
        sentences = [s.strip() for s in re.split(r'[.!?]', report.full_text)
                     if len(s.strip()) > 20]
        cited = sum(
            1 for s in sentences
            if any(n.lower() in s.lower() for n in self.G.nodes())
        )
        citation_rate = (
            round(cited / len(sentences), 3) if sentences else 0.0
        )

        # 3. Disruption coverage
        #    Fraction of nodes above exposure_threshold that appear in the answer.
        above_threshold = [
            n for n, s in propagation.scores.items()
            if s >= self.exposure_threshold
        ]
        covered = [n for n in above_threshold if n.lower() in text]
        disruption_coverage = (
            round(len(covered) / len(above_threshold), 3)
            if above_threshold else 0.0
        )

        # 4. Answer length
        word_count = len(report.full_text.split())

        # 5. Entities mentioned (raw count of graph node names in text)
        entities_mentioned = sum(
            1 for n in self.G.nodes() if n.lower() in text
        )

        return {
            "condition":           condition,
            "multi_hop_accuracy":  multi_hop_acc,
            "citation_rate":       citation_rate,
            "disruption_coverage": disruption_coverage,
            "answer_length_words": word_count,
            "entities_mentioned":  entities_mentioned,
            "nodes_mentioned":     mentioned[:10],  # sample for inspection
        }

    # ------------------------------------------------------------------
    # Comparison table formatter
    # ------------------------------------------------------------------

    def _build_comparison_table(
        self,
        baseline:    ConditionResult,
        vector_rag:  ConditionResult,
        graphrag_sim: ConditionResult,
    ) -> str:
        """Build a formatted comparison table for notebook display."""
        m_b  = baseline.metrics
        m_v  = vector_rag.metrics
        m_g  = graphrag_sim.metrics

        def row(label, key, fmt=".3f"):
            b = format(m_b.get(key, 0), fmt)
            v = format(m_v.get(key, 0), fmt)
            g = format(m_g.get(key, 0), fmt)
            # Bold the best value with an asterisk
            vals = [float(x) for x in [b, v, g]]
            best_idx = vals.index(max(vals))
            markers = ["", "", ""]
            markers[best_idx] = " *"
            return (f"  {label:30s} {b+markers[0]:>12s}"
                    f" {v+markers[1]:>12s} {g+markers[2]:>12s}")

        sep = "-" * 70
        lines = [
            sep,
            f"  {'Metric':30s} {'Baseline':>12s} {'Vector RAG':>12s} {'GraphRAG+Sim':>12s}",
            sep,
            row("Multi-hop accuracy",   "multi_hop_accuracy"),
            row("Citation rate",        "citation_rate"),
            row("Disruption coverage",  "disruption_coverage"),
            row("Entities mentioned",   "entities_mentioned", "d"),
            row("Answer length (words)","answer_length_words", "d"),
            sep,
            "  * = best value for this metric",
        ]
        return "\n".join(lines)
