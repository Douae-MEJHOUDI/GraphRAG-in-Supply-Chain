"""
evaluator.py
------------
Three-metric evaluation framework .

Metrics
----------------
  1. Graph Construction 
       Manually labeled relationships checked against the extracted graph.
       Computed as: |correctly extracted triples| / |gold triples|  (recall).
       Reference: edge_case_gold_triples.jsonl vs extracted graph.

  2. Retrieval Quality 
       Expected entities checked in the returned subgraph.
       Computed as: |expected entities found in subgraph| / |expected entities|

  3. Report Coverage 
       Expected entities checked in Claude's output.
       Computed as: |expected entities found in report text| / |expected entities|

Ablation study (Final Report Coverage)
---------------------------------------------------------
  Condition A — Baseline LLM  (no graph context)
  Condition B — Vector RAG    (top-K semantic only)
  Condition C — GraphRAG+Sim  (full pipeline)

Usage
-----
    evaluator = PipelineEvaluator(G, encoder, pipeline, generator)

    gc  = evaluator.graph_construction_score(gold_triples_path)
    rq  = evaluator.retrieval_quality_score(queries_with_expected)
    rc  = evaluator.report_coverage_score(queries_with_expected, events)

    # Ablation
    ablation = evaluator.run_ablation(event, query, expected_entities)
    print(ablation.table())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.schema import NODE_ATTR_TYPE, EDGE_ATTR_RELATION
from src.simulation.events import DisruptionEvent


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphConstructionResult:
    """
    Metric 1: Graph Construction.
    Manually labeled relationships checked against extracted graph.
    """
    gold_total:    int
    pred_total:    int
    true_positive: int
    precision:     float
    recall:        float
    f1:            float
    per_relation:  dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"Graph Construction\n"
            f"  Gold triples     : {self.gold_total}\n"
            f"  Predicted        : {self.pred_total}\n"
            f"  True positives   : {self.true_positive}\n"
            f"  Precision        : {self.precision:.2%}\n"
            f"  Recall           : {self.recall:.2%}\n"
            f"  F1               : {self.f1:.2%}\n"
        )


@dataclass
class RetrievalQualityResult:
    """
    Metric 2: Retrieval Quality.
    Expected entities checked in returned subgraph.
    """
    queries_evaluated:  int
    mean_coverage:      float   
    per_query:          list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Retrieval Quality\n"
            f"  Queries evaluated: {self.queries_evaluated}\n"
            f"  Mean coverage    : {self.mean_coverage:.2%}\n"
        )


@dataclass
class ReportCoverageResult:
    """
    Metric 3: Report Coverage.
    Expected entities checked in Claude's output.
    """
    queries_evaluated: int
    mean_coverage:     float   
    per_query:         list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Report Coverage\n"
            f"  Queries evaluated: {self.queries_evaluated}\n"
            f"  Mean coverage    : {self.mean_coverage:.2%}\n"
        )


@dataclass
class AblationCondition:
    """Result for one ablation condition."""
    name:          str     # "Baseline LLM", "Vector RAG", "GraphRAG+Sim"
    report_text:   str
    coverage:      float   # report coverage against expected entities
    entities_hit:  list[str]
    entities_missed: list[str]


@dataclass
class AblationResult:
    """
    Final Report Coverage ablation.
    Baseline LLM, Vector RAG, Our Approach.
    """
    query:       str
    event_name:  str
    baseline:    AblationCondition
    vector_rag:  AblationCondition
    graphrag_sim: AblationCondition

    def table(self) -> str:
        sep = "─" * 60
        rows = [
            sep,
            f"  {'Condition':<22} {'Coverage':>10}   Entities found",
            sep,
            f"  {'Baseline LLM':<22} {self.baseline.coverage:>9.1%}"
            f"   {len(self.baseline.entities_hit)}/{len(self.baseline.entities_hit)+len(self.baseline.entities_missed)}",
            f"  {'Vector RAG':<22} {self.vector_rag.coverage:>9.1%}"
            f"   {len(self.vector_rag.entities_hit)}/{len(self.vector_rag.entities_hit)+len(self.vector_rag.entities_missed)}",
            f"  {'GraphRAG + Simulation':<22} {self.graphrag_sim.coverage:>9.1%}"
            f"   {len(self.graphrag_sim.entities_hit)}/{len(self.graphrag_sim.entities_hit)+len(self.graphrag_sim.entities_missed)}",
            sep,
            "  Final Report Coverage",
        ]
        return "\n".join(rows)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "query":       self.query,
                "event_name":  self.event_name,
                "baseline":    {"coverage": self.baseline.coverage,
                                "hits":     self.baseline.entities_hit},
                "vector_rag":  {"coverage": self.vector_rag.coverage,
                                "hits":     self.vector_rag.entities_hit},
                "graphrag_sim":{"coverage": self.graphrag_sim.coverage,
                                "hits":     self.graphrag_sim.entities_hit},
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# PipelineEvaluator
# ---------------------------------------------------------------------------

class PipelineEvaluator:
    """
    Computes the three metrics and runs the ablation study.

    Parameters
    ----------
    graph     : the supply chain knowledge graph (nx.DiGraph)
    encoder   : NodeEncoder with FAISS index loaded
    pipeline  : GraphRAGPipeline for retrieval
    generator : RiskReportGenerator for report generation (optional)
    """

    def __init__(self, graph: nx.DiGraph, encoder, pipeline, generator=None):
        self.G         = graph
        self.encoder   = encoder
        self.pipeline  = pipeline
        self.generator = generator

    # ------------------------------------------------------------------
    # Metric 1: Graph Construction
    # ------------------------------------------------------------------

    def graph_construction_score(
        self,
        gold_path: str | Path,
        format:    str = "jsonl",   # "jsonl" or "json"
    ) -> GraphConstructionResult:
        """
        Graph Construction.

        Loads gold triples from `gold_path` and checks each triple against
        the extracted graph.  A triple (head, relation, tail) is a true
        positive if the graph contains an edge head→tail with matching relation.

        Parameters
        ----------
        gold_path : path to manually-labeled ground-truth triples
        format    : "jsonl" (one JSON per line) or "json" (array)
        """
        gold_path = Path(gold_path)
        raw = []
        if format == "jsonl":
            for line in gold_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    raw.append(json.loads(line))
        else:
            raw = json.loads(gold_path.read_text(encoding="utf-8"))

        gold_triples = [
            (r["head"], r["relation"], r["tail"])
            for r in raw
            if all(k in r for k in ("head", "relation", "tail"))
        ]

        true_positives = 0
        per_relation: dict[str, dict] = {}

        for head, relation, tail in gold_triples:
            rel_stats = per_relation.setdefault(
                relation, {"gold": 0, "tp": 0}
            )
            rel_stats["gold"] += 1

            # Check if graph contains a matching edge
            if (
                self.G.has_node(head)
                and self.G.has_node(tail)
                and self.G.has_edge(head, tail)
            ):
                edge_data  = self.G.get_edge_data(head, tail)
                edge_rel   = edge_data.get(EDGE_ATTR_RELATION, "")
                if edge_rel == relation:
                    true_positives += 1
                    rel_stats["tp"] += 1

        gold_total = len(gold_triples)
        pred_total = self.G.number_of_edges()
        precision  = true_positives / pred_total if pred_total else 0.0
        recall     = true_positives / gold_total  if gold_total  else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) else 0.0)

        # Per-relation recall
        for rel, stats in per_relation.items():
            stats["recall"] = round(
                stats["tp"] / stats["gold"] if stats["gold"] else 0.0, 3
            )

        result = GraphConstructionResult(
            gold_total=gold_total,
            pred_total=pred_total,
            true_positive=true_positives,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            per_relation=per_relation,
        )
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Metric 2: Retrieval Quality
    # ------------------------------------------------------------------

    def retrieval_quality_score(
        self,
        eval_suite: list[dict],
    ) -> RetrievalQualityResult:
        """
        Retrieval Quality.

        For each query in eval_suite, retrieve the subgraph and measure what
        fraction of `expected_entities` appear in the returned nodes.

        Parameters
        ----------
        eval_suite : list of dicts, each with keys:
            "query"             : str
            "expected_entities" : list[str]  — nodes that MUST appear in subgraph
        """
        per_query   = []
        coverages   = []

        for item in eval_suite:
            query    = item["query"]
            expected = item.get("expected_entities", [])
            if not expected:
                continue

            result       = self.pipeline.query(query)
            subgraph_nodes = set(result.subgraph_result.graph.nodes())

            hits   = [e for e in expected if e in subgraph_nodes]
            cov    = len(hits) / len(expected)
            coverages.append(cov)

            per_query.append({
                "query":    query,
                "coverage": round(cov, 4),
                "hits":     hits,
                "missed":   [e for e in expected if e not in subgraph_nodes],
            })

        mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
        result = RetrievalQualityResult(
            queries_evaluated=len(coverages),
            mean_coverage=round(mean_cov, 4),
            per_query=per_query,
        )
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Metric 3: Report Coverage
    # ------------------------------------------------------------------

    def report_coverage_score(
        self,
        eval_suite:    list[dict],
        events_by_name: dict,
    ) -> ReportCoverageResult:
        """
        Report Coverage.

        For each query in eval_suite, run the full pipeline and measure what
        fraction of `expected_entities` appear in Claude's output.

        Parameters
        ----------
        eval_suite : list of dicts, each with keys:
            "query"             : str
            "event_name"        : str
            "expected_entities" : list[str]
        events_by_name : dict[str, DisruptionEvent]
        """
        if self.generator is None:
            raise RuntimeError(
                "Report coverage requires a generator. "
                "Pass generator= to PipelineEvaluator()."
            )

        per_query = []
        coverages = []

        for item in eval_suite:
            query    = item["query"]
            expected = item.get("expected_entities", [])
            event    = events_by_name.get(item.get("event_name", ""))
            if not expected or event is None:
                continue

            from src.simulation.engine import SimulationEngine
            from src.simulation.propagator import DisruptionPropagator

            # Build a minimal SimulationResult for the generator
            propagator = DisruptionPropagator(self.G)
            prop       = propagator.propagate(event)
            rag_result = self.pipeline.query(query)

            from src.simulation.engine import SimulationResult
            from src.simulation.engine import SimulationEngine as _SE
            tmp_engine = _SE(self.G, self.pipeline)
            sim_result = tmp_engine.run(event=event, query=query)

            report     = self.generator.generate(sim_result)
            report_text = report.full_text.lower()

            hits   = [e for e in expected if e.lower() in report_text]
            cov    = len(hits) / len(expected)
            coverages.append(cov)

            per_query.append({
                "query":    query,
                "coverage": round(cov, 4),
                "hits":     hits,
                "missed":   [e for e in expected if e.lower() not in report_text],
            })

        mean_cov = sum(coverages) / len(coverages) if coverages else 0.0
        result = ReportCoverageResult(
            queries_evaluated=len(coverages),
            mean_coverage=round(mean_cov, 4),
            per_query=per_query,
        )
        print(result.summary())
        return result

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def run_ablation(
        self,
        event:             "DisruptionEvent",
        query:             str,
        expected_entities: list[str],
    ) -> AblationResult:
        """
        Ablation: compare three conditions on Final Report Coverage.

        Condition A — Baseline LLM  (no graph context)
        Condition B — Vector RAG    (top-K semantic nodes, no traversal)
        Condition C — GraphRAG+Sim  (full pipeline — our approach)

        Parameters
        ----------
        event             : DisruptionEvent to simulate
        query             : natural language question
        expected_entities : ground-truth list of entities that should appear
                            in a correct answer
        """
        if self.generator is None:
            raise RuntimeError("Ablation requires a generator.")

        print(f"\n[Evaluator] Ablation: {event.name}")
        print(f"  Query: {query[:70]}...")
        print(f"  Expected entities: {expected_entities}")

        # ── Condition A: Baseline LLM ─────────────────────────────────
        print("\n[Evaluator] A) Baseline LLM...")
        baseline_text = self._call_llm_baseline(event, query)
        baseline = self._score_condition(
            "Baseline LLM", baseline_text, expected_entities
        )

        # ── Condition B: Vector RAG ───────────────────────────────────
        print("\n[Evaluator] B) Vector RAG...")
        vector_text = self._call_llm_vector_rag(event, query, top_k=8)
        vector_rag  = self._score_condition(
            "Vector RAG", vector_text, expected_entities
        )

        # ── Condition C: GraphRAG + Simulation ────────────────────────
        print("\n[Evaluator] C) GraphRAG + Simulation (our approach)...")
        from src.simulation.engine import SimulationEngine
        engine     = SimulationEngine(self.G, self.pipeline)
        sim_result = engine.run(event=event, query=query)
        report     = self.generator.generate(sim_result)
        graphrag_sim = self._score_condition(
            "GraphRAG+Sim", report.full_text, expected_entities
        )

        result = AblationResult(
            query=query,
            event_name=event.name,
            baseline=baseline,
            vector_rag=vector_rag,
            graphrag_sim=graphrag_sim,
        )
        print("\n" + result.table())
        return result

    # ------------------------------------------------------------------
    # Ablation helpers
    # ------------------------------------------------------------------

    def _score_condition(
        self,
        name:     str,
        text:     str,
        expected: list[str],
    ) -> AblationCondition:
        """Check which expected entities appear in the report text."""
        lower = text.lower()
        hits   = [e for e in expected if e.lower() in lower]
        missed = [e for e in expected if e.lower() not in lower]
        cov    = len(hits) / len(expected) if expected else 0.0
        return AblationCondition(
            name=name,
            report_text=text,
            coverage=round(cov, 4),
            entities_hit=hits,
            entities_missed=missed,
        )

    def _call_llm_baseline(
        self, event: "DisruptionEvent", query: str
    ) -> str:
        """Generate a report with no graph context (Condition A)."""
        prompt = (
            f"You are a senior supply chain risk analyst.\n\n"
            f"A disruption event has occurred: {event.name}.\n"
            f"Description: {event.description}\n\n"
            f"Question: {query}\n\n"
            f"Provide a detailed risk assessment based on your knowledge.\n\n"
            f"--- RISK REPORT ---"
        )
        return self.generator._raw_generate(prompt)

    def _call_llm_vector_rag(
        self, event: "DisruptionEvent", query: str, top_k: int
    ) -> str:
        """Generate a report with top-K semantic nodes only (Condition B)."""
        results = self.encoder.search(query, k=top_k)
        lines   = [
            f"=== Vector RAG Context (top {top_k} semantically similar nodes) ===",
            f"Event: {event.name}",
            "",
        ]
        for node, score in results:
            if node not in self.G:
                continue
            etype = self.G.nodes[node].get(NODE_ATTR_TYPE, "Unknown")
            lines.append(f"  [{etype}] {node}  (similarity={score:.3f})")

        context = "\n".join(lines)
        prompt = (
            f"You are a senior supply chain risk analyst.\n"
            f"Use ONLY the context below to answer.\n\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            f"--- RISK REPORT ---"
        )
        return self.generator._raw_generate(prompt)


# ---------------------------------------------------------------------------
# Benchmark runner — loads the standard eval suite and runs all 3 metrics
# ---------------------------------------------------------------------------

def run_full_benchmark(
    evaluator:         PipelineEvaluator,
    gold_triples_path: str | Path,
    benchmark_path:    str | Path,
    events_by_name:    dict,
    output_dir:        str | Path = "outputs/evaluation",
) -> dict:
    """
    Run the complete evaluation and save results to `output_dir`.

    Returns a summary dict:
        {
          "graph_construction": recall,
          "retrieval_quality":  mean_coverage,
          "report_coverage":    mean_coverage,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_path = Path(benchmark_path)
    suite = json.loads(benchmark_path.read_text(encoding="utf-8"))

    # Metric 1
    gc = evaluator.graph_construction_score(gold_triples_path)
    (output_dir / "metric1_graph_construction.json").write_text(
        json.dumps({
            "recall":    gc.recall,
            "precision": gc.precision,
            "f1":        gc.f1,
            "per_relation": gc.per_relation,
        }, indent=2),
        encoding="utf-8",
    )

    # Metric 2
    rq = evaluator.retrieval_quality_score(suite)
    (output_dir / "metric2_retrieval_quality.json").write_text(
        json.dumps({
            "mean_coverage": rq.mean_coverage,
            "per_query":     rq.per_query,
        }, indent=2),
        encoding="utf-8",
    )

    # Metric 3
    rc = evaluator.report_coverage_score(suite, events_by_name)
    (output_dir / "metric3_report_coverage.json").write_text(
        json.dumps({
            "mean_coverage": rc.mean_coverage,
            "per_query":     rc.per_query,
        }, indent=2),
        encoding="utf-8",
    )

    summary = {
        "graph_construction": gc.recall,
        "retrieval_quality":  rq.mean_coverage,
        "report_coverage":    rc.mean_coverage,
    }

    print("\n=== Benchmark Summary ===")
    print(f"  Graph Construction : {gc.recall:.0%}")
    print(f"  Retrieval Quality  : {rq.mean_coverage:.0%}")
    print(f"  Report Coverage    : {rc.mean_coverage:.0%}")

    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
