import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import kuzu

logger = logging.getLogger(__name__)

GRAPH_DIR = Path("data/graph/supply_chain.db")


@dataclass
class CoverageResult:
    description: str
    passed: bool
    actual: int
    expected_min: int


@dataclass
class RetrievalResult:
    query: str
    intent_type: str
    expected: list[str]
    found: list[str]
    missing: list[str]
    hit_rate: float
    latency_ms: float


@dataclass
class ReportFactResult:
    query: str
    expected_facts: list[str]
    found_facts: list[str]
    missing_facts: list[str]
    coverage: float
    latency_ms: float


@dataclass
class EvalResult:
    coverage:           list[CoverageResult] = field(default_factory=list)
    retrieval:          list[RetrievalResult] = field(default_factory=list)
    report_facts:       list[ReportFactResult] = field(default_factory=list)
    known_gaps:         list[tuple] = field(default_factory=list)

    coverage_pass_rate:     float = 0.0
    retrieval_mean_hit:     float = 0.0
    report_mean_coverage:   float = 0.0


class Evaluator:

    def __init__(self):
        self.db   = kuzu.Database(str(GRAPH_DIR))
        self.conn = kuzu.Connection(self.db)

    def run_coverage(self) -> list[CoverageResult]:
        from m_eval.benchmark import GRAPH_COVERAGE
        results = []
        for desc, cypher, expected_min in GRAPH_COVERAGE:
            try:
                r = self.conn.execute(cypher)
                actual = r.get_next()[0] if r.has_next() else 0
                passed = int(actual) >= expected_min
            except Exception as ex:
                logger.warning(f"Coverage query failed: {ex}")
                actual  = -1
                passed  = False
            results.append(CoverageResult(desc, passed, int(actual), expected_min))
        return results

    def run_retrieval(self) -> list[RetrievalResult]:
        from m_eval.benchmark import RETRIEVAL_CASES
        from m6_rag_retrieval.retriever import GraphRetriever
        from m6_rag_retrieval.context_builder import parse_query

        retriever      = GraphRetriever.__new__(GraphRetriever)
        retriever.db   = self.db
        retriever.conn = self.conn
        results        = []

        for query, expected, intent_type in RETRIEVAL_CASES:
            t0 = time.time()

            intent   = parse_query(query)
            subgraph = retriever.retrieve(intent)

            ms = (time.time() - t0) * 1000

            retrieved_names = set()
            for key in ("companies", "countries", "minerals", "products"):
                for item in subgraph.get(key, []):
                    retrieved_names.add(item["name"])
            for e in subgraph.get("supply_edges", []) + subgraph.get("depends_edges", []):
                retrieved_names.update([e.get("from",""), e.get("to",""),
                                        e.get("product",""), e.get("company","")])
            for e in subgraph.get("produces_edges", []):
                retrieved_names.update([e.get("country",""), e.get("mineral","")])

            found   = [e for e in expected if any(
                e.lower() in name.lower() for name in retrieved_names
            )]
            missing = [e for e in expected if e not in found]
            hit_rate = len(found) / len(expected) if expected else 1.0

            results.append(RetrievalResult(
                query=query,
                intent_type=intent_type,
                expected=expected,
                found=found,
                missing=missing,
                hit_rate=hit_rate,
                latency_ms=round(ms, 1),
            ))

        return results

    def run_report_facts(self) -> list[ReportFactResult]:
 
        from m_eval.benchmark import REPORT_FACTS
        from m6_rag_retrieval.retriever import GraphRetriever
        from m6_rag_retrieval.context_builder import parse_query, build_context
        from m7_report_generation.report_generator import ReportGenerator

        if not os.environ.get("ANTHROPIC_API_KEY"):
            logger.warning("ANTHROPIC_API_KEY not set — skipping report fact evaluation.")
            return []

        retriever      = GraphRetriever.__new__(GraphRetriever)
        retriever.db   = self.db
        retriever.conn = self.conn
        generator      = ReportGenerator()
        results        = []

        for query, expected_facts in REPORT_FACTS:
            t0 = time.time()
            try:
                intent   = parse_query(query)
                subgraph = retriever.retrieve(intent)
                context  = build_context(subgraph)
                report   = generator.generate(query, subgraph, context)
                report_text = report.llm_response.lower()
            except Exception as ex:
                logger.error(f"Report generation failed for '{query}': {ex}")
                results.append(ReportFactResult(
                    query=query,
                    expected_facts=expected_facts,
                    found_facts=[],
                    missing_facts=expected_facts,
                    coverage=0.0,
                    latency_ms=0.0,
                ))
                continue

            ms = (time.time() - t0) * 1000

            found   = [f for f in expected_facts if f.lower() in report_text]
            missing = [f for f in expected_facts if f not in found]
            coverage = len(found) / len(expected_facts) if expected_facts else 1.0

            results.append(ReportFactResult(
                query=query,
                expected_facts=expected_facts,
                found_facts=found,
                missing_facts=missing,
                coverage=coverage,
                latency_ms=round(ms, 1),
            ))

        return results

    def run_all(self, include_reports: bool = False) -> EvalResult:
        logger.info("[EVAL] Running graph coverage tests...")
        coverage = self.run_coverage()

        logger.info("[EVAL] Running retrieval quality tests...")
        retrieval = self.run_retrieval()

        report_facts = []
        if include_reports:
            logger.info("[EVAL] Running report factual accuracy tests...")
            report_facts = self.run_report_facts()

        from m_eval.benchmark import KNOWN_GAPS

        result = EvalResult(
            coverage=coverage,
            retrieval=retrieval,
            report_facts=report_facts,
            known_gaps=KNOWN_GAPS,
        )

        if coverage:
            result.coverage_pass_rate = sum(1 for r in coverage if r.passed) / len(coverage)
        if retrieval:
            result.retrieval_mean_hit = sum(r.hit_rate for r in retrieval) / len(retrieval)
        if report_facts:
            result.report_mean_coverage = sum(r.coverage for r in report_facts) / len(report_facts)

        return result
