import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)



def _pct(f: float) -> str:
    return f"{f*100:.1f}%"

def _pass(b: bool) -> str:
    return "PASS" if b else "FAIL"


def build_report(result) -> str:
    lines = []

    lines += [
        "# Supply Chain GraphRAG — Evaluation Report",
        "",
        "## Overview",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Graph Coverage Pass Rate | {_pct(result.coverage_pass_rate)} "
        f"({sum(1 for r in result.coverage if r.passed)}/{len(result.coverage)} tests) |",
        f"| Retrieval Mean Hit Rate  | {_pct(result.retrieval_mean_hit)} "
        f"(avg across {len(result.retrieval)} queries) |",
    ]

    if result.report_facts:
        lines.append(
            f"| Report Factual Coverage | {_pct(result.report_mean_coverage)} "
            f"(avg across {len(result.report_facts)} queries) |"
        )
    else:
        lines.append("| Report Factual Coverage | Not run (use --reports flag) |")

    lines += ["", f"| Known Gaps Documented | {len(result.known_gaps)} |", ""]

    lines += [
        "---",
        "",
        "## 1. Graph Coverage Tests",
        "",
        "Tests whether known facts exist in the graph as nodes or edges.",
        "Sources: USGS 2024, SEC filings, iFixit teardowns.",
        "",
        "| Test | Result | Actual | Expected Min |",
        "|------|--------|--------|--------------|",
    ]
    for r in result.coverage:
        icon = "✓" if r.passed else "✗"
        lines.append(f"| {r.description} | {icon} {_pass(r.passed)} | {r.actual} | >= {r.expected_min} |")

    passed = sum(1 for r in result.coverage if r.passed)
    lines += [
        "",
        f"**Coverage pass rate: {passed}/{len(result.coverage)} ({_pct(result.coverage_pass_rate)})**",
        "",
    ]

    lines += [
        "---",
        "",
        "## 2. Retrieval Quality",
        "",
        "For each benchmark query, checks whether expected entities appear",
        "in the retrieved subgraph. Hit Rate = found / expected.",
        "",
        "| Query | Intent | Hit Rate | Found | Missing |",
        "|-------|--------|----------|-------|---------|",
    ]
    for r in result.retrieval:
        found_str   = ", ".join(r.found)   or "—"
        missing_str = ", ".join(r.missing) or "—"
        icon = "✓" if r.hit_rate == 1.0 else ("~" if r.hit_rate >= 0.5 else "✗")
        lines.append(
            f"| {r.query[:55]}... | {r.intent_type} | "
            f"{icon} {_pct(r.hit_rate)} | {found_str} | {missing_str} |"
        )

    lines += [
        "",
        f"**Mean hit rate: {_pct(result.retrieval_mean_hit)}**",
        "",
        "### Latency",
        "",
        "| Query | Latency (ms) |",
        "|-------|-------------|",
    ]
    for r in result.retrieval:
        lines.append(f"| {r.query[:60]} | {r.latency_ms} ms |")
    lines.append("")

    if result.report_facts:
        lines += [
            "---",
            "",
            "## 3. Report Factual Accuracy",
            "",
            "Checks whether generated LLM reports contain expected factual claims.",
            "Coverage = fraction of expected claims found in report text.",
            "",
            "| Query | Coverage | Found | Missing |",
            "|-------|----------|-------|---------|",
        ]
        for r in result.report_facts:
            found_str   = ", ".join(r.found_facts)   or "—"
            missing_str = ", ".join(r.missing_facts) or "—"
            icon = "✓" if r.coverage == 1.0 else ("~" if r.coverage >= 0.5 else "✗")
            lines.append(
                f"| {r.query[:55]} | {icon} {_pct(r.coverage)} | {found_str} | {missing_str} |"
            )
        lines += [
            "",
            f"**Mean factual coverage: {_pct(result.report_mean_coverage)}**",
            "",
        ]

    lines += [
        "---",
        "",
        "## 4. Known Data Gaps",
        "",
        "Relationships that should exist in the graph but are missing.",
        "Documented honestly as system limitations.",
        "",
    ]
    for desc, reason, severity in result.known_gaps:
        lines += [
            f"### [{severity}] {desc}",
            "",
            f"{reason}",
            "",
        ]

    overall = (
        result.coverage_pass_rate * 0.4 +
        result.retrieval_mean_hit  * 0.6
    )
    lines += [
        "---",
        "",
        "## 5. Summary",
        "",
        "| Component | Weight | Score |",
        "|-----------|--------|-------|",
        f"| Graph Coverage    | 40% | {_pct(result.coverage_pass_rate)} |",
        f"| Retrieval Quality | 60% | {_pct(result.retrieval_mean_hit)} |",
        f"| **Weighted Total** | 100% | **{_pct(overall)}** |",
        "",
        "### Interpretation",
        "",
        "| Score | Grade |",
        "|-------|-------|",
        "| >= 85% | Strong — graph has high coverage and retrieval is reliable |",
        "| 70–84% | Good — core facts present, some retrieval gaps |",
        "| 50–69% | Fair — graph works but missing important relationships |",
        "| < 50%  | Weak — significant data or retrieval failures |",
        "",
    ]

    if overall >= 0.85:
        grade = "Strong"
    elif overall >= 0.70:
        grade = "Good"
    elif overall >= 0.50:
        grade = "Fair"
    else:
        grade = "Weak"

    lines += [
        f"**Overall system rating: {grade} ({_pct(overall)})**",
        "",
        "> Note: Report factual accuracy not included in weighted score",
        "> (requires --reports flag and ANTHROPIC_API_KEY).",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the GraphRAG pipeline")
    parser.add_argument("--reports", action="store_true",
                        help="Also test LLM report factual accuracy (requires API key)")
    parser.add_argument("--out", default="",
                        help="Write markdown report to this file (default: print to stdout)")
    args = parser.parse_args()

    from m_eval.evaluator import Evaluator
    ev = Evaluator()
    result = ev.run_all(include_reports=args.reports)

    report_md = build_report(result)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report_md, encoding="utf-8")
        print(f"\nEvaluation report written to: {out_path}")
    else:
        print("\n" + report_md)

    sys.exit(0 if result.coverage_pass_rate >= 0.70 else 1)


if __name__ == "__main__":
    main()
