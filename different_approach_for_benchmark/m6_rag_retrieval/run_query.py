import argparse
import logging
import sys

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


def run_query(query: str, context_only: bool = False) -> str:
    from m6_rag_retrieval.retriever import GraphRetriever
    from m6_rag_retrieval.context_builder import parse_query, build_context

    intent   = parse_query(query)
    retriever = GraphRetriever()
    subgraph  = retriever.retrieve(intent)
    context   = build_context(subgraph)

    if context_only:
        return context

    from m7_report_generation.report_generator import ReportGenerator
    generator = ReportGenerator()
    report    = generator.generate(query, subgraph, context)
    return report.to_markdown()


def interactive_loop():
    print("Supply Chain GraphRAG — Interactive Mode")
    print("Prefix query with '--context' to skip LLM. Type 'quit' to exit.\n")

    while True:
        try:
            raw = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not raw or raw.lower() in ("quit", "exit", "q"):
            break

        context_only = raw.startswith("--context")
        query = raw.replace("--context", "").strip()
        if not query:
            continue

        print("\n" + run_query(query, context_only=context_only) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Supply chain what-if query engine")
    parser.add_argument("query", nargs="?", help="Natural language query")
    parser.add_argument("--context-only", action="store_true",
                        help="Print graph context only (no LLM call)")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive query loop")
    args = parser.parse_args()

    if args.interactive:
        interactive_loop()
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print(run_query(args.query, context_only=args.context_only))


if __name__ == "__main__":
    main()
