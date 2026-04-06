"""
build_graph.py
--------------
CLI entry point for LLMGraphBuilder.

Usage (Ollama, local GPU — recommended):
    python -m src.graph.build_graph \\
        --input       data/raw/ \\
        --output-dir  data/processed/ \\
        --model       qwen2.5:32b \\
        --backend     ollama

Usage (Groq, cloud fallback):
    python -m src.graph.build_graph \\
        --input       data/raw/ \\
        --output-dir  data/processed/ \\
        --model       llama-3.3-70b-versatile \\
        --backend     groq

Resume (automatically skips already-processed chunks):
    Just re-run the same command — the progress log is at
    data/processed/llm_graph.progress.jsonl by default.

Quick test on a single file:
    python -m src.graph.build_graph \\
        --input data/raw/my_document.txt \\
        --max-docs 1
"""

import argparse
from pathlib import Path

from src.graph.llm_builder import LLMGraphBuilder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a supply chain knowledge graph with an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i",
        default="data/raw/",
        help="Path to a directory of .txt/.json/.jsonl files, or a single file.",
    )
    p.add_argument(
        "--output-dir", "-o",
        default="data/processed/",
        help="Directory to save graph, triples, canon_map, and stats.",
    )
    p.add_argument(
        "--prefix",
        default="llm_graph",
        help="Filename prefix for all output files.",
    )
    p.add_argument(
        "--model",
        default="qwen2.5:32b",
        help="LLM model name.",
    )
    p.add_argument(
        "--backend",
        choices=["ollama", "groq"],
        default="ollama",
        help="LLM backend. 'ollama' uses local GPU; 'groq' uses cloud API.",
    )
    p.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama server base URL.",
    )
    p.add_argument(
        "--groq-api-key",
        default="",
        help="Groq API key (reads GROQ_API_KEY from env if not given).",
    )
    p.add_argument(
        "--chunk-chars",
        type=int,
        default=2400,
        help="Max characters per chunk sent to the LLM.",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="Overlap characters between consecutive chunks.",
    )
    p.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum triple confidence threshold.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Retries per LLM call on transient failures.",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="LLM call timeout in seconds.",
    )
    p.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Cap number of documents processed (0 = all). Useful for testing.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring any existing progress log.",
    )
    p.add_argument(
        "--prune-isolated",
        action="store_true",
        help="Remove nodes with no edges from the final graph.",
    )
    p.add_argument(
        "--skip-canonicalize",
        action="store_true",
        help="Skip the LLM canonicalization pass (faster but lower quality).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    builder = LLMGraphBuilder(
        model                = args.model,
        backend              = args.backend,
        ollama_base_url      = args.ollama_url,
        groq_api_key         = args.groq_api_key,
        chunk_chars          = args.chunk_chars,
        chunk_overlap        = args.chunk_overlap,
        confidence_threshold = args.confidence,
        max_retries          = args.max_retries,
        timeout              = args.timeout,
    )

    # Load documents
    input_path = Path(args.input)
    builder.load_documents(input_path)

    # Cap documents if requested (useful for quick tests)
    if args.max_docs and args.max_docs < len(builder.documents):
        builder.documents = builder.documents[: args.max_docs]
        print(f"[build_graph] Capped to {args.max_docs} document(s) for testing")

    # Phase 1: extraction
    progress_log = Path(args.output_dir) / f"{args.prefix}.progress.jsonl"
    builder.extract(
        progress_log=progress_log,
        resume=not args.no_resume,
    )

    # Phase 2: canonicalization (optional but highly recommended)
    if not args.skip_canonicalize:
        builder.canonicalize()
    else:
        print("[build_graph] Skipping canonicalization (--skip-canonicalize)")
        builder.build_graph()

    if not args.skip_canonicalize:
        builder.build_graph()

    # Phase 3+: validate, weights, save
    builder.validate_graph(prune_isolated=args.prune_isolated)
    builder.assign_weights()
    builder.save_all(args.output_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
