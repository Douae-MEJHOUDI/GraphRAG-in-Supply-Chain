from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TURN_SPLIT_RE = re.compile(
    r"(?=(?:[A-Z][A-Za-z0-9&.,'()/+\- ]{0,80})\s*:\s)"
)

STRONG_KEEP_KEYWORDS = (
    "supplier",
    "suppliers",
    "supply chain",
    "depends on",
    "depend on",
    "rely on",
    "relies on",
    "single source",
    "single-source",
    "sole source",
    "raw material",
    "raw materials",
    "logistics",
    "shipment",
    "shipments",
    "shipping",
    "freight",
    "delay",
    "delays",
    "shortage",
    "shortages",
    "bottleneck",
    "procurement",
    "distribution",
    "inventory",
    "plant",
    "plants",
    "factory",
    "factories",
    "manufacturing capacity",
    "capacity constraint",
    "capacity constraints",
)

SECONDARY_KEYWORDS = (
    "manufacturing",
    "capacity",
    "production",
    "producing",
    "produced",
    "operations",
    "operational",
    "facility",
    "facilities",
    "warehouse",
    "warehousing",
    "network",
    "distribution center",
)

DROP_IF_ONLY_FINANCE = (
    "revenue",
    "eps",
    "earnings",
    "gross margin",
    "operating margin",
    "buyback",
    "buybacks",
    "dividend",
    "guidance",
    "cash flow",
    "share repurchase",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the broad SP500 RAG corpus from cleaned company transcripts."
    )
    parser.add_argument(
        "--input",
        default="data/interim/sp500_company_corpus_clean.jsonl",
        help="Input cleaned company corpus.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/sp500_company_rag_corpus.jsonl",
        help="Output broad RAG corpus.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=20,
        help="Minimum words per kept passage.",
    )
    parser.add_argument(
        "--max-neighbors",
        type=int,
        default=1,
        help="How many neighboring turns to keep around a matched turn.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def split_turns(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = TURN_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def hits(text: str, keywords: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in keywords if keyword in lowered]


def is_finance_only(text: str) -> bool:
    lowered = text.lower()
    finance_hits = [keyword for keyword in DROP_IF_ONLY_FINANCE if keyword in lowered]
    strong_hits = hits(text, STRONG_KEEP_KEYWORDS)
    secondary_hits = hits(text, SECONDARY_KEYWORDS)
    return bool(finance_hits) and not strong_hits and len(set(secondary_hits)) < 2


def should_keep_turn(text: str, min_words: int) -> tuple[bool, list[str]]:
    if len(text.split()) < min_words:
        return False, []

    strong_hits = hits(text, STRONG_KEEP_KEYWORDS)
    if strong_hits:
        return True, strong_hits

    secondary_hits = hits(text, SECONDARY_KEYWORDS)
    if len(set(secondary_hits)) >= 2 and not is_finance_only(text):
        return True, secondary_hits

    return False, []


def filter_turns(
    turns: list[str], min_words: int, max_neighbors: int
) -> tuple[list[str], list[str]]:
    keep_indices: set[int] = set()
    matched_keywords: set[str] = set()

    for idx, turn in enumerate(turns):
        keep, matched = should_keep_turn(turn, min_words=min_words)
        if not keep:
            continue
        matched_keywords.update(matched)
        start = max(0, idx - max_neighbors)
        end = min(len(turns), idx + max_neighbors + 1)
        keep_indices.update(range(start, end))

    kept_turns = [turns[idx] for idx in sorted(keep_indices)]
    return kept_turns, sorted(matched_keywords)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    kept_records = 0

    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            turns = split_turns(record.get("text", ""))
            kept_turns, matched_keywords = filter_turns(
                turns, min_words=args.min_words, max_neighbors=args.max_neighbors
            )
            if not kept_turns:
                continue

            out = dict(record)
            out["text"] = "\n\n".join(kept_turns)
            out["filtered_turn_count"] = len(kept_turns)
            out["matched_keywords"] = matched_keywords
            out["word_count"] = len(out["text"].split())

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept_records += 1

    print(
        f"[filter_sp500] wrote {kept_records} RAG company records "
        f"from {total_records} cleaned records to {output_path}"
    )


if __name__ == "__main__":
    main()
