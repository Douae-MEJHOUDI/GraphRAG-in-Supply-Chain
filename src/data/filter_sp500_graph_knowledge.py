from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TURN_SPLIT_RE = re.compile(
    r"(?=(?:[A-Z][A-Za-z0-9&.,'()/+\- ]{0,80})\s*:\s)"
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

DEPENDENCY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("depends_on", re.compile(r"\bdepend(?:s|ed|ing)? on\b", re.I)),
    ("relies_on", re.compile(r"\brel(?:y|ies|ied|ying) on\b", re.I)),
    ("supplied_by", re.compile(r"\bsupplied by\b", re.I)),
    ("supplies", re.compile(r"\bsuppl(?:y|ies|ied|ying)\b", re.I)),
    ("sources_from", re.compile(r"\bsource(?:d|s|ing)? from\b", re.I)),
    ("procures_from", re.compile(r"\bprocure(?:d|s|ing)? from\b", re.I)),
    ("buys_from", re.compile(r"\bbuy(?:s|ing|ought)? from\b", re.I)),
    ("single_source", re.compile(r"\b(?:single|sole)[ -]?source\b", re.I)),
)

DISRUPTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("supply_constraints", re.compile(r"\bsupply constraints?\b", re.I)),
    ("shortage", re.compile(r"\bshortages?\b", re.I)),
    ("delay", re.compile(r"\bdelay(?:s|ed|ing)?\b", re.I)),
    ("shipment_delay", re.compile(r"\b(?:delayed|late) shipments?\b", re.I)),
    ("disruption", re.compile(r"\bdisrupt(?:ion|ions|ed|ing)?\b", re.I)),
    ("shutdown", re.compile(r"\b(?:plant|factory)\s+shutdowns?\b", re.I)),
    ("closure", re.compile(r"\b(?:plant|factory)\s+closures?\b", re.I)),
    ("bottleneck", re.compile(r"\bbottlenecks?\b", re.I)),
)

SUPPLY_CONTEXT_TERMS = (
    "supplier",
    "suppliers",
    "supply chain",
    "supply",
    "shipment",
    "shipments",
    "shipping",
    "logistics",
    "factory",
    "factories",
    "plant",
    "plants",
    "manufacturing",
    "raw material",
    "raw materials",
    "inventory",
    "procurement",
    "distribution",
    "facility",
    "facilities",
    "warehouse",
    "warehousing",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stricter SP500 graph-knowledge corpus from the broader RAG corpus."
    )
    parser.add_argument(
        "--input",
        default="data/interim/sp500_company_rag_corpus.jsonl",
        help="Input broad RAG corpus.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/sp500_company_graph_knowledge.jsonl",
        help="Output stricter graph-knowledge corpus.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=8,
        help="Minimum words per kept sentence.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_turns(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = TURN_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def split_turn(turn: str) -> tuple[str, str]:
    if ":" not in turn:
        return "", turn.strip()
    speaker, body = turn.split(":", 1)
    return speaker.strip(), body.strip()


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]


def sentence_hits(sentence: str) -> list[str]:
    hits: list[str] = []
    lowered = sentence.lower()

    for label, pattern in DEPENDENCY_PATTERNS:
        if pattern.search(sentence):
            hits.append(label)

    disruption_hits = [
        label for label, pattern in DISRUPTION_PATTERNS if pattern.search(sentence)
    ]
    if disruption_hits and any(term in lowered for term in SUPPLY_CONTEXT_TERMS):
        hits.extend(disruption_hits)

    return sorted(set(hits))


def filter_turn(turn: str, min_words: int) -> tuple[str | None, list[str]]:
    speaker, body = split_turn(turn)
    if not body:
        return None, []

    kept_sentences: list[str] = []
    matched: set[str] = set()

    for sentence in split_sentences(body):
        if len(sentence.split()) < min_words:
            continue
        sentence_match = sentence_hits(sentence)
        if not sentence_match:
            continue
        kept_sentences.append(sentence)
        matched.update(sentence_match)

    if not kept_sentences:
        return None, []

    prefix = f"{speaker}: " if speaker else ""
    return prefix + " ".join(kept_sentences), sorted(matched)


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

            kept_turns: list[str] = []
            matched_patterns: set[str] = set()

            for turn in turns:
                kept_turn, turn_hits = filter_turn(turn, min_words=args.min_words)
                if not kept_turn:
                    continue
                kept_turns.append(kept_turn)
                matched_patterns.update(turn_hits)

            if not kept_turns:
                continue

            out = dict(record)
            out["text"] = "\n\n".join(kept_turns)
            out["graph_passage_count"] = len(kept_turns)
            out["matched_patterns"] = sorted(matched_patterns)
            out["word_count"] = len(out["text"].split())

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            kept_records += 1

    print(
        f"[filter_sp500_graph_knowledge] wrote {kept_records} graph company records "
        f"from {total_records} RAG records to {output_path}"
    )


if __name__ == "__main__":
    main()
