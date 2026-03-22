from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HEADER_LINE_PATTERNS = (
    re.compile(r"^Company:\s*", re.I),
    re.compile(r"^Ticker:\s*", re.I),
    re.compile(r"^CIK:\s*", re.I),
    re.compile(r"^Sector:\s*", re.I),
    re.compile(r"^Industry:\s*", re.I),
    re.compile(r"^\[Earnings Call\s*\|.*\]$", re.I),
)

TURN_SPLIT_RE = re.compile(
    r"(?=(?:[A-Z][A-Za-z0-9&.,'()/+\- ]{0,80})\s*:\s)"
)
TURN_RE = re.compile(
    r"^(?P<label>[A-Z][A-Za-z0-9&.,'()/+\- ]{0,80})\s*:\s*(?P<content>.*)$",
    re.S,
)

DROP_LABELS = {
    "operator",
    "executives",
    "analysts",
    "analyst",
    "participants",
    "presentation",
}

DROP_CONTENT_PATTERNS = (
    re.compile(r"\bforward-looking statements?\b", re.I),
    re.compile(r"\bsafe harbor\b", re.I),
    re.compile(r"\brisk factors?\b", re.I),
    re.compile(r"\bthis call is being recorded\b", re.I),
    re.compile(r"\boperator instructions?\b", re.I),
    re.compile(r"\brebroadcast\b", re.I),
    re.compile(r"\bwebcast\b", re.I),
    re.compile(r"\bpress release\b", re.I),
    re.compile(r"\binvestor relations\b", re.I),
    re.compile(r"\btelephone replay\b", re.I),
    re.compile(r"\bconference call\b", re.I),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean the merged SP500 company corpus by removing headers and boilerplate."
    )
    parser.add_argument(
        "--input",
        default="data/interim/sp500_company_corpus.jsonl",
        help="Input company-level JSONL corpus.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/sp500_company_corpus_clean.jsonl",
        help="Output cleaned company-level JSONL corpus.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def strip_header_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if any(pattern.match(stripped) for pattern in HEADER_LINE_PATTERNS):
            continue
        lines.append(stripped)
    return "\n".join(lines)


def should_drop_turn(label: str, content: str) -> bool:
    normalized_label = label.strip().lower()
    if normalized_label in DROP_LABELS:
        return True
    if "operator" in normalized_label:
        return True
    if "analyst" in normalized_label and len(content.split()) < 20:
        return True
    return any(pattern.search(content) for pattern in DROP_CONTENT_PATTERNS)


def clean_turn_content(content: str) -> str:
    content = normalize_text(content)
    content = re.sub(r"\[[^\]]{0,120}\]", " ", content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def extract_turns(text: str) -> list[str]:
    text = strip_header_lines(normalize_text(text))
    if not text:
        return []

    parts = TURN_SPLIT_RE.split(text)
    turns = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        match = TURN_RE.match(part)
        if not match:
            continue
        label = match.group("label").strip()
        content = clean_turn_content(match.group("content"))
        if not content:
            continue
        if should_drop_turn(label, content):
            continue
        if len(content.split()) < 15:
            continue
        turns.append(f"{label}: {content}")
    return turns


def clean_record(record: dict) -> dict:
    turns = extract_turns(record.get("text", ""))
    cleaned_text = "\n\n".join(turns)
    new_record = dict(record)
    new_record["text"] = cleaned_text
    new_record["cleaned_turn_count"] = len(turns)
    new_record["word_count"] = len(cleaned_text.split())
    return new_record


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept_records = 0
    total_records = 0

    with input_path.open(encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            total_records += 1
            record = json.loads(line)
            cleaned = clean_record(record)
            if not cleaned["text"]:
                continue
            dst.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            kept_records += 1

    print(
        f"[clean_sp500] wrote {kept_records} cleaned company records "
        f"from {total_records} input records to {output_path}"
    )


if __name__ == "__main__":
    main()
