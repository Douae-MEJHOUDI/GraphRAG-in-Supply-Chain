from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


TRANSCRIPT_COLUMNS = ("transcript", "content")
SUPPLY_CHAIN_KEYWORDS = (
    "supplier",
    "suppliers",
    "supply chain",
    "depends on",
    "depend on",
    "rely on",
    "relies on",
    "single-source",
    "single source",
    "sole source",
    "raw material",
    "raw materials",
    "inventory",
    "logistics",
    "shipment",
    "shipments",
    "shipping",
    "delay",
    "delays",
    "manufacturing",
    "factory",
    "factories",
    "plant",
    "plants",
    "capacity",
    "procurement",
    "distribution",
    "warehouse",
    "warehousing",
    "bottleneck",
    "shortage",
    "shortages",
)
STRONG_KEYWORDS = {
    "supplier",
    "suppliers",
    "supply chain",
    "depends on",
    "depend on",
    "rely on",
    "relies on",
    "single-source",
    "single source",
    "sole source",
    "raw material",
    "raw materials",
    "logistics",
    "shipment",
    "shipments",
    "shipping",
    "delay",
    "delays",
    "procurement",
    "shortage",
    "shortages",
}
BOILERPLATE_PATTERNS = (
    re.compile(r"\[operator instructions?\]", re.I),
    re.compile(r"\bconference call\b", re.I),
    re.compile(r"\bwebcast\b", re.I),
    re.compile(r"\breplay\b", re.I),
    re.compile(r"\bforward-looking statements?\b", re.I),
    re.compile(r"\bsafe harbor\b", re.I),
    re.compile(r"\binvestor relations\b", re.I),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize and chunk SP500 transcript parquet files into JSONL."
    )
    parser.add_argument(
        "--input-glob",
        default="data/sp500/*.parquet",
        help="Glob for source parquet files.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/sp500_company_corpus.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=80,
        help="Minimum words required to keep a relevant transcript section.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional cap on number of transcript rows to process.",
    )
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="Keep all cleaned transcript text instead of supply-chain-only blocks.",
    )
    return parser.parse_args()


def choose_text_column(columns: Iterable[str]) -> str:
    for column in TRANSCRIPT_COLUMNS:
        if column in columns:
            return column
    raise ValueError(f"No transcript-like column found in {list(columns)}")


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_blocks(text: str) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    blocks = re.split(r"\n+(?=[A-Z][A-Za-z0-9 .,&'()/+-]{1,80}:)", text)
    if len(blocks) == 1:
        blocks = re.split(r"\n\n+", text)
    return [block.strip() for block in blocks if block.strip()]


def keyword_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [keyword for keyword in SUPPLY_CHAIN_KEYWORDS if keyword in lowered]


def is_boilerplate(text: str) -> bool:
    return any(pattern.search(text) for pattern in BOILERPLATE_PATTERNS)


def block_is_relevant(text: str) -> bool:
    hits = keyword_hits(text)
    if not hits:
        return False

    strong_hits = [hit for hit in hits if hit in STRONG_KEYWORDS]
    if strong_hits:
        return True

    return len(set(hits)) >= 2


def filter_blocks(blocks: list[str], keep_all: bool) -> list[str]:
    if keep_all:
        return [block for block in blocks if len(block.split()) >= 20]

    kept = []
    for block in blocks:
        if block_is_relevant(block):
            kept.append(block)
            continue
        if is_boilerplate(block):
            continue
    return kept


def normalize_scalar(value):
    try:
        return None if pd.isna(value) else value
    except TypeError:
        return value


def normalize_row_value(value):
    value = normalize_scalar(value)
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def company_key(row: dict) -> tuple[str, str]:
    ticker = str(normalize_row_value(row.get("ticker") or row.get("symbol")) or "")
    company = str(
        normalize_row_value(row.get("company") or row.get("company_name")) or ""
    )
    return ticker, company


def build_section(row: dict, merged_text: str, source_file: str) -> dict:
    year = normalize_row_value(row.get("year"))
    quarter = normalize_row_value(row.get("quarter"))
    date = normalize_row_value(row.get("earnings_date") or row.get("date"))
    label_bits = []
    if date:
        label_bits.append(str(date))
    if year is not None and quarter is not None:
        label_bits.append(f"{year} Q{quarter}")

    header = " | ".join(label_bits) if label_bits else source_file
    return {
        "header": header,
        "source_file": source_file,
        "date": date,
        "year": year,
        "quarter": quarter,
        "keyword_hits": keyword_hits(merged_text),
        "text": merged_text,
    }


def format_company_document(bucket: dict) -> str:
    lines = [
        f"Company: {bucket['company']}",
        f"Ticker: {bucket['ticker']}",
    ]
    if bucket.get("cik") is not None:
        lines.append(f"CIK: {bucket['cik']}")
    if bucket.get("sector"):
        lines.append(f"Sector: {bucket['sector']}")
    if bucket.get("industry"):
        lines.append(f"Industry: {bucket['industry']}")
    lines.append("")

    for section in bucket["sections"]:
        lines.append(f"[Earnings Call | {section['header']}]")
        lines.append(section["text"])
        lines.append("")

    return "\n".join(lines).strip()


def build_company_record(bucket: dict) -> dict:
    all_keywords = sorted(
        {keyword for section in bucket["sections"] for keyword in section["keyword_hits"]}
    )
    return {
        "source": f"sp500_company_{bucket['ticker'] or bucket['company']}",
        "doc_type": "earnings_call_company_corpus",
        "company": bucket["company"],
        "ticker": bucket["ticker"],
        "cik": bucket.get("cik"),
        "sector": bucket.get("sector"),
        "industry": bucket.get("industry"),
        "transcript_count": len(bucket["sections"]),
        "keyword_hits": all_keywords,
        "text": format_company_document(bucket),
    }


def process_parquet(
    parquet_path: Path,
    min_words: int,
    keep_all: bool,
    remaining_docs: int | None,
    buckets: dict[tuple[str, str], dict],
) -> int:
    df = pd.read_parquet(parquet_path)
    text_column = choose_text_column(df.columns)

    docs_written = 0

    for _, row in df.iterrows():
        if remaining_docs is not None and docs_written >= remaining_docs:
            break

        row_dict = row.to_dict()
        raw_text = row_dict.get(text_column)
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        blocks = split_blocks(raw_text)
        kept_blocks = filter_blocks(blocks, keep_all=keep_all)
        if not kept_blocks:
            continue

        merged_text = "\n\n".join(kept_blocks)
        if len(merged_text.split()) < min_words:
            continue

        ticker, company = company_key(row_dict)
        key = (ticker, company)
        bucket = buckets.setdefault(
            key,
            {
                "company": company or "Unknown Company",
                "ticker": ticker or "unknown",
                "cik": normalize_row_value(row_dict.get("cik")),
                "sector": normalize_row_value(row_dict.get("sector")),
                "industry": normalize_row_value(row_dict.get("industry")),
                "sections": [],
            },
        )

        if bucket.get("cik") is None:
            bucket["cik"] = normalize_row_value(row_dict.get("cik"))
        if not bucket.get("sector"):
            bucket["sector"] = normalize_row_value(row_dict.get("sector"))
        if not bucket.get("industry"):
            bucket["industry"] = normalize_row_value(row_dict.get("industry"))

        bucket["sections"].append(
            build_section(row_dict, merged_text, parquet_path.name)
        )
        docs_written += 1

    return docs_written


def main() -> None:
    args = parse_args()

    input_paths = sorted(Path().glob(args.input_glob))
    if not input_paths:
        raise FileNotFoundError(f"No parquet files matched {args.input_glob}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    docs_left = args.max_docs
    buckets: dict[tuple[str, str], dict] = {}

    for parquet_path in input_paths:
        if docs_left is not None and docs_left <= 0:
            break

        docs_written = process_parquet(
            parquet_path=parquet_path,
            min_words=args.min_words,
            keep_all=args.keep_all,
            remaining_docs=docs_left,
            buckets=buckets,
        )
        total_docs += docs_written
        if docs_left is not None:
            docs_left -= docs_written

        print(f"[sp500] {parquet_path.name}: kept {docs_written} transcript docs")

    sorted_buckets = sorted(
        buckets.values(),
        key=lambda bucket: (bucket["ticker"], bucket["company"]),
    )

    with output_path.open("w", encoding="utf-8") as writer:
        for bucket in sorted_buckets:
            bucket["sections"].sort(
                key=lambda section: (
                    str(section.get("date") or ""),
                    str(section.get("year") or ""),
                    str(section.get("quarter") or ""),
                )
            )
            record = build_company_record(bucket)
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[sp500] wrote {len(sorted_buckets)} company documents from {total_docs} "
        f"transcript docs to {output_path}"
    )


if __name__ == "__main__":
    main()
