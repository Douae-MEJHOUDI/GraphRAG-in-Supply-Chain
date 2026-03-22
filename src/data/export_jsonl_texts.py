from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export selected JSONL records into a readable text file."
    )
    parser.add_argument(
        "--input",
        default="data/interim/sp500_company_rag_corpus.jsonl",
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="data/interim/sp500_company_rag_first3.txt",
        help="Output text file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of records to export from the top of the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    blocks: list[str] = []

    with input_path.open(encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            record = json.loads(line)
            blocks.append(
                "\n".join(
                    [
                        f"===== RECORD {exported + 1} =====",
                        f"Company: {record.get('company')}",
                        f"Ticker: {record.get('ticker')}",
                        f"Sector: {record.get('sector')}",
                        f"Industry: {record.get('industry')}",
                        f"Transcript Count: {record.get('transcript_count')}",
                        f"Filtered Turn Count: {record.get('filtered_turn_count')}",
                        f"Matched Keywords: {', '.join(record.get('matched_keywords', []))}",
                        "",
                        record.get("text", ""),
                    ]
                )
            )
            exported += 1
            if exported >= args.limit:
                break

    output_path.write_text("\n\n".join(blocks), encoding="utf-8")
    print(f"[export_jsonl_texts] wrote {exported} records to {output_path}")


if __name__ == "__main__":
    main()
