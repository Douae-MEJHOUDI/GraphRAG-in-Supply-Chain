"""
Document ingestion script.

Run once to build the knowledge graph from all documents in data/raw/:
    python -m src.ingest

Supports .txt and .pdf files.
"""

import os
import glob
from pathlib import Path
from lightrag import LightRAG

from src.pipeline import build_rag, insert_text

DATA_DIR = "./data/raw"


def load_txt(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(filepath: str) -> str:
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print("[WARN] PyPDF2 not installed — skipping PDF:", filepath)
        return ""


def ingest_all(rag: LightRAG, data_dir: str = DATA_DIR) -> int:
    """
    Load every .txt and .pdf from data_dir and insert into the graph.
    Returns the number of documents inserted.
    """
    files = (
        glob.glob(os.path.join(data_dir, "*.txt"))
        + glob.glob(os.path.join(data_dir, "*.pdf"))
    )

    if not files:
        print(f"[WARN] No documents found in {data_dir}")
        return 0

    for filepath in files:
        ext = Path(filepath).suffix.lower()
        print(f"Loading: {filepath}")

        if ext == ".txt":
            text = load_txt(filepath)
        elif ext == ".pdf":
            text = load_pdf(filepath)
        else:
            continue

        if text.strip():
            insert_text(rag, text)
            print(f"  [OK] Inserted ({len(text):,} chars)")
        else:
            print(f"  [SKIP] Empty or unreadable file")

    return len(files)


if __name__ == "__main__":
    print("Building knowledge graph from documents in data/raw/ ...")
    rag = build_rag()
    n = ingest_all(rag)
    print(f"\nDone. {n} document(s) ingested into graph_storage/")
    print("You can now run: streamlit run app/chatbot.py")
