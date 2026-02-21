"""
LightRAG pipeline — initialisation and query helpers.

LLM  : Groq API, dual-model routing (see llm_config.py)
Embed: sentence-transformers all-MiniLM-L6-v2 (local CPU)

Usage:
    from src.pipeline import build_rag, insert_text, ask

    rag = build_rag()
    insert_text(rag, "Some document text...")
    answer = ask(rag, "Which suppliers were affected by the Suez blockage?")
"""

import os
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop

from src.llm_config import groq_complete, embed_texts

load_dotenv()

WORKING_DIR = "./graph_storage"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# All modes supported by this LightRAG version
MODES = {
    "naive":  "Standard RAG (no graph) — baseline",
    "local":  "Entity-level graph retrieval",
    "global": "Community-level (big picture) retrieval",
    "hybrid": "Local + Global — best for complex questions",
    "mix":    "Hybrid + vector search — most comprehensive",
}


def build_rag() -> LightRAG:
    """
    Initialise (or reload) a LightRAG instance backed by graph_storage/.
    Storages are initialised synchronously before returning.
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    if not os.getenv("GROQ_API_KEY"):
        raise EnvironmentError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
        )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        # Custom dual-model function: fast 8b for answers, 17b only for keyword extraction
        llm_model_func=groq_complete,
        # Groq free tier has only 6,000 TPM for llama-3.1-8b-instant.
        # max_async=1 → sequential requests → no rate-limit collisions.
        llm_model_max_async=1,
        # Longer timeout so the openai client has time to retry after rate-limit waits
        # (worker timeout = 2 × this value, so 240 s total before giving up)
        default_llm_timeout=120,
        # Smaller chunks → more chunks per document → better retrieval granularity
        chunk_token_size=400,
        chunk_overlap_token_size=50,
        # Embeddings: local sentence-transformers (no API call needed)
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=512,
            func=embed_texts,
        ),
    )

    # Required in LightRAG >= 1.3: initialise storage backends before use
    loop = always_get_an_event_loop()
    loop.run_until_complete(rag.initialize_storages())

    return rag


# --------------------------------------------------------------------------- #
# Insertion helpers
# --------------------------------------------------------------------------- #

def insert_text(rag: LightRAG, text: str) -> None:
    """Insert a raw text string into the knowledge graph."""
    rag.insert(text)


def insert_file(rag: LightRAG, filepath: str) -> None:
    """Read a .txt file and insert its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    insert_text(rag, text)
    print(f"[OK] Inserted: {filepath}")


# --------------------------------------------------------------------------- #
# Query helper
# --------------------------------------------------------------------------- #

def ask(rag: LightRAG, question: str, mode: str = "local") -> str:
    """
    Query the knowledge graph.

    Args:
        rag:      LightRAG instance from build_rag()
        question: Natural language question
        mode:     One of 'naive' | 'local' | 'global' | 'hybrid' | 'mix'

    Returns:
        Answer string from the LLM grounded in the graph.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {list(MODES)}")
    return rag.query(question, param=QueryParam(mode=mode))
