"""
LightRAG pipeline — initialisation and query helpers.

LLM  : Groq API via OpenAI-compatible endpoint (free, no GPU needed)
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
from lightrag.llm.openai import openai_complete

from src.llm_config import embed_texts

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

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Copy .env.example to .env and add your key."
        )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        # LLM: use LightRAG's built-in openai_complete pointed at Groq
        llm_model_func=openai_complete,
        llm_model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # supports structured outputs on Groq
        llm_model_kwargs={
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": groq_api_key,
        },
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

def ask(rag: LightRAG, question: str, mode: str = "hybrid") -> str:
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
