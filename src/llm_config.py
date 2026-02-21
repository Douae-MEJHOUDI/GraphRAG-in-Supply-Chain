"""
LLM and embedding configuration.

LLM strategy (dual-model):
  - llama-3.1-8b-instant     → fast, used for all regular text generation
  - llama-4-scout-17b        → used ONLY for keyword extraction (needs json_schema)

Embed: sentence-transformers all-MiniLM-L6-v2 — runs on CPU / MPS
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from lightrag.llm.openai import openai_complete_if_cache
from dotenv import load_dotenv

load_dotenv()

# ── Groq models ─────────────────────────────────────────────────────────────
# llama-3.1-8b-instant has only 6,000 TPM — a single query context already
# exceeds that. llama-3.3-70b-versatile has a much higher limit and handles
# large contexts fine.
_FAST_MODEL       = "llama-3.3-70b-versatile"                     # regular generation + ingestion
_STRUCTURED_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"   # json_schema (keyword extraction only)

# ── Embedding model ──────────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


# ── Dual-model LLM function (signature expected by LightRAG) ─────────────────
async def groq_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list | None = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """
    Route to the right Groq model depending on the call type:
      - keyword_extraction=True  → structured output required → llama-4-scout
      - keyword_extraction=False → regular generation → llama-3.1-8b (faster)
    """
    model = _STRUCTURED_MODEL if keyword_extraction else _FAST_MODEL

    return await openai_complete_if_cache(
        model,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        keyword_extraction=keyword_extraction,
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        **kwargs,
    )


# ── Async embedding function (signature expected by LightRAG) ────────────────
async def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    return model.encode(texts, convert_to_numpy=True)
