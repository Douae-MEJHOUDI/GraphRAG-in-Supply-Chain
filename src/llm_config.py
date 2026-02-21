"""
Embedding configuration.
- Embed : sentence-transformers all-MiniLM-L6-v2 — runs on CPU / MPS

Note: LLM is configured in pipeline.py using LightRAG's built-in
openai_complete with Groq's OpenAI-compatible API endpoint.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

_embed_model: SentenceTransformer | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        # ~90 MB, CPU-friendly, 384-dim vectors
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


async def embed_texts(texts: list[str]) -> np.ndarray:
    """Async embedding function expected by LightRAG's EmbeddingFunc."""
    model = _get_embed_model()
    return model.encode(texts, convert_to_numpy=True)
