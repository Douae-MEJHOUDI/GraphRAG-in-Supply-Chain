"""
semantic_cache.py
-----------------
SemanticCache — query-level caching with semantic similarity matching.
CachedPipeline — drop-in wrapper around SimulationEngine that checks the
                 cache before running the full pipeline.

Implements the flow:

    Query
      ↓
    Semantic cache check
      ├── Cache HIT  → return cached Risk Report immediately
      └── Cache MISS → Full pipeline → Store in cache → return Risk Report

Queries are embedded with the same BGE model used for node retrieval
(BAAI/bge-large-en-v1.5, 1024-dim, L2-normalised, IndexFlatIP).
Two queries are considered a cache hit when their cosine similarity
exceeds HIT_THRESHOLD (default 0.92).

Cache is persisted to disk as two files:
  <cache_dir>/query_index.bin       — FAISS index of embedded queries
  <cache_dir>/query_index.bin.meta  — JSON list of {query, report, event_name}

Usage
-----
    engine = SimulationEngine(G, pipeline)
    cache  = SemanticCache()
    cpipe  = CachedPipeline(engine, cache)

    report = cpipe.run(event, query="What if Taiwan has a conflict?")
    # second call with a semantically equivalent query hits the cache
    report = cpipe.run(event, query="Impact of a Taiwan conflict?")
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.simulation.engine import SimulationEngine, SimulationResult
from src.simulation.events import DisruptionEvent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL     = "BAAI/bge-large-en-v1.5"  # same model as node encoder
HIT_THRESHOLD     = 0.92                        # cosine similarity for cache hit
DEFAULT_CACHE_DIR = Path("data/processed/cache")


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """One stored query + its serialised risk report."""
    query:      str
    event_name: str
    report:     dict   # serialised SimulationResult payload


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Semantic query cache backed by a FAISS IndexFlatIP index.

    All stored query embeddings are L2-normalised, so inner-product search
    is equivalent to cosine similarity search.

    Parameters
    ----------
    cache_dir   : directory for persisted index + metadata files
    model_name  : sentence-transformer to use for query embedding
    threshold   : cosine similarity threshold for a cache hit
    """

    def __init__(
        self,
        cache_dir:  Path | str = DEFAULT_CACHE_DIR,
        model_name: str        = DEFAULT_MODEL,
        threshold:  float      = HIT_THRESHOLD,
    ):
        self.cache_dir = Path(cache_dir)
        self.threshold = threshold
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Cache] Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._dim   = self._model.get_sentence_embedding_dimension()

        self._index:   faiss.IndexFlatIP = faiss.IndexFlatIP(self._dim)
        self._entries: list[CacheEntry]  = []

        # Load persisted cache if it exists
        self._index_path = self.cache_dir / "query_index.bin"
        self._meta_path  = self.cache_dir / "query_index.bin.meta"
        if self._index_path.exists() and self._meta_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, query: str) -> Optional[CacheEntry]:
        """
        Return a CacheEntry if a semantically similar query was already
        answered, otherwise return None (cache miss).
        """
        if self._index.ntotal == 0:
            return None

        emb = self._embed(query)
        scores, indices = self._index.search(emb, 1)
        best_score = float(scores[0][0])
        best_idx   = int(indices[0][0])

        if best_idx < 0 or best_score < self.threshold:
            return None

        print(f"[Cache] HIT  (similarity={best_score:.4f}) — "
              f"returning cached report for: {self._entries[best_idx].query!r}")
        return self._entries[best_idx]

    def store(self, query: str, event_name: str, report: dict) -> None:
        """
        Embed the query and store it with its report payload.
        Persists index and metadata to disk after each write.
        """
        emb = self._embed(query)
        self._index.add(emb)
        self._entries.append(CacheEntry(query=query, event_name=event_name, report=report))
        self._save()
        print(f"[Cache] STORED query ({self._index.ntotal} total): {query!r}")

    def size(self) -> int:
        return self._index.ntotal

    def clear(self) -> None:
        """Wipe the cache (in memory and on disk)."""
        self._index   = faiss.IndexFlatIP(self._dim)
        self._entries = []
        if self._index_path.exists():
            self._index_path.unlink()
        if self._meta_path.exists():
            self._meta_path.unlink()
        print("[Cache] Cleared.")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        meta = [
            {"query": e.query, "event_name": e.event_name, "report": e.report}
            for e in self._entries
        ]
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        self._index = faiss.read_index(str(self._index_path))
        raw = json.loads(self._meta_path.read_text(encoding="utf-8"))
        self._entries = [
            CacheEntry(query=r["query"], event_name=r["event_name"], report=r["report"])
            for r in raw
        ]
        print(f"[Cache] Loaded {self._index.ntotal} cached queries from {self.cache_dir}")

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Return a (1, dim) float32 L2-normalised embedding matrix."""
        emb = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        faiss.normalize_L2(emb)
        return emb


# ---------------------------------------------------------------------------
# CachedPipeline
# ---------------------------------------------------------------------------

class CachedPipeline:
    """
    Drop-in wrapper around SimulationEngine that adds semantic caching.

    On every call to run():
      1. Check the SemanticCache for a similar previous query.
      2. On hit  → return the cached report as a dict (no LLM call).
      3. On miss → run the full SimulationEngine pipeline, store result.

    Parameters
    ----------
    engine     : a fully configured SimulationEngine instance
    cache      : a SemanticCache instance (shared or dedicated)
    """

    def __init__(self, engine: SimulationEngine, cache: SemanticCache):
        self.engine = engine
        self.cache  = cache

    def run(
        self,
        event: DisruptionEvent,
        query: Optional[str] = None,
    ) -> dict:
        """
        Run the pipeline with semantic cache.

        Returns the risk report as a dict (same structure as SimulationResult.save()).
        On a cache hit, the full pipeline is skipped entirely.

        Parameters
        ----------
        event : DisruptionEvent
        query : str, optional — defaults to auto-generated from event name

        Returns
        -------
        dict : risk report payload
        """
        if query is None:
            query = (
                f"What are the supply chain impacts of the {event.name}? "
                f"Which entities are most exposed and what are the critical "
                f"dependency paths?"
            )

        # --- Cache check ---
        hit = self.cache.lookup(query)
        if hit is not None:
            return hit.report

        # --- Cache miss: run full pipeline ---
        print(f"[CachedPipeline] Cache miss — running full pipeline for: {query!r}")
        result: SimulationResult = self.engine.run(event=event, query=query)

        # Serialise result to a plain dict (same structure as SimulationResult.save())
        payload = {
            "event_name":     result.event.name,
            "event_category": result.event.category.value,
            "initial_shock":  result.event.initial_shock,
            "stats":          result.propagation.stats,
            "top_10_exposed": result.propagation.top_n(10),
            "critical_nodes": result.propagation.critical_nodes(),
            "high_nodes":     result.propagation.high_nodes(),
            "tiers":          result.propagation.tiers,
            "merged_scores":  result.merged_scores,
            "risk_report":    result.risk_report_prompt,
        }

        # Store in cache
        self.cache.store(
            query=query,
            event_name=result.event.name,
            report=payload,
        )

        return payload
