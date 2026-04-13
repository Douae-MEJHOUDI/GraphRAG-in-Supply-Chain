"""
disambiguator.py
----------------
Entity disambiguation for the supply chain knowledge graph.

The same real-world entity often appears under many surface forms in raw text:
  "TSMC", "Taiwan Semiconductor", "Taiwan Semiconductor Manufacturing Co."

This module resolves aliases to a single canonical name using two-stage matching:
  1. String normalization  — catches exact matches after cleaning
  2. Embedding similarity  — catches paraphrases within the same entity type

Designed to be called once after initial NER extraction and before graph
construction. All downstream code works with canonical names only.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.graph.schema import Entity, EntityType


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL     = "BAAI/bge-large-en-v1.5"  # 1024-dim, consistent with encoder
SIMILARITY_THRESH = 0.88                  # cosine sim threshold for merging
CROSS_TYPE_MERGE  = False                 # never merge entities of different types


# ---------------------------------------------------------------------------
# String helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Lowercase, strip punctuation, collapse whitespace, remove unicode accents.
    Used as a first-pass cheap comparison before computing embeddings.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Disambiguator class
# ---------------------------------------------------------------------------

class EntityDisambiguator:
    """
    Resolves a list of raw Entity objects to a deduplicated set with
    canonical names and merged alias lists.

    Usage
    -----
    >>> dis = EntityDisambiguator()
    >>> canonical = dis.disambiguate(raw_entities)
    >>> mapping   = dis.alias_to_canonical   # {"Taiwan Semiconductor": "TSMC", ...}
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Disambiguator] Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self.alias_to_canonical: dict[str, str] = {}
        self._canonical_entities: dict[str, Entity] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def disambiguate(self, entities: list[Entity]) -> list[Entity]:
        """
        Main entry point. Takes a list of (possibly duplicated) Entity objects
        and returns a deduplicated list with merged aliases.

        Parameters
        ----------
        entities : list[Entity]
            Raw entities as extracted from text.

        Returns
        -------
        list[Entity]
            Deduplicated list; each item has a canonical name and all known
            aliases attached.
        """
        self._canonical_entities = {}
        self.alias_to_canonical  = {}

        # Group by entity type — we never merge across types
        by_type: dict[EntityType, list[Entity]] = {}
        for ent in entities:
            by_type.setdefault(ent.entity_type, []).append(ent)

        for etype, group in by_type.items():
            self._disambiguate_group(group, etype)

        result = list(self._canonical_entities.values())
        print(
            f"[Disambiguator] {len(entities)} raw entities → "
            f"{len(result)} canonical entities"
        )
        return result

    def resolve(self, name: str) -> Optional[str]:
        """
        Given any surface form, return its canonical name if known.
        Returns None if the name was never seen during disambiguation.
        """
        norm = _normalize(name)
        # Check direct alias map
        if norm in self.alias_to_canonical:
            return self.alias_to_canonical[norm]
        # Check if it is itself a canonical name
        if name in self._canonical_entities:
            return name
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disambiguate_group(
        self,
        entities: list[Entity],
        etype: EntityType,
    ) -> None:
        """
        Disambiguate a list of entities of the same type.
        Uses two-pass strategy:
          Pass 1 — exact string match after normalization
          Pass 2 — embedding cosine similarity for near-matches
        """
        # Sort longer names first so we prefer more specific canonical names
        entities = sorted(entities, key=lambda e: len(e.name), reverse=True)

        # --- Pass 1: string normalization match ---------------------------
        norm_to_canonical: dict[str, str] = {}
        first_pass_groups: dict[str, Entity] = {}

        for ent in entities:
            norm = _normalize(ent.name)
            if norm in norm_to_canonical:
                # Merge into existing canonical
                canon_name = norm_to_canonical[norm]
                first_pass_groups[canon_name].aliases.append(ent.name)
                self.alias_to_canonical[norm] = canon_name
            else:
                norm_to_canonical[norm] = ent.name
                first_pass_groups[ent.name] = ent
                self.alias_to_canonical[norm] = ent.name

        remaining = list(first_pass_groups.values())

        # --- Pass 2: embedding similarity ---------------------------------
        if len(remaining) <= 1:
            for ent in remaining:
                self._register_canonical(ent)
            return

        names      = [e.name for e in remaining]
        embeddings = self._model.encode(names, convert_to_numpy=True)

        merged   = [False] * len(remaining)
        clusters: list[tuple[Entity, np.ndarray]] = []

        for i, (ent_i, emb_i) in enumerate(zip(remaining, embeddings)):
            if merged[i]:
                continue
            cluster_rep = ent_i
            for j in range(i + 1, len(remaining)):
                if merged[j]:
                    continue
                sim = _cosine_similarity(emb_i, embeddings[j])
                if sim >= SIMILARITY_THRESH:
                    # Merge j into i's cluster
                    cluster_rep.aliases.append(remaining[j].name)
                    cluster_rep.aliases.extend(remaining[j].aliases)
                    self.alias_to_canonical[_normalize(remaining[j].name)] = (
                        cluster_rep.name
                    )
                    merged[j] = True
            clusters.append((cluster_rep, emb_i))

        for ent, _ in clusters:
            self._register_canonical(ent)

    def _register_canonical(self, ent: Entity) -> None:
        """Store a canonical entity and register all its aliases in the lookup."""
        self._canonical_entities[ent.name] = ent
        self.alias_to_canonical[_normalize(ent.name)] = ent.name
        for alias in ent.aliases:
            self.alias_to_canonical[_normalize(alias)] = ent.name
