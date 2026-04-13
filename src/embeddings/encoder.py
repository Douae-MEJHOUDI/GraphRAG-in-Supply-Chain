"""
encoder.py
----------
NodeEncoder — generates and manages vector embeddings for graph nodes.

Every node in the supply chain knowledge graph gets a dense vector
representation using sentence-transformers. These embeddings are:

  - Stored as node attributes on the NetworkX graph (NODE_ATTR_EMBEDDING)
  - Indexed in a FAISS flat index for fast approximate nearest-neighbour search
  - Used in Module 2 for semantic seed-node retrieval (query → top-K nodes)

The encoder is intentionally decoupled from the graph builder so it can be
re-run independently (e.g. after fine-tuning, after adding new nodes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer

from src.graph.schema import (
    NODE_ATTR_EMBEDDING,
    NODE_ATTR_TYPE,
    NODE_ATTR_METADATA,
    EDGE_ATTR_SOURCE,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "BAAI/bge-large-en-v1.5"  # 1024-dim, optimised for retrieval
FAISS_INDEX_TYPE = "flat_ip"                  # inner-product (cosine on L2-normed vecs)
BATCH_SIZE       = 64                         # sentence-transformer encoding batch size


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------

class NodeEncoder:
    """
    Encodes graph nodes as dense embeddings and builds a FAISS index.

    The text used to represent each node for embedding is a rich description:
      "<entity_type>: <node_name>. Evidence: <source sentences from edges>"
    This gives the model enough context to distinguish "Supplier: TSMC"
    from "Region: Taiwan" and captures relational context from the graph.

    Usage
    -----
    >>> encoder = NodeEncoder()
    >>> encoder.encode_graph(G)                           # adds embeddings to G
    >>> encoder.save_index("data/processed/faiss.bin")   # persist index
    >>> top_nodes = encoder.search("semiconductor disruption risk", k=5)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"[Encoder] Loading model: {model_name}")
        self._model      = SentenceTransformer(model_name)
        self.dim         = self._model.get_sentence_embedding_dimension()
        self._index:     Optional[faiss.IndexFlatIP] = None
        self._node_names: list[str] = []   # maps FAISS position → node name
        self._graph:     Optional[nx.DiGraph] = None

    # ------------------------------------------------------------------
    # Core — encode all nodes in a graph
    # ------------------------------------------------------------------

    def encode_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        """
        Generate embeddings for every node in G and attach them as
        NODE_ATTR_EMBEDDING attributes. Also builds the FAISS index.

        Parameters
        ----------
        G : nx.DiGraph
            The supply chain knowledge graph from KnowledgeGraphBuilder.

        Returns
        -------
        nx.DiGraph
            The same graph with embeddings attached to each node.
        """
        self._graph = G
        node_names  = list(G.nodes())

        if not node_names:
            print("[Encoder] Graph has no nodes — nothing to encode.")
            return G

        # Build text representations for each node
        texts = [self._node_to_text(name, G.nodes[name]) for name in node_names]

        print(f"[Encoder] Encoding {len(node_names)} nodes in batches of {BATCH_SIZE}...")
        embeddings = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # normalise for cosine similarity
        )

        # Attach embeddings as node attributes
        for name, emb in zip(node_names, embeddings):
            G.nodes[name][NODE_ATTR_EMBEDDING] = emb

        # Build FAISS index
        self._build_index(node_names, embeddings)

        print(f"[Encoder] Done. Embedding dim: {self.dim}, "
              f"FAISS index size: {self._index.ntotal}")
        return G

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def search(
        self, query: str, k: int = 5, G: Optional[nx.DiGraph] = None
    ) -> list[tuple[str, float]]:
        """
        Find the top-K graph nodes most semantically similar to a query string.

        Parameters
        ----------
        query : str
            Natural language query (e.g. "semiconductor suppliers in Taiwan").
        k     : int
            Number of results to return.
        G     : nx.DiGraph, optional
            If provided, used for fallback node lookup. Otherwise uses the
            graph passed to encode_graph().

        Returns
        -------
        list of (node_name, similarity_score)
            Sorted descending by similarity. Scores are in [0, 1] (cosine).
        """
        if self._index is None:
            raise RuntimeError("FAISS index is empty. Call encode_graph() first.")

        query_emb = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # IndexFlatIP returns inner-product scores directly.
        # Since vectors are L2-normalised, inner product == cosine similarity.
        ip_scores, indices = self._index.search(query_emb, k)
        results = []
        for score, idx in zip(ip_scores[0], indices[0]):
            if idx < 0 or idx >= len(self._node_names):
                continue
            name = self._node_names[idx]
            results.append((name, round(float(score), 4)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Return the normalised embedding vector for a query string.
        Useful when you want to compute similarities manually.
        """
        return self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_index(self, path: str | Path) -> None:
        """
        Save the FAISS index and node name list to disk.
        Saves two files:
          <path>          — FAISS binary index
          <path>.names    — JSON list mapping FAISS position → node name
        """
        if self._index is None:
            raise RuntimeError("No index to save. Call encode_graph() first.")
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        names_path = path.with_suffix(path.suffix + ".names")
        names_path.write_text(
            json.dumps(self._node_names, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Encoder] Index saved → {path}")
        print(f"[Encoder] Node names saved → {names_path}")

    def load_index(self, path: str | Path) -> None:
        """
        Load a previously saved FAISS index and node name list.
        After loading, search() is immediately usable.
        """
        import json
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}")
        self._index      = faiss.read_index(str(path))
        names_path       = path.with_suffix(path.suffix + ".names")
        self._node_names = json.loads(names_path.read_text(encoding="utf-8"))
        print(f"[Encoder] Loaded index: {self._index.ntotal} vectors from {path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_to_text(self, name: str, attrs: dict) -> str:
        """
        Build the rich description used to embed a node.
        Format: "<EntityType>: <name>. Evidence: <source sentences from edges>"

        Spec: Name + Type + Evidence.
        Evidence is collected from the source_text of up to 2 connected edges,
        giving the model relational context beyond the bare node name.
        """
        etype = attrs.get(NODE_ATTR_TYPE, "Entity")

        # Collect up to 2 unique evidence sentences from outgoing edges
        evidence_parts: list[str] = []
        if self._graph is not None and name in self._graph:
            seen: set[str] = set()
            for _, _, edata in self._graph.out_edges(name, data=True):
                src = edata.get(EDGE_ATTR_SOURCE, "")
                if src and src not in seen:
                    seen.add(src)
                    evidence_parts.append(src)
                    if len(evidence_parts) >= 2:
                        break

        if evidence_parts:
            return f"{etype}: {name}. Evidence: {' '.join(evidence_parts)}"
        return f"{etype}: {name}"

    def _build_index(
        self, node_names: list[str], embeddings: np.ndarray
    ) -> None:
        """
        Build a FAISS IndexFlatIP index from a matrix of L2-normalised embeddings.

        IndexFlatIP (inner product) on L2-normalised vectors is mathematically
        equivalent to cosine similarity search. Exact (no approximation) and
        suitable for graphs up to ~100k nodes.
        """
        emb_matrix = embeddings.astype(np.float32)
        # Ensure vectors are unit-normalised (encode() already does this with
        # normalize_embeddings=True, but we re-normalise here as a safety measure)
        faiss.normalize_L2(emb_matrix)
        index = faiss.IndexFlatIP(self.dim)
        index.add(emb_matrix)
        self._index      = index
        self._node_names = node_names
