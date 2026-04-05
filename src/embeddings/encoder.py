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

from src.graph.schema import NODE_ATTR_EMBEDDING, NODE_ATTR_TYPE, NODE_ATTR_METADATA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL    = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality
FAISS_INDEX_TYPE = "flat_l2"            # exact L2 search — swap for IVF at scale
BATCH_SIZE       = 64                   # sentence-transformer encoding batch size


# ---------------------------------------------------------------------------
# NodeEncoder
# ---------------------------------------------------------------------------

class NodeEncoder:
    """
    Encodes graph nodes as dense embeddings and builds a FAISS index.

    The text used to represent each node for embedding is:
      "<entity_type>: <node_name>"
    This gives the model enough context to distinguish "Supplier: TSMC"
    from "Region: Taiwan" even though both contain "Taiwan".

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
        self._index:     Optional[faiss.IndexFlatL2] = None
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

        # FAISS returns L2 distances — convert to cosine similarity
        # (valid because embeddings are L2-normalised: cos_sim = 1 - dist²/2)
        distances, indices = self._index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._node_names):
                continue
            name    = self._node_names[idx]
            sim     = float(1.0 - dist / 2.0)   # convert L2 → cosine
            results.append((name, round(sim, 4)))

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
        Build the text string used to embed a node.
        Format: "<EntityType>: <name> [<metadata key=value pairs>]"
        The entity type prefix is crucial — it disambiguates "Taiwan" as a
        Region from "Taiwan Semiconductor" as a Supplier.
        """
        etype    = attrs.get(NODE_ATTR_TYPE, "Entity")
        metadata = attrs.get(NODE_ATTR_METADATA, {})

        # Include a few high-signal metadata fields if available
        meta_str = ""
        for key in ("country", "capacity", "category", "product"):
            if key in metadata:
                meta_str += f" {key}={metadata[key]}"

        return f"{etype}: {name}{meta_str}".strip()

    def _build_index(
        self, node_names: list[str], embeddings: np.ndarray
    ) -> None:
        """
        Build a FAISS flat L2 index from a matrix of embeddings.
        Flat L2 is exact (no approximation) and suitable for graphs
        up to ~100k nodes. For larger graphs, switch to IndexIVFFlat.
        """
        emb_matrix = embeddings.astype(np.float32)
        index = faiss.IndexFlatL2(self.dim)
        index.add(emb_matrix)
        self._index      = index
        self._node_names = node_names
