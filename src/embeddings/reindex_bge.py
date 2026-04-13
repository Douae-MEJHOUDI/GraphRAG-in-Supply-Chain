"""
reindex_bge.py
--------------
Rebuilds the FAISS index using BAAI/bge-large-en-v1.5 (1024-dim, IndexFlatIP).

Run after any change to the embedding model or after adding new nodes to the graph.

Usage
-----
    python -m src.embeddings.reindex_bge [--graph PATH] [--index-out PATH]

Defaults
--------
    graph     : data/processed/sp500_graph_knowledge_graph_qwen25.gpickle
    index-out : data/processed/faiss_index_bge.bin
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.embeddings.encoder import NodeEncoder, DEFAULT_MODEL


DEFAULT_GRAPH = (
    Path(__file__).resolve().parents[2]
    / "data/processed/sp500_graph_knowledge_graph_qwen25.gpickle"
)
DEFAULT_INDEX = (
    Path(__file__).resolve().parents[2]
    / "data/processed/faiss_index_bge.bin"
)


def reindex(graph_path: Path, index_path: Path) -> None:
    print(f"[reindex_bge] Loading graph: {graph_path}")
    with open(graph_path, "rb") as fh:
        G = pickle.load(fh)
    print(f"[reindex_bge] Graph loaded — {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges")

    encoder = NodeEncoder(model_name=DEFAULT_MODEL)
    encoder.encode_graph(G)
    encoder.save_index(index_path)

    print(f"[reindex_bge] Done. Index written to {index_path}")
    print(f"[reindex_bge] Embedding dim: {encoder.dim}")
    print(f"[reindex_bge] Vectors in index: {encoder._index.ntotal}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex graph nodes with BGE.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--index-out", type=Path, default=DEFAULT_INDEX)
    args = parser.parse_args()

    reindex(graph_path=args.graph, index_path=args.index_out)


if __name__ == "__main__":
    main()
