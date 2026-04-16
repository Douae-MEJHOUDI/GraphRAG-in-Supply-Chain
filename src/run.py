"""
run.py
------
Top-level entry point for the Supply Chain Risk Intelligence pipeline.

Wires together all modules in the correct order and exposes a single
`build_pipeline()` factory function that returns a ready-to-use CachedPipeline.

Pipeline architecture:

    Query
      ↓
    SemanticCache.lookup()
      ├─ HIT  → return cached Risk Report
      └─ MISS ↓
            SimulationEngine.run()
              ├─ DisruptionPropagator  (Model 3: Attenuated Bottleneck Routing)
              ├─ GraphRAGPipeline      (BGE seed retrieval + subgraph expansion)
              └─ RiskReportGenerator   (Claude Sonnet 4.6)
              ↓
            SemanticCache.store()
              ↓
            return Risk Report

Usage
-----
    from src.run import build_pipeline
    from src.simulation.events import SCENARIO_LIBRARY

    pipeline = build_pipeline()
    report   = pipeline.run(
        event=SCENARIO_LIBRARY["taiwan_earthquake"],
        query="What if Taiwan has a conflict?",
    )
    print(report["risk_report"])
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import networkx as nx

from src.cache.semantic_cache   import CachedPipeline, SemanticCache
from src.embeddings.encoder     import NodeEncoder, DEFAULT_MODEL
from src.generation.generator   import RiskReportGenerator
from src.retrieval.pipeline     import GraphRAGPipeline
from src.simulation.engine      import SimulationEngine


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

ROOT             = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH    = ROOT / "data/processed/sp500_graph_knowledge_graph_qwen25.gpickle"
DEFAULT_FAISS    = ROOT / "data/processed/faiss_index_bge.bin"
DEFAULT_COMMUNITIES = ROOT / "data/processed/communities.json"
DEFAULT_CACHE_DIR   = ROOT / "data/processed/cache"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_pipeline(
    graph_path:       Path = DEFAULT_GRAPH,
    faiss_path:       Path = DEFAULT_FAISS,
    communities_path: Path = DEFAULT_COMMUNITIES,
    cache_dir:        Path = DEFAULT_CACHE_DIR,
    cache_threshold:  float = 0.92,
    decay:            float = 0.85,
    max_hops:         int   = 5,
    seed_k:           int   = 5,
    hop_radius:       int   = 2,
    max_nodes:        int   = 40,
    llm_backend:      str   = "anthropic",
    llm_model:        str   = "claude-sonnet-4-6",
) -> CachedPipeline:
    """
    Build and return a fully wired CachedPipeline.

    Steps
    -----
    1. Load the knowledge graph (.gpickle).
    2. Load BGE encoder + FAISS index.
       If the BGE index does not exist yet, run reindex_bge.py first:
         python -m src.embeddings.reindex_bge
    3. Build the GraphRAGPipeline (loads pre-computed communities if available).
    4. Build the SimulationEngine (Attenuated Bottleneck Routing, γ=decay).
    5. Build the RiskReportGenerator (calls Claude to produce the report).
    6. Wrap with SemanticCache → CachedPipeline.

    Parameters
    ----------
    graph_path       : path to the .gpickle knowledge graph
    faiss_path       : path to the BGE FAISS index (faiss_index_bge.bin)
    communities_path : path to pre-computed communities JSON (optional)
    cache_dir        : directory for the semantic query cache
    cache_threshold  : cosine similarity threshold for cache hits (default 0.92)
    decay            : γ for Attenuated Bottleneck Routing (default 0.85)
    max_hops         : maximum propagation depth (default 5)
    seed_k           : top-K FAISS seeds per query (default 5)
    hop_radius       : ego-graph expansion radius (default 2)
    max_nodes        : subgraph size cap (default 40)
    llm_backend      : LLM backend — "anthropic" | "openai" | "ollama"
    llm_model        : model name string passed to the backend

    Returns
    -------
    CachedPipeline
    """
    # 1 — Load graph
    print(f"[run] Loading graph: {graph_path}")
    with open(graph_path, "rb") as fh:
        G: nx.DiGraph = pickle.load(fh)
    print(f"[run] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2 — Load encoder + FAISS index
    encoder = NodeEncoder(model_name=DEFAULT_MODEL)
    if faiss_path.exists():
        encoder.load_index(faiss_path)
        encoder._graph = G  # attach graph for evidence-enriched node text
    else:
        print(
            f"[run] WARNING: BGE FAISS index not found at {faiss_path}.\n"
            f"       Run:  python -m src.embeddings.reindex_bge\n"
            f"       Falling back to in-memory encoding (slow)."
        )
        encoder.encode_graph(G)

    # 3 — GraphRAG pipeline
    rag_pipeline = GraphRAGPipeline(
        graph=G,
        encoder=encoder,
        seed_k=seed_k,
        hop_radius=hop_radius,
        max_nodes=max_nodes,
    )
    if communities_path.exists():
        rag_pipeline.load_communities(str(communities_path))
    else:
        print("[run] Communities file not found — building (this may take a while)...")
        rag_pipeline.build_communities(save_path=str(communities_path))

    # 4 — Simulation engine (Attenuated Bottleneck Routing)
    engine = SimulationEngine(
        graph=G,
        pipeline=rag_pipeline,
        decay=decay,
        max_hops=max_hops,
    )

    # 5 — Risk report generator (calls Claude / OpenAI / Ollama)
    generator = RiskReportGenerator(backend=llm_backend, model=llm_model)

    # 6 — Semantic cache
    cache = SemanticCache(cache_dir=cache_dir, threshold=cache_threshold)

    print("[run] Pipeline ready.")
    return CachedPipeline(engine=engine, cache=cache, generator=generator)
