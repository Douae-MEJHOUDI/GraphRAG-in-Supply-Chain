# GraphRAG for Supply Chain Risk Intelligence
### Knowledge Graphs · Attenuated Bottleneck Routing · GraphRAG · LLM Risk Reporting

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [Architecture Overview](#3-architecture-overview)
4. [GenAI Concepts Coverage](#4-genai-concepts-coverage)
5. [Repository Structure](#5-repository-structure)
6. [Module Roadmap](#6-module-roadmap)
7. [Benchmark Approach](#7-benchmark-approach)
8. [Data Sources](#8-data-sources)
9. [Literature Foundation](#9-literature-foundation)
10. [Roadmap & Future Work](#10-roadmap--future-work)

---

## 1. Project Overview

This project builds an end-to-end **GraphRAG system for supply chain risk intelligence**, combining knowledge graph reasoning with large language model generation and a novel **disruption propagation engine** — Attenuated Bottleneck Routing — that simulates how shocks cascade through supplier dependency networks.

Given a disruption event — a port closure, a geopolitical sanction, a factory fire — the system:

1. Propagates the disruption score through the dependency graph using a modified Dijkstra traversal with configurable per-hop decay
2. Retrieves the most relevant graph context using dual-stage retrieval: FAISS semantic seed search followed by K-hop ego-graph expansion
3. Merges retrieval relevance scores with disruption exposure scores into a unified priority signal
4. Generates a structured natural language risk report via Claude Sonnet 4.6, grounded in specific graph paths
5. Caches results semantically so equivalent future queries are served instantly

The pipeline is exposed through a FastAPI backend and is evaluated against a parallel benchmark approach (traditional multi-source RAG without graph reasoning) to demonstrate the measurable value of graph structure in supply chain risk analysis.

---

## 2. Research Motivation

### Why not plain RAG?

Standard Retrieval-Augmented Generation retrieves text chunks by semantic similarity. For supply chain risk this is fundamentally insufficient because:

- **Relationships matter as much as content.** Knowing that "TSMC is a semiconductor manufacturer in Taiwan" is far less actionable than knowing the full dependency chain: `Apple → TSMC → ASML (lithography machines) → Taiwan (earthquake risk zone)`.

- **Multi-hop reasoning is required.** A tier-3 supplier disruption cannot be surfaced by any amount of semantic similarity search — it requires traversing multiple edges in a dependency graph.

- **Global questions cannot be answered locally.** "What is our overall exposure to East Asian geopolitical risk?" requires reasoning over the entire corpus, not a retrieved chunk. Microsoft's GraphRAG paper (Edge et al., 2024) demonstrates that vanilla RAG fails completely on this class of query.

### Why Attenuated Bottleneck Routing?

Three propagation models were considered:

| Model | Formula | Problem |
|-------|---------|---------|
| Multiplicative decay | `score(v) = score(u) × decay × weight` | Fractions multiply across hops — deep failures vanish entirely |
| Simple bottleneck | `score(v) = min(parent_flow, bottleneck)` | No distance attenuation — false positives far downstream |
| **Attenuated Bottleneck (ours)** | `F_v = min(F_u, C_{u,v}) × γ` | Bottleneck awareness + guaranteed dissipation across hops |

Model 3 is implemented via a modified Dijkstra traversal on a max-heap (highest flow first), which guarantees that the first time a node is settled it holds the globally optimal path — analogous to Dijkstra's shortest-path guarantee but for maximum flow.

### Why add a benchmark comparison?

None of the five papers in our literature review compare GraphRAG against a classical multi-source RAG baseline on the same task. The `different_approach_for_benchmark` pipeline implements exactly that baseline — same data sources (SEC filings, USGS minerals, iFixit teardowns, GDELT events), same LLM, but without graph structure — allowing a controlled ablation that isolates the value of graph reasoning.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│   SEC 10-K/20-F · USGS Minerals · iFixit Teardowns · GDELT News    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 1 — KNOWLEDGE GRAPH CONSTRUCTION                │
│                                                                     │
│  LLM Extraction (Qwen 2.5:32b / Llama 3.3 via Groq)               │
│  → Running entity registry (cross-chunk canonicalization)           │
│  → LLM canonicalization pass (alias deduplication)                 │
│  → NetworkX DiGraph · Edge weight assignment                        │
│  → BGE-large-en-v1.5 node embeddings → FAISS IndexFlatIP           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 2 — GRAPHRAG RETRIEVAL LAYER                    │
│                                                                     │
│  Query → BGE embedding → FAISS seed search (top-K nodes)           │
│  → K-hop ego-graph expansion (undirected traversal)                 │
│  → Score fusion: 0.6 × semantic + 0.4 × degree centrality          │
│  + Louvain community detection → BART community summaries           │
│  + Query type routing: local (subgraph only) / global (+summaries)  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│         MODULE 3 — DISRUPTION SIMULATION ENGINE  ◀── NOVEL          │
│                                                                     │
│  DisruptionEvent (ground_zero nodes + initial_shock)                │
│  → Ground-zero resolution (semantic fallback + location expansion)  │
│  → Attenuated Bottleneck Routing: F_v = min(F_u, C_{u,v}) × γ     │
│     Modified Dijkstra on max-heap — optimal path guarantee          │
│  → Score merging: 0.5 × retrieval + 0.5 × disruption               │
│  → Enriched context with path traces, severity tiers, critical edges│
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 4 — LLM GENERATION & RISK REPORTING             │
│                                                                     │
│  Enriched context → Claude Sonnet 4.6 (primary)                     │
│  Structured output: 6 sections — Executive Summary · Exposed        │
│  Entities · Cascade Chains · Critical Dependencies · Mitigations ·  │
│  Resilience Assessment                                              │
│  + Semantic query cache (BGE + FAISS, threshold 0.92)              │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 5 — EVALUATION & ABLATION STUDY                 │
│                                                                     │
│  GraphRAG+Sim  vs.  Vector RAG  vs.  Baseline LLM                   │
│  vs.  Benchmark approach (classical multi-source RAG)               │
│  Metrics: faithfulness · multi-hop accuracy · hallucination rate    │
│           disruption coverage · answer completeness                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. GenAI Concepts Coverage

| Course Concept | Where it appears in this project |
|---|---|
| **Embeddings** | BGE-large-en-v1.5 (1024-dim) for node and query encoding; used for FAISS semantic search and semantic cache |
| **Transformer architecture** | Foundation of every model used (BGE, BART, Claude, Qwen) |
| **Encoder-only models** | BGE encoder for node/query embeddings (Module 1, 2, cache) |
| **Decoder-only models** | Claude Sonnet 4.6 for risk report generation (Module 4); Qwen 2.5:32b for LLM graph extraction (Module 1) |
| **Encoder-decoder models** | BART for community summary generation (Module 2) |
| **RAG** | Benchmark baseline pipeline (`different_approach_for_benchmark`) — same data, no graph structure |
| **GraphRAG** | Core retrieval architecture combining semantic search with graph traversal (Modules 2–4) |
| **Prompt engineering** | System prompts with explicit output schemas; chain-of-thought context enrichment; few-shot canonicalization |
| **Semantic caching** | FAISS-backed query cache at similarity threshold 0.92 — skips LLM on semantically equivalent queries |

---

## 5. Repository Structure

```
GraphRAG-in-Supply-Chain/
│
├── README.md
│
├── src/                                  ← Main GraphRAG pipeline
│   ├── run.py                            ← Pipeline factory (build_pipeline)
│   ├── graph/
│   │   ├── schema.py                     ← Ontology: EntityType, RelationType
│   │   ├── builder.py                    ← Rule-based graph builder (spaCy + regex)
│   │   ├── llm_builder.py               ← LLM graph builder (Qwen/Llama — production)
│   │   └── disambiguator.py             ← Embedding-based entity disambiguation
│   ├── embeddings/
│   │   ├── encoder.py                    ← NodeEncoder (BGE + FAISS)
│   │   └── reindex_bge.py               ← CLI: rebuild FAISS index
│   ├── retrieval/
│   │   ├── retriever.py                  ← SubgraphRetriever (FAISS + K-hop)
│   │   ├── community.py                  ← Louvain + BART community summarizer
│   │   └── pipeline.py                  ← GraphRAGPipeline (local/global routing)
│   ├── simulation/
│   │   ├── events.py                     ← DisruptionEvent, SCENARIO_LIBRARY
│   │   ├── propagator.py                ← Attenuated Bottleneck Routing engine
│   │   └── engine.py                    ← SimulationEngine (orchestrator)
│   ├── generation/
│   │   └── generator.py                  ← RiskReportGenerator (Claude / OpenAI / Ollama)
│   ├── cache/
│   │   └── semantic_cache.py            ← SemanticCache + CachedPipeline
│   ├── api/
│   │   └── app.py                        ← FastAPI backend
│   └── utils/
│       └── visualization.py             ← Graph rendering (matplotlib + pyvis)
│
├── different_approach_for_benchmark/    ← Classical RAG baseline (no graph)
│   ├── m1_data_collection/              ← SEC, USGS, iFixit, GDELT connectors
│   ├── m2_data_processing/              ← Per-source text processors
│   ├── m3_entity_resolution/           ← String + embedding alias resolution
│   ├── m4_graph_construction/          ← Flat document store (no graph reasoning)
│   ├── m6_rag_retrieval/               ← Vector-only retrieval pipeline
│   ├── m7_report_generation/           ← Same LLM, no graph context
│   ├── m8_api/                          ← FastAPI backend for benchmark
│   ├── m9_ui/                           ← Frontend UI
│   └── m_eval/                          ← Shared evaluation + benchmark harness
│
├── data/
│   ├── raw/                             ← Source documents
│   └── processed/
│       ├── sp500_graph_knowledge_graph_qwen25.gpickle   ← Production graph
│       ├── faiss_index_bge.bin          ← BGE node embeddings index
│       ├── communities.json             ← Pre-computed Louvain communities
│       └── cache/                       ← Semantic query cache
│
├── notebooks/                           ← One notebook per module
│
└── requirements.txt
```

---

## 6. Module Roadmap

### Module 1 — Knowledge Graph Construction

**Goal:** Transform raw supply chain documents into a structured, queryable knowledge graph where nodes are SC entities and directed edges are typed relationships with criticality weights.

**Pipeline (LLM-driven — production path):**
1. **Chunking** — Documents split into overlapping 2400-character chunks at sentence boundaries
2. **LLM extraction** — Qwen 2.5:32b (Ollama) or Llama 3.3-70b (Groq) extracts `(head, relation, tail, confidence, evidence)` triples per chunk
3. **Running entity registry** — After each chunk, newly seen entities are added to a shared context injected into the next chunk, steering the LLM toward canonical names
4. **LLM canonicalization** — A second LLM pass groups aliases and returns a canonical-name map; all triples are rewritten before graph construction
5. **Graph validation** — Self-loops removed, parallel edges merged (keep highest confidence), dangling edges cleaned
6. **Edge weight assignment** — `weight = 1 / (1 + n_alternatives)`: single-source dependencies carry full shock; nodes with alternatives get proportionally lower weights
7. **Node embedding** — Every node encoded with `BAAI/bge-large-en-v1.5` (1024-dim, L2-normalised) and stored in a FAISS `IndexFlatIP`

**Ontology:**

| Entity Types | Relation Types |
|---|---|
| Supplier, Manufacturer, Part | supplies, depends_on |
| Port, LogisticsRoute | ships_through, located_in |
| Region, Customer | sells_to, alternative_to |
| DisruptionEvent | affected_by |

---

### Module 2 — GraphRAG Retrieval Layer

**Goal:** Given a natural language query, retrieve a semantically and relationally relevant subgraph that captures both query-relevant entities and their multi-hop dependency context.

**Dual-stage retrieval:**
1. **Semantic seed search** — Query embedded with BGE; FAISS top-K nodes above similarity threshold 0.20 become seed nodes
2. **K-hop ego-graph expansion** — Each seed expanded 2 hops on the undirected graph, pulling in dependency context invisible to vector search alone
3. **Score fusion** — `node_score = 0.6 × semantic_similarity + 0.4 × degree_centrality`
4. **Subgraph trimming** — Top 40 nodes by score kept; induced subgraph extracted

**Global query support:**
- Query type classified as `local` or `global` via cosine similarity to prototype embeddings
- Global queries prepend pre-computed Louvain community summaries (BART-generated) to the subgraph context

---

### Module 3 — Disruption Simulation Engine

**Goal:** Propagate a disruption event through the dependency graph and assign every reachable node a numeric exposure score. This is the novel contribution of the project.

**Algorithm — Attenuated Bottleneck Routing:**

```
F_v = min(F_u, C_{u,v}) × γ

where:
  F_u      = disruption score settled at parent node u
  C_{u,v}  = edge_weight × relation_type_factor
               relation_type_factor: supplies/depends_on → 1.0
                                     ships_through → 0.70
                                     located_in → 0.50
  γ (gamma) = global decay per hop (default 0.85)
```

Traversal uses a max-heap (modified Dijkstra's) ensuring every node is first settled via the globally highest-flow path. Propagation stops at depth 5 or below score threshold 0.03.

**Score merging with Module 2:**
```
merged_score = 0.5 × retrieval_score + 0.5 × disruption_score
```
Nodes that are both semantically relevant to the query AND highly exposed to the disruption are prioritised first in the LLM context.

**Severity tiers:** Critical ≥ 0.80 · High ≥ 0.50 · Moderate ≥ 0.25 · Low < 0.25

**Pre-built scenarios:** Taiwan Earthquake · Shanghai Port Closure · Congo Cobalt Strike · Red Sea Disruption · ASML Export Restriction

---

### Module 4 — LLM Generation & Risk Reporting

**Goal:** Produce a structured, graph-grounded risk report from the enriched simulation context.

**Generation pipeline:**
1. Enriched context serialised with disruption scores, severity badges, path traces, and critical edge flags injected per node
2. Claude Sonnet 4.6 called with a system prompt defining the analyst role and exact 6-section output format
3. Output parsed into structured fields: Executive Summary · Critically Exposed Entities · Cascade Chains · Single-Source Dependencies · Mitigations · Resilience Assessment

**Semantic query cache:**
- Every query embedded with BGE and compared against a FAISS `IndexFlatIP` of past queries
- Cosine similarity ≥ 0.92 → return cached report, skip LLM entirely
- Cache persisted to disk; survives server restarts

**Secondary backends:** OpenAI GPT-4o-mini · Ollama (any local model)

---

### Module 5 — Evaluation & Ablation Study

**Goal:** Quantify the contribution of each architectural component through controlled comparison.

| System | Description |
|--------|-------------|
| Baseline | Plain Claude with no retrieval |
| Vector RAG | BGE + FAISS semantic retrieval, no graph structure |
| GraphRAG | Module 2 retrieval, no disruption simulation |
| **GraphRAG + Simulation** | Full pipeline (this project) |
| **Benchmark approach** | Classical multi-source RAG (separate pipeline) |

**Metrics:** Multi-hop accuracy · Answer faithfulness · Hallucination rate · Disruption coverage · Answer completeness

---

## 7. Benchmark Approach

The `different_approach_for_benchmark/` directory contains a full parallel pipeline implementing classical multi-source RAG — the same data, the same LLM, but without graph reasoning — to serve as a controlled baseline for evaluation.

**Data sources ingested:** SEC 10-K/20-F filings · USGS Mineral Commodity Summaries · iFixit teardown guides · GDELT geopolitical event database

**Pipeline stages:**
- `m1_data_collection` — Source-specific connectors for each data provider
- `m2_data_processing` — Per-source text cleaning and schema normalisation
- `m3_entity_resolution` — String normalisation + embedding-based alias resolution (no graph built)
- `m4_graph_construction` — Flat document store; triples stored but no graph traversal used at query time
- `m6_rag_retrieval` — Vector-only retrieval from a flat FAISS index
- `m7_report_generation` — Same Claude prompting as Module 4 but on vector-retrieved context
- `m8_api` + `m9_ui` — FastAPI backend and frontend UI mirroring the main pipeline
- `m_eval` — Shared evaluation harness comparing both pipelines on identical question sets

The benchmark serves one purpose: demonstrate that graph structure, not just data richness, is what enables accurate multi-hop supply chain risk reasoning.

---

## 8. Data Sources

| Dataset | Description |
|---------|-------------|
| **SEC EDGAR** | 10-K / 20-F filings — supplier relationships, geographic concentrations, risk factors |
| **USGS Mineral Commodity Summaries 2024** | Country-level mineral production and reserve data |
| **iFixit Teardown Guides** | Product-level component and supplier dependencies |
| **GDELT Project** | Global geopolitical event database with Goldstein conflict scale scores |
| **S&P 500 corpus** | Filtered supply chain passages used to build the production Qwen-extracted graph |

---

## 9. Literature Foundation

| Paper | Key contribution to this project |
|-------|----------------------------------|
| Edge et al. (2024) — Microsoft GraphRAG | Community summarization architecture; local vs. global query routing |
| Peng et al. (2024) — GraphRAG Survey | G-Indexing / G-Retrieval / G-Generation taxonomy |
| Almahri et al. (2024) — SC KG + LLMs | LLM-driven NER+RE prompting strategy for graph construction |
| Kosasih et al. (2024) — Neurosymbolic SC Risk | Explainability requirement; edge criticality weighting concept |
| Wasi et al. (2024) — GNN SC Benchmarks | Task taxonomy, evaluation framing, and SupplyGraph dataset |
| Pressure Wave Propagation (MDPI, 2026) | Theoretical basis for the Attenuated Bottleneck Routing model |

---

## 10. Roadmap & Future Work

- **Fine-tuned NER** — Replace zero-shot LLM extraction with a BERT encoder fine-tuned on SC-specific entity labels; measure graph quality improvement
- **Temporal edges** — Add timestamps to edges and model disruption decay over time
- **Multi-scenario comparison** — Run parallel disruption scenarios and compare exposure profiles across geopolitical risk events
- **Real-time news integration** — GDELT event stream → automatic graph updates when new SC disruption events are detected
- **Confidence propagation** — Propagate extraction confidence scores through the graph and surface uncertainty in the risk report

---

*GraphRAG · Knowledge Graphs · Attenuated Bottleneck Routing · Claude Sonnet 4.6 · BGE Embeddings · Louvain Communities · Semantic Caching*
