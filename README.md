# GraphRAG for Supply Chain Risk Intelligence
### A Generative AI Project — Knowledge Graphs · GraphRAG · Disruption Simulation

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Research Motivation](#2-research-motivation)
3. [Architecture Overview](#3-architecture-overview)
4. [GenAI Concepts Coverage](#4-genai-concepts-coverage)
5. [Repository Structure](#5-repository-structure)
6. [Module Roadmap](#6-module-roadmap)
   - [Module 1 — Knowledge Graph Construction](#module-1--knowledge-graph-construction)
   - [Module 2 — GraphRAG Retrieval Layer](#module-2--graphrag-retrieval-layer)
   - [Module 3 — Disruption Simulation Engine](#module-3--disruption-simulation-engine)
   - [Module 4 — LLM Generation & Risk Reporting](#module-4--llm-generation--risk-reporting)
   - [Module 5 — Evaluation & Ablation Study](#module-5--evaluation--ablation-study)
7. [Data Sources](#7-data-sources)
8. [Installation](#8-installation)
9. [Quickstart](#9-quickstart)
10. [Literature Foundation](#10-literature-foundation)
11. [Roadmap & Future Work](#11-roadmap--future-work)

---

## 1. Project Overview

This project builds an end-to-end **GraphRAG system for supply chain risk intelligence**,
combining knowledge graph reasoning with large language model generation and a novel
**disruption propagation engine** that simulates how shocks cascade through supplier
dependency networks.

Given a disruption event — a port closure, a geopolitical sanction, a factory fire —
the system:

1. Identifies all supply chain entities exposed to the event
2. Propagates the disruption score through the dependency graph with configurable decay
3. Retrieves the most relevant graph context using both vector similarity and graph traversal
4. Generates a structured natural language risk report, citing specific graph paths
5. Visualizes the affected subgraph with nodes color-coded by exposure severity

The result is a system that goes well beyond what any of the five foundational papers
achieves individually, combining their best ideas into one coherent, demo-able pipeline.

---

## 2. Research Motivation

### Why not plain RAG?

Standard Retrieval-Augmented Generation retrieves text chunks by semantic similarity.
For supply chain risk this is fundamentally insufficient because:

- **Relationships matter as much as content.** Knowing that "TSMC is a semiconductor
  manufacturer in Taiwan" is far less actionable than knowing the full dependency chain:
  `Apple → TSMC → ASML (lithography machines) → Taiwan (earthquake risk zone)`.

- **Multi-hop reasoning is required.** A tier-3 supplier disruption cannot be surfaced
  by any amount of semantic similarity search — it requires traversing 3 edges in a
  dependency graph.

- **Global questions cannot be answered locally.** "What is our overall exposure to
  East Asian geopolitical risk?" requires reasoning over the entire corpus, not a
  retrieved chunk. Microsoft's GraphRAG paper (Edge et al., 2024) shows vanilla RAG
  fails completely on this class of query.

### Why add disruption simulation?

None of the five papers in our literature review combine GraphRAG with forward-looking
disruption propagation. Almahri et al. build a knowledge graph but stop there. Kosasih
et al. do risk reasoning but with fixed symbolic rules, not generative output. Our
simulation engine makes the system *predictive*, not just *descriptive*.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│   SupplyGraph Dataset · News Articles · Synthetic SC Documents      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 1 — KNOWLEDGE GRAPH CONSTRUCTION                │
│                                                                     │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ NLP Pipeline│───▶│ Triple       │───▶│ Graph Store          │   │
│  │ (spaCy +    │    │ Extraction   │    │ (NetworkX / Neo4j)   │   │
│  │  LLM NER)   │    │ (entities +  │    │                      │   │
│  └─────────────┘    │  relations)  │    │ + Node Embeddings    │   │
│                     └──────────────┘    │   (FAISS / Chroma)   │   │
│                                         └──────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 2 — GRAPHRAG RETRIEVAL LAYER                    │
│                                                                     │
│  Query Embedding → Seed Node Search → K-hop Subgraph Traversal      │
│  + Louvain Community Detection → T5/BART Community Summarization    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│         MODULE 3 — DISRUPTION SIMULATION ENGINE  ◀── NOVEL          │
│                                                                     │
│  Disruption Node → Weighted BFS Propagation → Exposure Scoring      │
│  → Criticality Weighting (alternative supplier awareness)           │
│  → Color-coded subgraph visualization                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 4 — LLM GENERATION & RISK REPORTING             │
│                                                                     │
│  Structured Prompt (subgraph + scores) → Decoder-only LLM           │
│  → Risk Report with cited graph paths + mitigation suggestions      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              MODULE 5 — EVALUATION & ABLATION STUDY                 │
│                                                                     │
│  Baseline LLM vs. Vector RAG vs. GraphRAG+Sim                       │
│  Metrics: faithfulness · multi-hop accuracy · hallucination rate    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. GenAI Concepts Coverage

This table maps every concept from the Generative AI course to a concrete component
in the project, so the connection is explicit and defensible in the final presentation.

| Course Concept              | Where it appears in this project                          |
|-----------------------------|-----------------------------------------------------------|
| **Embeddings**              | Node vector representations (sentence-transformers);      |
|                             | used for semantic seed-node retrieval in Module 2         |
| **Transformer architecture**| Foundation of every model used (BERT, T5, GPT/Mistral)   |
| **Encoder-only models**     | NER & relation extraction (Module 1);                     |
|                             | query encoding for retrieval (Module 2)                   |
| **Decoder-only models**     | Risk report generation (Module 4);                        |
|                             | zero-shot triple extraction from raw text                 |
| **Encoder-decoder models**  | Community summary generation using T5/BART (Module 2)    |
| **Fine-tuning**             | Module B (next phase): BERT fine-tuned on SC-specific     |
|                             | NER labels for higher-quality graph construction          |
| **RAG**                     | Baseline system in ablation study (Module 5)             |
| **GraphRAG**                | Core retrieval architecture (Modules 2–4)                |

---

## 5. Repository Structure

```
graphrag_supply_chain/
│
├── README.md                        ← You are here
│
├── data/
│   ├── raw/                         ← Original, unprocessed source files
│   │   ├── supply_graph_raw.json    ← SupplyGraph dataset (Wasi et al.)
│   │   └── disruption_news.jsonl    ← News article corpus
│   └── processed/
│       ├── triples.json             ← Extracted (head, relation, tail) triples
│       ├── entities.json            ← Deduplicated entity registry
│       └── graph.gpickle            ← Serialized NetworkX graph
│
├── notebooks/
│   ├── 01_knowledge_graph_construction.ipynb   ← MODULE 1 (this file)
│   ├── 02_graphrag_retrieval.ipynb             ← MODULE 2
│   ├── 03_disruption_simulation.ipynb          ← MODULE 3
│   ├── 04_llm_generation.ipynb                 ← MODULE 4
│   └── 05_evaluation.ipynb                     ← MODULE 5
│
├── src/
│   ├── __init__.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py               ← KnowledgeGraphBuilder class
│   │   ├── schema.py                ← Entity types, edge types, ontology
│   │   └── disambiguator.py        ← Entity disambiguation logic
│   ├── embeddings/
│   │   ├── __init__.py
│   │   └── encoder.py               ← NodeEncoder class (sentence-transformers)
│   └── utils/
│       ├── __init__.py
│       ├── io.py                    ← File I/O helpers
│       └── visualization.py        ← Graph plotting utilities
│
├── outputs/
│   ├── graphs/                      ← Saved graph visualizations
│   └── reports/                     ← Generated risk reports
│
└── requirements.txt
```

---

## 6. Module Roadmap

### Module 1 — Knowledge Graph Construction

**Notebook:** `notebooks/01_knowledge_graph_construction.ipynb`

**Goal:** Transform raw supply chain text data into a structured, queryable knowledge
graph where nodes are SC entities and edges are typed relationships.

**Steps:**
1. **Data loading** — Load the SupplyGraph dataset and/or synthetic SC documents
2. **Entity extraction** — Use spaCy for baseline NER + LLM prompting (zero-shot)
   to identify SC-specific entities: `Supplier`, `Part`, `Port`, `Region`,
   `Manufacturer`, `LogisticsRoute`, `DisruptionEvent`
3. **Relation extraction** — Extract typed triples: `(head_entity, relation, tail_entity)`
   where relations include: `supplies`, `depends_on`, `ships_through`, `located_in`,
   `affected_by`, `alternative_to`
4. **Entity disambiguation** — Resolve aliases to canonical entities using string
   normalization and embedding cosine similarity
5. **Graph construction** — Load triples into a directed NetworkX `DiGraph` with
   node and edge attributes
6. **Node embedding** — Encode every node using `sentence-transformers` and store
   vectors in a FAISS index for later retrieval
7. **Validation & visualization** — Inspect graph statistics, spot-check extracted
   triples, render a sample subgraph

**Key outputs:**
- `data/processed/triples.json`
- `data/processed/entities.json`
- `data/processed/graph.gpickle`
- FAISS index persisted to `data/processed/faiss_index.bin`

**Source modules used:** `src/graph/schema.py`, `src/graph/builder.py`,
`src/graph/disambiguator.py`, `src/embeddings/encoder.py`

---

### Module 2 — GraphRAG Retrieval Layer

**Notebook:** `notebooks/02_graphrag_retrieval.ipynb`

**Goal:** Given a natural language query, retrieve a semantically and relationally
relevant subgraph to serve as LLM context.

**Steps:**
1. Encode query with sentence-transformer
2. Search FAISS index for top-K seed nodes
3. Expand seeds via K-hop ego graph traversal
4. Apply Louvain community detection on the full graph
5. Pre-generate community summaries with T5/BART (encoder-decoder tie-in)
6. Serialize retrieved subgraph as structured text for prompt injection

---

### Module 3 — Disruption Simulation Engine

**Notebook:** `notebooks/03_disruption_simulation.ipynb`

**Goal:** Propagate a disruption event through the dependency graph and score all
downstream entities by exposure level. This is the novel contribution of this project.

**Steps:**
1. Trigger a disruption on one or more graph nodes
2. Compute edge criticality weights (penalize nodes with no `alternative_to` edges)
3. Run weighted BFS propagation with configurable decay factor
4. Score all reachable nodes and rank by exposure severity
5. Filter the retrieved subgraph by exposure threshold
6. Visualize with color-coded severity (red → orange → yellow → gray)

---

### Module 4 — LLM Generation & Risk Reporting

**Notebook:** `notebooks/04_llm_generation.ipynb`

**Goal:** Combine the retrieved subgraph context with disruption scores into a
structured prompt, then generate a natural language risk report with an explicit
decoder-only LLM.

**Steps:**
1. Build prompt: system role + serialized subgraph + disruption scores + user query
2. Call generator LLM (GPT-4o-mini via API or Mistral-7B via Ollama locally)
3. Parse output: extract cited graph paths, at-risk entity list, mitigation suggestions
4. Format as a structured risk report

---

### Module 5 — Evaluation & Ablation Study

**Notebook:** `notebooks/05_evaluation.ipynb`

**Goal:** Rigorously compare three systems on a curated set of SC risk questions to
demonstrate the value of each architectural addition.

**Conditions:**
| System | Description |
|--------|-------------|
| Baseline | Plain LLM with no retrieval |
| Vector RAG | Semantic chunk retrieval only |
| **GraphRAG + Sim** | This project (full pipeline) |

**Metrics:**
- Multi-hop accuracy (questions requiring 3+ dependency hops)
- Answer faithfulness (does answer contradict the graph?)
- Hallucination rate (invented relationships not in graph)
- Disruption coverage (fraction of exposed nodes above threshold surfaced)

---

## 7. Data Sources

| Dataset | Description | Access |
|---------|-------------|--------|
| **SupplyGraph** (Wasi et al., 2024) | Real FMCG SC graph from Bangladesh company | [GitHub](https://github.com/ciol-researchlab/SupplyGraph) |
| **GDELT Project** | Global news event database, free | [gdeltproject.org](https://www.gdeltproject.org) |
| **Synthetic SC corpus** | LLM-generated supplier documents | Generated in Module 1 notebook |

For this project we use a **curated synthetic dataset** built directly in the notebook,
which guarantees clean ground-truth triples for evaluation while mirroring the structure
of the SupplyGraph data. Real datasets can be swapped in with no code changes.

---

## 8. Installation

```bash
# Clone the project
git clone https://github.com/your-username/graphrag-supply-chain.git
cd graphrag-supply-chain

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Download spaCy English model
python -m spacy download en_core_web_sm

# (Optional) For local LLM generation — install Ollama and pull Mistral
# https://ollama.ai/download
# ollama pull mistral
```

**requirements.txt** is located at the project root and pins all versions for
reproducibility.

---

## 9. Quickstart

```python
# Run the full Module 1 pipeline programmatically
from src.graph.builder import KnowledgeGraphBuilder
from src.graph.schema import ENTITY_TYPES, RELATION_TYPES
from src.embeddings.encoder import NodeEncoder

# 1. Build the graph from raw documents
builder = KnowledgeGraphBuilder()
builder.load_documents("data/raw/")
builder.extract_triples()
builder.disambiguate_entities()
G = builder.build_graph()

# 2. Encode and index all nodes
encoder = NodeEncoder()
encoder.encode_graph(G)
encoder.save_index("data/processed/faiss_index.bin")

# 3. Inspect
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")
```

---

## 10. Literature Foundation

| Paper | Key contribution to this project |
|-------|----------------------------------|
| Edge et al. (2024) — Microsoft GraphRAG | Community summarization architecture; GraphRAG pipeline design |
| Peng et al. (2024) — GraphRAG Survey | G-Indexing / G-Retrieval / G-Generation taxonomy |
| Almahri et al. (2024) — SC KG + LLMs | Zero-shot NER+RE prompting strategy for graph construction |
| Kosasih et al. (2024) — Neurosymbolic SC Risk | Explainability requirement; edge criticality weighting concept |
| Wasi et al. (2024) — GNN SC Benchmarks | SupplyGraph dataset; task taxonomy and evaluation framing |

---

## 11. Roadmap & Future Work

- **Module B (next phase):** Fine-tune a BERT encoder on SC-specific NER labels
  to replace zero-shot extraction, and measure graph quality improvement
- **Temporal reasoning:** Add timestamps to edges and model disruption decay over time
- **Multi-scenario simulation:** Run parallel disruption scenarios and compare
  exposure profiles across different geopolitical risk events
- **Interactive dashboard:** Streamlit or Gradio frontend for the full pipeline
- **Real-time news integration:** GDELT event stream → automatic graph updates
  when new SC disruption events are detected

---

*Built for the Generative AI course project — combining GraphRAG, knowledge graphs,
encoder/decoder models, embeddings, and fine-tuning into a unified supply chain
risk intelligence system.*
