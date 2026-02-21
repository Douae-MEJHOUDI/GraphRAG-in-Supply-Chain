# SupplyGraph: GraphRAG for Supply Chain Disruption Analysis

A Graph Retrieval-Augmented Generation (GraphRAG) system that answers complex questions about supply chain disruptions by reasoning over a structured knowledge graph built from domain documents.

## Project Overview

This project implements a GraphRAG pipeline using **LightRAG** for the supply chain domain, focusing on logistics disruption analysis (port delays, supplier dependencies, cascading failures). It demonstrates the advantage of graph-based retrieval over standard RAG through four query modes.

## Architecture

```
Documents (txt/pdf)
       │
       ▼
  [ src/ingest.py ]
       │  Entity & Relation Extraction (via Groq LLM)
       ▼
  Knowledge Graph  ◄──── graph_storage/
       │
       ▼
  [ src/pipeline.py ] ── ask(question, mode=hybrid)
       │
       ▼
  [ app/chatbot.py ]  ── Streamlit UI
```

**Query modes (for evaluation):**
| Mode | Description |
|------|-------------|
| `naive` | Standard RAG — baseline (no graph) |
| `local` | Entity-level graph retrieval |
| `global` | Community/pattern-level retrieval |
| `hybrid` | Local + Global — best accuracy |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com) — no credit card needed.

```bash
cp .env.example .env
# Edit .env and paste your key
```

### 3. Add documents

Place `.txt` or `.pdf` supply chain documents in `data/raw/`. Sample documents are already included.

### 4. Build the knowledge graph

```bash
python -m src.ingest
```

This reads all documents, extracts entities and relationships, and stores the graph in `graph_storage/`. Run once; the graph persists.

### 5. Launch the chatbot

```bash
streamlit run app/chatbot.py
```

## Example Questions

- *Which suppliers were affected by the Suez Canal blockage?*
- *How did the semiconductor shortage cascade from automotive to consumer electronics?*
- *Which ports experienced congestion during COVID-19 and why?*
- *What risk mitigation strategies did companies adopt after 2021?*
- *Which companies depend on TSMC and what is their exposure?*

## Project Structure

```
GraphRAG-in-Supply-Chain/
├── data/
│   └── raw/                    # Input documents
│       ├── suez_canal_disruption.txt
│       ├── semiconductor_shortage.txt
│       └── port_congestion_covid.txt
├── graph_storage/              # LightRAG graph (auto-generated)
├── src/
│   ├── llm_config.py           # Groq LLM + sentence-transformer embeddings
│   ├── pipeline.py             # LightRAG init, insert, query
│   └── ingest.py               # Batch document ingestion script
├── app/
│   └── chatbot.py              # Streamlit chatbot UI
├── notebooks/                  # Graph exploration (coming soon)
├── .env.example
├── requirements.txt
└── README.md
```

## Compute Requirements

- **CPU only** — all components run without a GPU
- Embedding model: `all-MiniLM-L6-v2` (~90 MB, CPU-friendly)
- LLM inference: handled by Groq API (remote, free tier)
- Works on: local Mac (MPS/CPU), Google Colab, university cluster (CPU)
