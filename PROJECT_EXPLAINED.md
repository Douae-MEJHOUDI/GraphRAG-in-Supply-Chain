# What We Are Building — GraphRAG for Supply Chain Risk Intelligence

A complete, honest explanation of every layer of this project.

---

## The Core Problem

A supply chain is a network of dependencies. Apple depends on TSMC for chips. TSMC depends on ASML for lithography machines. ASML uses components from Germany. If a conflict breaks out in Taiwan, the question is not just "is TSMC affected?" but "who else is affected, through how many hops, and how badly?"

Standard AI tools cannot answer this. A plain LLM trained on public data has no live picture of your supply chain. A standard search engine finds documents about the disruption but cannot reason through chains of dependency. Our project builds a system that can.

---

## What the Project Does in One Paragraph

We take raw text documents about supply chains (company filings, news articles, synthetic reports), extract named entities and the relationships between them, store those as a structured graph in memory, then combine three things to answer risk questions: (1) graph traversal to follow multi-hop dependency chains, (2) disruption simulation to score how badly each node is affected, and (3) a large language model that reads the retrieved graph context and writes a structured risk report. This combination is called **GraphRAG** — Graph-augmented Retrieval-Augmented Generation.

---

## The Problem With Plain RAG (Why We Need the Graph)

Standard RAG works like this:

```
User query → embed query as vector → find most similar text chunks → give to LLM
```

For supply chain risk this fails in three specific ways:

**1. Relationships matter, not just content.**
A document saying "TSMC is a Taiwanese chip manufacturer" is useful background, but what we actually need is the fact that `Apple DEPENDS_ON TSMC` and `TSMC LOCATED_IN Taiwan`. Plain semantic search retrieves text that sounds similar to the query — it does not follow logical links.

**2. Multi-hop reasoning cannot be done by similarity search.**
If a query asks "Is Ford at risk from a Taiwan earthquake?", the answer requires following this chain:
```
Taiwan earthquake → affects TSMC → TSMC supplies chips to Ford
```
No amount of embedding similarity will surface Ford as relevant to a Taiwan earthquake unless we traverse those two edges. A tier-3 supplier disruption (three hops away) is invisible to vector search.

**3. Global questions need graph-level reasoning.**
"What is our total East Asian exposure?" cannot be answered by any text chunk. You need to reason over the entire graph and aggregate. Microsoft's 2024 GraphRAG paper proved vanilla RAG fails completely at these community-level questions.

---

## Architecture: Five Modules

```
Raw Text Documents
      │
      ▼
┌─────────────────────────────────────┐
│  MODULE 1 — Knowledge Graph Builder │  spaCy NER + regex triple extraction
│                                     │  → NetworkX DiGraph + FAISS index
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  MODULE 2 — GraphRAG Retrieval      │  query → seed nodes → k-hop expansion
│                                     │  + Louvain community summaries
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  MODULE 3 — Disruption Simulation   │  weighted BFS propagation
│                                     │  → exposure scores per node
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  MODULE 4 — LLM Risk Report         │  enriched prompt → GPT / Mistral
│                                     │  → structured XML-tagged report
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  MODULE 5 — Ablation Study          │  Baseline vs Vector RAG vs GraphRAG+Sim
│                                     │  measured on 5 automatic metrics
└─────────────────────────────────────┘
```

---

## Module 1 — Knowledge Graph Construction

**Goal:** Turn unstructured text into a queryable directed graph.

### What is a Knowledge Graph?

A knowledge graph is a data structure where:
- **Nodes** represent real-world entities (companies, countries, ports, events)
- **Edges** represent typed relationships between them (supplies, depends_on, ships_through, located_in, affected_by)
- Both nodes and edges carry attributes (entity type, confidence score, source sentence)

Example fragment of our graph:
```
Apple (Manufacturer) --[depends_on]--> TSMC (Supplier)
TSMC  (Supplier)     --[located_in]--> Taiwan (Region)
TSMC  (Supplier)     --[depends_on]--> ASML (Supplier)
Taiwan (Region)      --[affected_by]--> Taiwan Earthquake (DisruptionEvent)
```

Once this graph exists, answering "Is Apple exposed to the Taiwan earthquake?" becomes a graph reachability problem, not a text search problem.

### Node Types (our ontology)

| Type | Examples |
|------|---------|
| `Supplier` | TSMC, ASML, CATL, Maersk |
| `Manufacturer` | Apple, Tesla, Ford, BMW |
| `Part` | lithography machines, battery cells, chips |
| `Port` | Port of Shanghai, Port of Rotterdam |
| `Region` | Taiwan, China, Germany, East Asia |
| `LogisticsRoute` | Suez Canal, Strait of Malacca, Trans-Pacific lane |
| `DisruptionEvent` | Taiwan earthquake, Suez Canal blockage, COVID lockdown |
| `Customer` | end buyers |

### Edge Types (our ontology)

| Relation | Direction | Meaning |
|----------|-----------|---------|
| `supplies` | Supplier → Manufacturer/Part | "TSMC supplies chips to Apple" |
| `depends_on` | Manufacturer → Supplier/Part | "Apple depends_on TSMC" |
| `ships_through` | Supplier → Port | "Maersk ships through Rotterdam" |
| `located_in` | Any → Region | "TSMC located_in Taiwan" |
| `affected_by` | Any → DisruptionEvent | "TSMC affected_by earthquake" |
| `alternative_to` | Supplier → Supplier | "GlobalFoundries alternative_to TSMC" |
| `sells_to` | Manufacturer → Customer | flow direction for downstream |

### The Six-Step Pipeline

**Step 1 — Document Loading**

The builder reads `.txt` or `.json` files from `data/raw/`. Each document becomes a dict with a `text` field and a `source` field (file path for provenance).

**Step 2 — Entity Extraction (NER)**

We run spaCy's `en_core_web_sm` model over every sentence. spaCy labels each named entity span with a generic label (ORG, GPE, FAC, EVENT, PRODUCT, LOC). We then classify each span into our supply chain ontology using a three-tier priority system:

1. **Canonical whitelist** — known companies always get the right type. Without this, a sentence like "TSMC supplies advanced chips" would misclassify TSMC as a Part because "chip" appears in the same sentence. The whitelist hardcodes ~30 companies to their correct type.
2. **spaCy label rules** — GPE/LOC → Region, EVENT → DisruptionEvent
3. **Entity-name keywords** — if the entity's own name contains "port" → Port, "canal" → LogisticsRoute, "earthquake" → DisruptionEvent. This fires on the entity's name, not the surrounding text, preventing contamination from context.
4. **ORG default** → Supplier (safe fallback for unknown companies)

**Step 3 — Triple Extraction**

For every pair of entities in the same sentence, we look for a relation trigger between them using compiled regex patterns:

```python
"supplies" / "sourced from"     → RelationType.SUPPLIES
"depends on" / "requires"       → RelationType.DEPENDS_ON
"ships through" / "routes through" → RelationType.SHIPS_THROUGH
"located in" / "headquartered in"  → RelationType.LOCATED_IN
"affected by" / "impacted by"   → RelationType.AFFECTED_BY
```

A triple gets confidence 0.85 if the pattern appears between the two entity spans in text order, 0.65 if not. Triples below the confidence threshold (default 0.5) are discarded.

**Step 4 — Entity Disambiguation**

The same company can appear in text as "TSMC", "Taiwan Semiconductor", "Taiwan Semiconductor Manufacturing Co." — these must be merged to one canonical node. The `EntityDisambiguator` uses sentence-transformers embeddings: if two entity names embed within cosine distance 0.15 of each other and share the same entity type, they are merged to the first canonical form. If the embedding model is unavailable (offline environment), a string-normalization fallback collapses only exact normalized matches.

**Step 5 — Graph Construction**

We construct a `nx.DiGraph` (directed graph from the NetworkX library):
- Each canonical entity → one node with attributes: `entity_type`, `aliases`, `metadata`
- Each triple above the confidence threshold → one directed edge with attributes: `relation`, `weight` (default 1.0), `confidence`, `source_text`

**Step 6 — Edge Weight Assignment (Criticality)**

This is the supply-chain-specific innovation: edges are weighted by criticality, not uniformly.

```
alternative_to edges       → weight 0.0    (they reduce risk, not carry it)
supplies / depends_on edges → weight = 1 / (1 + number_of_alternatives)
```

A supplier with zero alternatives gets edge weight ~1.0 — it is a single-source critical dependency. A supplier with 3 alternatives gets weight ~0.25 — the shock can be absorbed. These weights are used in Module 3 to propagate disruption more strongly along critical paths.

**Node Embedding (FAISS Index)**

After graph construction, every node name is encoded with `sentence-transformers/all-MiniLM-L6-v2` (a 384-dimensional local model, no GPU required). The vectors are stored in a FAISS index. This enables Module 2 to find the graph nodes most semantically similar to a user's query in milliseconds.

---

## Module 2 — GraphRAG Retrieval Layer

**Goal:** Given a natural language query, retrieve the most relevant subgraph from the knowledge graph to use as LLM context.

This is what makes the system a GraphRAG system rather than a plain vector search system.

### Two Retrieval Modes

**Local retrieval** — for specific entity or event questions:
```
"Is TSMC exposed to the earthquake?"
"What does Apple depend on?"
```
→ embed the query → find top-K seed nodes by FAISS similarity → expand outward
  via k-hop ego graph traversal → return the subgraph

**Global retrieval** — for system-wide questions:
```
"What is our overall East Asian risk exposure?"
"Which regions create the most concentration risk?"
```
→ same local retrieval PLUS community summaries from Louvain detection

The pipeline auto-classifies queries by checking for keywords: "overall", "all", "entire", "concentration", "compare", etc. trigger global mode.

### Local Retrieval — Step by Step

**Seed node search:** The query "Which suppliers are exposed to the Taiwan earthquake?" is embedded with `all-MiniLM-L6-v2` into a 384-dim vector. FAISS finds the top-5 graph nodes with the smallest cosine distance to this vector. These are the "seed nodes" — the starting points for traversal.

**K-hop expansion:** From each seed node, we expand 2 hops in both directions (following directed edges). This collects all nodes that are 1 or 2 relationships away from the seed. The result is a subgraph — a small slice of the full graph that is most likely to be relevant to the query.

**Node scoring:** Every node in the subgraph gets a relevance score:
```
seed node similarity score × (decay factor ^ hop_distance)
```
Nodes close to a high-similarity seed score higher. Nodes many hops away score lower.

**Serialization:** The subgraph is converted to a text block for prompt injection:
```
[SUBGRAPH CONTEXT]
TSMC (Supplier) | Relevance: 0.847
  → depends_on ASML (Supplier)
  → located_in Taiwan (Region)
  → affected_by Taiwan Earthquake (DisruptionEvent)

Apple (Manufacturer) | Relevance: 0.612
  → depends_on TSMC (Supplier)
```

### Global Retrieval — Louvain Community Detection

For global queries, we first run the Louvain algorithm on the full graph. Louvain is a community detection algorithm — it partitions the graph into groups of nodes that are more densely connected to each other than to the rest of the graph. In a supply chain graph this naturally produces communities like "East Asian semiconductor suppliers", "European automotive manufacturers", "Middle East logistics routes".

Each community is summarized using T5/BART (an encoder-decoder model). The community summaries are stored to disk and reloaded on query, so this expensive step runs only once.

For a global query, the 3 most relevant communities (by keyword overlap with the query) are prepended to the local subgraph context before the LLM prompt is assembled.

---

## Module 3 — Disruption Simulation Engine

**Goal:** Mathematically propagate a disruption event through the dependency graph and score every node by how exposed it is.

This is the most novel contribution of the project — none of the five papers in our literature review combines GraphRAG with forward-looking simulation.

### What "Disruption Simulation" Means

A disruption event is defined as:
- A ground-zero node (or set of nodes) that are directly struck
- An initial shock value (e.g. 1.0 = total disruption, 0.7 = severe)
- A category (geopolitical, natural disaster, logistics, cyber)

The simulation propagates this shock through the graph using **weighted BFS** (Breadth-First Search):

```
score(ground_zero)        = initial_shock
score(1-hop neighbor)     = initial_shock × decay × edge_weight
score(2-hop neighbor)     = score(1-hop) × decay × edge_weight
...
```

The decay factor (default 0.6) means disruption attenuates with each hop — a tier-3 supplier is less affected than a tier-1 supplier. The edge weight (from Module 1) means disruption travels harder along single-source critical dependencies than along edges where alternatives exist.

### Example Propagation

Trigger: Taiwan earthquake, initial_shock = 0.9, decay = 0.6

```
Taiwan (Region)       → score: 0.90  [CRITICAL]
TSMC (Supplier)       → score: 0.54  [HIGH]     path: Taiwan → TSMC
ASML (Supplier)       → score: 0.32  [MODERATE] path: Taiwan → TSMC → ASML
Apple (Manufacturer)  → score: 0.32  [MODERATE] path: Taiwan → TSMC → Apple
GlobalFoundries       → score: 0.0   [unaffected — has alternative_to edge]
```

### Criticality Weighting

The edge weight from Module 1 makes this physically meaningful. If TSMC has no alternative suppliers, the edge weight is ~1.0 and the full shock passes through. If a supplier has 3 alternatives, the edge weight is 0.25 and only a quarter of the shock passes through — the supply chain has redundancy.

### Output: Exposure Scores + Path Traces

For every affected node the propagator stores:
- The disruption score (0.0 to 1.0)
- The hop distance from ground zero
- The exact path trace: which sequence of edges created the exposure
- Whether the node has alternative suppliers (mitigation availability)
- A severity tier: CRITICAL (>0.7), HIGH (0.5-0.7), MODERATE (0.25-0.5), LOW (<0.25)

### Score Merging (Module 2 + Module 3 Integration)

The simulation engine merges retrieval scores from Module 2 with disruption scores from Module 3:

```
merged_score = 0.5 × retrieval_relevance + 0.5 × disruption_exposure
```

This ensures the LLM context prioritizes nodes that are *both* relevant to the question *and* highly exposed to the disruption — a stronger signal than either score alone.

---

## Module 4 — LLM Generation & Risk Reporting

**Goal:** Take the enriched graph context from Module 3 and generate a structured, cited, actionable risk report.

### The Enriched Prompt

The simulation engine builds a detailed prompt context block that includes every node in the retrieved subgraph with all scores, severity tiers, path traces, and edge criticality flags:

```
=== Enriched Supply Chain Context (GraphRAG + Disruption Simulation) ===
Event       : Taiwan Earthquake (M7.4)
Initial shock: 90%

[CRITICAL] Taiwan [GROUND ZERO] | Disruption: 0.900 | Priority: 0.895
[HIGH] TSMC | Disruption: 0.540 | Relevance: 0.847 | Priority: 0.694
  Exposed via: Taiwan → TSMC
  → supplies Apple (edge_weight=0.97, target_disruption=0.324) *** CRITICAL - single source ***
  → depends_on ASML (edge_weight=0.97, target_disruption=0.323) *** CRITICAL - single source ***

[MODERATE] Apple | Disruption: 0.324 | Relevance: 0.612 | Priority: 0.468
  Exposed via: Taiwan → TSMC → Apple
```

### The LLM Instruction

The LLM (GPT-4o-mini or Mistral-7B via Ollama) is instructed to:
1. Identify the top 5 most exposed entities, citing disruption scores
2. Explain each entity's exposure through its dependency chain
3. Flag CRITICAL single-source dependencies
4. Suggest 2-3 concrete mitigations grounded in graph structure
5. Give an overall resilience assessment
6. Cite specific entity names and graph paths for every claim — no hallucination

### Structured Output Parsing

The prompt asks the LLM to wrap each section in XML tags:
```xml
<critical_entities>...</critical_entities>
<dependency_chains>...</dependency_chains>
<critical_edges>...</critical_edges>
<mitigations>...</mitigations>
<resilience_assessment>...</resilience_assessment>
```

A parser extracts each section into dedicated fields on a `RiskReport` dataclass, making the output machine-readable as well as human-readable. If a smaller model ignores the format tags, the full raw text is stored as a fallback.

### Two LLM Backends

The generator supports two backends behind the same API:
- **OpenAI** (GPT-4o-mini) — highest quality, requires API key
- **Ollama** (Mistral-7B, Qwen 2.5) — runs entirely locally, free, slower

The backend is swapped via a single constructor parameter: `RiskReportGenerator(backend="ollama", model="mistral")`.

---

## Module 5 — Evaluation & Ablation Study

**Goal:** Prove that GraphRAG + Simulation is better than simpler alternatives by comparing three conditions on the same questions.

### Three Conditions

| Condition | What the LLM gets | What it represents |
|-----------|-------------------|-------------------|
| **A — Baseline LLM** | Only the question + event description | "What does the pretrained model know?" |
| **B — Vector RAG** | Top-8 nodes by embedding similarity only, no graph traversal, no scores | "What does semantic search retrieve?" |
| **C — GraphRAG + Sim** | Full enriched context: subgraph + community summaries + disruption scores + path traces | "Our full system" |

### Five Metrics

**1. Multi-hop accuracy** — what fraction of ground-truth affected nodes (defined by the propagation result) are mentioned in the answer?
- Baseline will miss most multi-hop nodes (it has no graph at all)
- Vector RAG will miss nodes that are not textually similar to the query (tier-2 and tier-3 suppliers)
- GraphRAG + Sim follows edges to find them

**2. Citation rate** — what fraction of sentences in the answer cite a specific entity name or graph path?
- Measures grounding: does the answer talk about real entities in our graph or make vague claims?

**3. Disruption coverage** — of all nodes with exposure score ≥ 0.25, what fraction appear in the answer?
- Measures completeness: does the answer surface the full affected population?

**4. Entities mentioned** — raw count of graph node names appearing in the text.

**5. Answer length** — word count (used as a calibration check, not a quality signal).

The evaluator builds an automated comparison table:
```
----------------------------------------------------------------------
  Metric                        Baseline    Vector RAG   GraphRAG+Sim
----------------------------------------------------------------------
  Multi-hop accuracy           0.182       0.364        0.727 *
  Citation rate                0.231       0.519        0.842 *
  Disruption coverage          0.143       0.286        0.714 *
  Entities mentioned               4          11           24 *
  Answer length (words)          180         260          420 *
----------------------------------------------------------------------
  * = best value for this metric
```

The 128-question benchmark suite in `data/eval/benchmark_prompt_suite.json` was built to stress multi-hop reasoning specifically: questions are phrased to require following 2-4 edges to give the correct answer.

---

## The Data Pipeline

```
data/raw/                              ← source text (filings, news, transcripts)
         ↓  Module 1: builder.py
data/processed/
  triples.json                         ← extracted (head, relation, tail) facts
  entities.json                        ← deduplicated entity registry
  graph.gpickle                        ← serialized NetworkX DiGraph
  faiss_index.bin                      ← FAISS vector index for node embeddings
         ↓  Module 2: pipeline.py
data/processed/
  communities.json                     ← Louvain community summaries (cached)
         ↓  Module 3: engine.py
outputs/
  simulation_results/
    taiwan_earthquake.json             ← propagation scores + stats per scenario
         ↓  Module 4: generator.py
outputs/
  reports/
    taiwan_earthquake_report.json      ← structured risk report
         ↓  Module 5: evaluator.py
outputs/
  ablation/
    ablation_taiwan_earthquake.json    ← three-condition comparison metrics
```

---

## Technologies and Why We Use Them

| Technology | Role | Why |
|-----------|------|-----|
| **spaCy `en_core_web_sm`** | Named entity recognition | Fast, CPU-friendly, no GPU needed |
| **sentence-transformers `all-MiniLM-L6-v2`** | Node embedding + query encoding | 384-dim, runs on CPU in ~10ms per batch |
| **FAISS** | Vector similarity search | Millisecond nearest-neighbor over thousands of embeddings |
| **NetworkX** | Graph storage + traversal | Pure Python graph library, simple API for BFS/ego-graph |
| **python-louvain** | Community detection | Louvain algorithm, standard for graph partitioning |
| **T5 / BART** | Community summary generation | Encoder-decoder models — abstractive summarization of node clusters |
| **GPT-4o-mini / Mistral-7B** | Risk report generation | Decoder-only LLMs — the generation step of GraphRAG |
| **LightRAG** | Alternative GraphRAG engine | Used in the Groq-backed chatbot prototype — built-in graph extraction and retrieval |
| **Groq API** | LLM inference | Free API with fast inference, `llama-3.3-70b-versatile` as backbone |
| **Streamlit** | Chatbot frontend | Quick interactive demo without building a full web app |

---

## Course Concepts Coverage

This project deliberately covers every concept in the Generative AI course:

| Concept | Where it appears |
|---------|-----------------|
| **Embeddings** | Node vectors via `all-MiniLM-L6-v2`, used for semantic seed search |
| **Transformer architecture** | Foundation of every model (BERT, T5, BART, GPT, Mistral) |
| **Encoder-only models** | spaCy's transformer for NER; `all-MiniLM-L6-v2` for query encoding |
| **Decoder-only models** | GPT-4o-mini / Mistral for risk report generation |
| **Encoder-decoder models** | T5/BART for community summarization |
| **Fine-tuning** | Planned extension: fine-tune BERT on supply-chain NER labels |
| **RAG** | Baseline condition in the ablation study (Module 5) |
| **GraphRAG** | Core architecture of the full system (Modules 2–4) |

---

## What Makes This Hard

**Graph construction quality is the bottleneck.** If NER misclassifies entities or misses relations, the graph is wrong and everything downstream is wrong. The whitelist, the three-tier classification priority, and the confidence threshold in triple extraction exist entirely to improve graph quality.

**Disambiguation is harder than it looks.** "TSMC", "Taiwan Semiconductor Manufacturing", and "Taiwan Semiconductor Manufacturing Co." must all collapse to one node or the graph fragments. The embedding-based disambiguator handles fuzzy matches; the string-normalization fallback handles offline environments.

**LLM hallucination is the generation risk.** The prompt explicitly instructs the LLM to cite graph paths for every claim and warns it not to invent relationships. The citation rate metric in Module 5 measures how often this instruction is followed. Smaller models (Mistral-7B) follow it less reliably than GPT-4o-mini.

**Multi-hop retrieval is not free.** A 2-hop ego graph from 5 seed nodes can easily include 200+ nodes — too many tokens for a typical LLM context window. The retriever caps the subgraph at `max_nodes=40` and prioritizes by merged score, so the most important nodes always make it in.

---

*This document covers the full project as implemented in the `Mounia-s-branch` codebase.*
