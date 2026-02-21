# GraphRAG for Supply Chain — Concepts Explained

> A beginner-friendly explanation of what this project is actually doing.

---

## The Core Problem

Imagine you ask ChatGPT:
> *"Which of our suppliers will be affected if the port of Rotterdam closes?"*

ChatGPT will give you a **generic answer** because it has never seen your company's data. It doesn't know your suppliers, your routes, your contracts.

The project is about solving this: **how do you make an LLM answer questions about a specific domain it wasn't trained on?**

---

## Solution 1 — Fine-tuning (the hard way)

You take an LLM and **retrain it** on your domain data. Like sending it back to school.

- Needs a lot of GPU compute and time
- Expensive
- The LLM "bakes in" the knowledge but can hallucinate
- If your data changes, you retrain again

This is called **fine-tuning** — it's one valid approach but not what this project does.

---

## Solution 2 — RAG (Retrieval-Augmented Generation)

Instead of retraining the LLM, you **give it the relevant documents at the moment it answers**.

Think of it like an **open-book exam**:
- The LLM = a smart student
- Your documents = the textbook
- RAG = letting the student look up the textbook before answering

**How standard RAG works:**

```
Your question
     │
     ▼
Search documents for relevant text chunks (by similarity)
     │
     ▼
Give those chunks + the question to the LLM
     │
     ▼
LLM reads them and answers
```

No retraining needed. The LLM stays exactly the same. You just feed it context.

---

## The Problem with Standard RAG

Standard RAG searches by **text similarity** — it finds paragraphs that *sound like* your question.

But supply chain questions are often **multi-hop**:
> *"If TSMC is disrupted, which European car manufacturers are at risk?"*

To answer this you need to **chain multiple facts**:
- TSMC → supplies chips to → Bosch
- Bosch → supplies parts to → Volkswagen
- Volkswagen → has assembly plant in → Wolfsburg, Germany

No single paragraph contains all of that. Standard RAG **fails** on these questions because it retrieves isolated text chunks, not connected facts.

---

## Solution 3 — GraphRAG (what this project builds)

GraphRAG first builds a **knowledge graph** from your documents, then queries that graph instead of raw text.

A knowledge graph is a network of facts:

```
[TSMC] ──SUPPLIES_CHIPS_TO──► [Bosch]
[Bosch] ──SUPPLIES_PARTS_TO──► [Volkswagen]
[Volkswagen] ──HAS_PLANT_IN──► [Wolfsburg, Germany]
[Suez Canal] ──BLOCKED_BY──► [Ever Given]
[Ever Given] ──DELAYED──► [Maersk vessels]
```

- **Nodes** = things (companies, ports, products, events, regions)
- **Edges** = relationships between them

When you ask a question, it **traverses this graph** to collect connected facts, then gives all of them to the LLM to synthesize a full answer.

```
Your question
     │
     ▼
Find relevant nodes in the graph
     │
     ▼
Follow edges to collect connected facts (multi-hop)
     │
     ▼
Give all connected context + the question to the LLM
     │
     ▼
LLM gives a grounded, connected answer
```

---

## What LightRAG Does (Our Engine)

We don't build the graph by hand. **LightRAG** reads your documents and uses an LLM to **automatically extract** entities and relationships from text.

The `python -m src.ingest` command does this:
1. Reads all `.txt` / `.pdf` files in `data/raw/`
2. Uses Groq LLM to find entities (companies, ports, products...) and their relations
3. Stores the resulting graph in `graph_storage/`

This graph then powers all queries.

---

## What This Project Delivers

| Step | What happens |
|---|---|
| Feed documents | Supply chain disruption reports go in |
| Graph extraction | LightRAG reads them, finds entities & relations, builds the graph |
| Query | User asks a disruption question via the chatbot |
| GraphRAG retrieval | System traverses the graph to find connected facts |
| Answer | Groq LLM synthesizes a grounded answer |
| **Evaluation** | Compare `naive` mode (standard RAG) vs `hybrid` mode (GraphRAG) — this is the academic contribution |

---

## The 4 Query Modes (Key for Evaluation)

| Mode | What it does | Best for |
|---|---|---|
| `naive` | Searches text only, no graph — **the baseline** | Showing what standard RAG gives |
| `local` | Looks at specific entities and their direct neighbors | Specific questions: *"Who supplies TSMC?"* |
| `global` | Looks at big patterns across the whole graph | Broad questions: *"What are the main risk clusters?"* |
| `hybrid` | Local + Global combined | Complex multi-hop questions |

**The core evaluation**: ask the same question in all 4 modes and show that GraphRAG modes (`local`, `global`, `hybrid`) give better, more connected answers than `naive`.
That difference **is** the academic result of the project.

---

## Tech Stack Summary

| Component | Tool | Why |
|---|---|---|
| GraphRAG engine | LightRAG | Lightweight, CPU-friendly, incremental |
| LLM (inference) | Groq API — Llama 3.1 | Free, fast, no GPU needed |
| Embeddings | sentence-transformers | Runs fully on CPU |
| Graph storage | LightRAG built-in (NetworkX) | No database server needed |
| UI | Streamlit | Simple chatbot in Python |
