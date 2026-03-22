# Data Notes

## Current Files

- `Edgar/`
  Raw EDGAR shards kept locally.
- `interim/sp500_company_corpus.jsonl`
  Company-level merged SP500 transcript corpus.
- `interim/sp500_company_corpus_clean.jsonl`
  Same corpus after boilerplate removal.
- `interim/sp500_company_rag_corpus.jsonl`
  Broad retrieval corpus, good for RAG.
- `interim/sp500_company_graph_knowledge.jsonl`
  Stricter dependency/disruption corpus, better for graph extraction.
- `interim/sp500_company_rag_first3.txt`
  First 3 RAG records in readable text form.
- `interim/sp500_company_graph_knowledge_first3.txt`
  First 3 graph-knowledge records in readable text form.

## Scripts

- `src/data/prepare_sp500_corpus.py`
- `src/data/clean_sp500_company_corpus.py`
- `src/data/filter_sp500_supply_passages.py`
- `src/data/filter_sp500_graph_knowledge.py`
- `src/data/export_jsonl_texts.py`

## What We Did

1. Merged SP500 earnings-call transcripts into one company document per company.
2. Removed transcript boilerplate such as headers, operator turns, and disclaimer-style noise.
3. Built a broad company corpus for retrieval and RAG.
4. Built a second stricter company corpus with explicit supply/dependency/disruption sentences for graph work.
5. Exported the first 3 records from both corpora into `.txt` files for inspection.

## Quick Counts

- `sp500_company_rag_corpus.jsonl`
  `496` company records.
- `sp500_company_graph_knowledge.jsonl`
  `496` company records, much smaller and more focused.
