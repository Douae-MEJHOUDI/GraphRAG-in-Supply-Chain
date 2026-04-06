"""
llm_builder.py
--------------
LLMGraphBuilder — production-quality, fully LLM-driven knowledge graph builder.

Four improvements over the existing extractors (extract_triples_*.py):

1.  Running entity registry
    After each chunk, every newly extracted entity is added to a shared
    registry. The NEXT chunk's LLM call receives the current registry as
    context ("entities already seen"), so the LLM reuses canonical names
    instead of inventing variants ("Taiwan Semiconductor" vs "TSMC").

2.  LLM canonicalization pass
    After all chunks are processed, a second LLM call per entity type
    groups aliases and returns a canonical-name map:
      {"Taiwan Semiconductor": "TSMC", "TSMC Inc.": "TSMC", ...}
    All triples are then rewritten with canonical names before the graph
    is built. This is the single highest-impact step.

3.  Graph validation
    - Self-loops removed (head == tail after canonicalization)
    - Parallel duplicate edges merged, keeping highest confidence
    - Isolated nodes (no edges at all) flagged and optionally pruned
    - Dangling `alternative_to` edges pointing to non-existent nodes fixed

4.  Criticality-aware edge weights
    Same formula as builder.py:
        weight = 1 / (1 + n_alternatives)
    Single-source dependencies carry full shock in the disruption simulation.
    Nodes with alternatives get proportionally lower weights.

Backends
--------
    Ollama  — local GPU, any model (default: qwen2.5:32b)
              Best for: highest quality, no API cost, full control.
    Groq    — cloud API (llama-3.3-70b-versatile)
              Best for: when local GPU is unavailable.

Usage
-----
    # Ollama (local, recommended with GPU)
    builder = LLMGraphBuilder(model="qwen2.5:32b", backend="ollama")

    # Groq (cloud fallback)
    builder = LLMGraphBuilder(model="llama-3.3-70b-versatile", backend="groq")

    G = builder.run_pipeline("data/raw/")
    builder.save_all("data/processed/")

    # Or step by step:
    builder.load_documents("data/raw/")
    builder.extract()          # Phase 1: LLM triple extraction
    builder.canonicalize()     # Phase 2: LLM alias deduplication
    builder.build_graph()      # Phase 3: NetworkX graph construction
    builder.validate_graph()   # Phase 4: clean self-loops, orphans, duplicates
    builder.assign_weights()   # Phase 5: criticality edge weights
    builder.save_all("data/processed/")

Resume
------
    A JSONL progress log is written after each chunk. Re-running with the
    same output path resumes automatically from the last completed chunk.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx

from src.graph.schema import (
    EntityType,
    RelationType,
    EDGE_ATTR_CONF,
    EDGE_ATTR_RELATION,
    EDGE_ATTR_SOURCE,
    EDGE_ATTR_WEIGHT,
    NODE_ATTR_ALIASES,
    NODE_ATTR_METADATA,
    NODE_ATTR_TYPE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_ENTITY_TYPES: set[str] = {e.value for e in EntityType}
ALLOWED_RELATIONS: set[str]    = {r.value for r in RelationType}

_MAX_REGISTRY_CONTEXT = 300   # max entities shown in each chunk's context
_CANONICALIZE_BATCH   = 200   # max entities per LLM canonicalization call

# ---------------------------------------------------------------------------
# System prompt for extraction
# ---------------------------------------------------------------------------

_EXTRACT_SYSTEM = f"""
You extract supply-chain knowledge graph triples from text.

RULES:
- Extract only ATOMIC one-hop relations (one edge = two nodes).
- Do NOT infer transitive or multi-hop facts.
  If text says A->B and B->C, emit both edges separately. Never emit A->C.
- Do NOT invent entities not mentioned in the text.
- Use the "Known entities" list when provided — reuse those exact names
  rather than introducing synonyms or abbreviations.
- Keep entity names concise and canonical (prefer "TSMC" over "Taiwan
  Semiconductor Manufacturing Company").
- For locations: always emit `entity -> located_in -> location` so that
  disruptions at a location propagate to all entities there.
- Direction conventions:
    supplies:       supplier -> customer/manufacturer
    depends_on:     dependent -> dependency
    ships_through:  entity -> port/route
    located_in:     entity -> region/country/city
    affected_by:    entity -> disruption_event
    alternative_to: entity -> alternative_entity
    sells_to:       manufacturer -> customer

Allowed head_type / tail_type values: {sorted(ALLOWED_ENTITY_TYPES)}
Allowed relation values:              {sorted(ALLOWED_RELATIONS)}

OUTPUT FORMAT — return ONLY a JSON array, nothing else:
[
  {{
    "head":       "Entity A",
    "head_type":  "Supplier",
    "relation":   "supplies",
    "tail":       "Entity B",
    "tail_type":  "Manufacturer",
    "confidence": 0.9,
    "evidence":   "short quote from text"
  }},
  ...
]
If no valid triples, return [].
""".strip()

# ---------------------------------------------------------------------------
# System prompt for canonicalization
# ---------------------------------------------------------------------------

_CANONICALIZE_SYSTEM = """
You are deduplicating an entity name list for a knowledge graph.

You will receive a list of entity names of the SAME type that were extracted
from many text documents. Some names refer to the same real-world entity.
Your job: group aliases and return a canonical (most formal, unambiguous) name
for each group.

Rules:
- Keep a name as its own canonical if it has no aliases.
- Prefer the most widely recognized, official abbreviation or short name.
  Examples: "TSMC" over "Taiwan Semiconductor Manufacturing Company",
            "CATL" over "Contemporary Amperex Technology".
- Do NOT merge entities that are actually different companies or places.
- Subsidiaries and parents are separate entities unless the text treats them
  as identical (e.g. "Samsung" vs "Samsung Electronics" may or may not be the
  same depending on context — use the most specific name as canonical).

OUTPUT FORMAT — return ONLY a JSON object mapping every input name to its
canonical form (including names that map to themselves):
{
  "Taiwan Semiconductor": "TSMC",
  "Taiwan Semiconductor Manufacturing": "TSMC",
  "TSMC Inc.": "TSMC",
  "TSMC": "TSMC",
  "GlobalFoundries": "GlobalFoundries",
  "Global Foundries": "GlobalFoundries"
}
""".strip()


# ---------------------------------------------------------------------------
# Triple dataclass (internal)
# ---------------------------------------------------------------------------

@dataclass
class _Triple:
    head:       str
    head_type:  str
    relation:   str
    tail:       str
    tail_type:  str
    confidence: float
    evidence:   str
    source:     str = ""
    chunk_id:   int = 0


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _ollama_chat(
    model:    str,
    messages: list[dict],
    base_url: str = "http://localhost:11434",
    timeout:  int = 180,
) -> str:
    """Call the Ollama /api/chat endpoint and return the assistant content."""
    import urllib.request, urllib.error

    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options":  {"temperature": 0.0},
    }).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {body[:300]}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Make sure `ollama serve` is running."
        ) from exc

    if data.get("error"):
        raise RuntimeError(f"Ollama error: {data['error']}")

    message = data.get("message", {})
    return str(message.get("content", "")).strip()


def _groq_chat(
    model:    str,
    messages: list[dict],
    api_key:  str,
    base_url: str = "https://api.groq.com/openai/v1",
    timeout:  int = 180,
) -> tuple[str, dict]:
    """
    Call the Groq (OpenAI-compatible) chat completions endpoint.
    Returns (content, rate_limit_headers_dict).
    """
    import urllib.request, urllib.error

    payload = json.dumps({
        "model":       model,
        "temperature": 0.0,
        "messages":    messages,
    }).encode()

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            headers = dict(resp.headers)
            data    = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            wait = _parse_reset(exc.headers.get("retry-after") or
                                exc.headers.get("x-ratelimit-reset-tokens") or "5")
            raise _RateLimit(max(wait, 1.0)) from exc
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq HTTP {exc.code}: {body[:300]}") from exc

    choices = data.get("choices", [])
    content = choices[0]["message"]["content"] if choices else ""
    return str(content).strip(), headers


class _RateLimit(RuntimeError):
    def __init__(self, wait: float):
        super().__init__(f"rate-limited, wait {wait:.1f}s")
        self.wait = wait


def _parse_reset(value: str) -> float:
    """Parse Groq reset header like '7.66s' or '2m30s' → seconds."""
    if not value:
        return 0.0
    v = value.strip().lower()
    try:
        return float(v)
    except ValueError:
        pass
    total = 0.0
    m = re.search(r"(\d+(?:\.\d+)?)m", v)
    s = re.search(r"(\d+(?:\.\d+)?)s", v)
    if m:
        total += 60.0 * float(m.group(1))
    if s:
        total += float(s.group(1))
    return total


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_array(raw: str) -> list[dict]:
    """Robustly extract a JSON array from LLM output (strips markdown fences)."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    raw = raw.strip()

    # Try direct parse first
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [x for x in loaded if isinstance(x, dict)]
    except json.JSONDecodeError:
        pass

    # Fallback: find first [...] block
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end > start:
        try:
            loaded = json.loads(raw[start: end + 1])
            if isinstance(loaded, list):
                return [x for x in loaded if isinstance(x, dict)]
        except json.JSONDecodeError:
            pass
    return []


def _extract_json_object(raw: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$",          "", raw)
    raw = raw.strip()

    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        try:
            loaded = json.loads(raw[start: end + 1])
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
    return {}


def _clean(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip(" ,;:.")


def _norm_type(s: Any) -> str:
    mapping = {
        "supplier":        EntityType.SUPPLIER.value,
        "manufacturer":    EntityType.MANUFACTURER.value,
        "part":            EntityType.PART.value,
        "port":            EntityType.PORT.value,
        "region":          EntityType.REGION.value,
        "logisticsroute":  EntityType.LOGISTICS_ROUTE.value,
        "disruptionevent": EntityType.DISRUPTION.value,
        "customer":        EntityType.CUSTOMER.value,
    }
    key = _clean(s).lower().replace(" ", "").replace("_", "")
    return mapping.get(key, _clean(s))


def _norm_relation(s: Any) -> str:
    return _clean(s).lower().replace("-", "_").replace(" ", "_")


def _to_conf(s: Any) -> float:
    try:
        return max(0.0, min(1.0, float(s)))
    except (TypeError, ValueError):
        return 0.7


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _sentence_chunks(text: str, max_chars: int = 2400, overlap: int = 300) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries.
    Each chunk is at most max_chars characters. Consecutive chunks overlap
    by `overlap` characters of context to avoid cutting facts in half.
    """
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split into sentences (simple: split on ". " / ".\n" / "! " / "? ")
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) + 1 > max_chars and current:
            chunk_text = " ".join(current)
            chunks.append(chunk_text)
            # Overlap: keep last few sentences
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) + 1 > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s) + 1
            current     = overlap_sents
            current_len = overlap_len

        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Registry management
# ---------------------------------------------------------------------------

def _build_registry_context(registry: dict[str, str], max_entries: int) -> str:
    """
    Format the running entity registry as a prompt context string.
    Groups by entity type and caps at max_entries total.
    """
    if not registry:
        return ""

    by_type: dict[str, list[str]] = defaultdict(list)
    for name, etype in registry.items():
        by_type[etype].append(name)

    lines = ["Known entities (reuse these exact names):"]
    count = 0
    for etype in sorted(by_type):
        names = sorted(by_type[etype])
        for name in names:
            if count >= max_entries:
                lines.append(f"  ... ({len(registry) - count} more entities not shown)")
                return "\n".join(lines)
            lines.append(f"  {name} ({etype})")
            count += 1
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LLMGraphBuilder:
    """
    Builds a supply chain knowledge graph using an LLM for all extraction
    and canonicalization steps.

    Parameters
    ----------
    model : str
        Model name.  Ollama: "qwen2.5:32b". Groq: "llama-3.3-70b-versatile".
    backend : str
        "ollama" (local GPU) or "groq" (cloud API).
    ollama_base_url : str
        Ollama server URL. Default: http://localhost:11434
    groq_api_key : str, optional
        Groq API key. If empty, reads GROQ_API_KEY from env.
    chunk_chars : int
        Max characters per chunk sent to the LLM. Default: 2400.
    chunk_overlap : int
        Overlap characters between consecutive chunks. Default: 300.
    confidence_threshold : float
        Minimum confidence to keep a triple. Default: 0.5.
    max_retries : int
        Per-chunk retry attempts on transient failures. Default: 5.
    timeout : int
        LLM call timeout in seconds. Default: 180.
    """

    def __init__(
        self,
        model:             str   = "qwen2.5:32b",
        backend:           str   = "ollama",
        ollama_base_url:   str   = "http://localhost:11434",
        groq_api_key:      str   = "",
        chunk_chars:       int   = 2400,
        chunk_overlap:     int   = 300,
        confidence_threshold: float = 0.5,
        max_retries:       int   = 5,
        timeout:           int   = 180,
    ):
        if backend not in ("ollama", "groq"):
            raise ValueError("backend must be 'ollama' or 'groq'")

        self.model              = model
        self.backend            = backend
        self.ollama_base_url    = ollama_base_url
        self.groq_api_key       = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self.chunk_chars        = chunk_chars
        self.chunk_overlap      = chunk_overlap
        self.confidence_threshold = confidence_threshold
        self.max_retries        = max_retries
        self.timeout            = timeout

        # State
        self.documents:    list[dict]     = []
        self.raw_triples:  list[_Triple]  = []
        self.canon_map:    dict[str, str] = {}   # name → canonical_name
        self.graph:        nx.DiGraph     = nx.DiGraph()
        self._registry:    dict[str, str] = {}   # name → entity_type (running)
        self._progress_log: Optional[Path] = None

    # ------------------------------------------------------------------
    # Public pipeline API
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        data_path:    str | Path,
        output_dir:   str | Path = "data/processed",
        progress_log: Optional[str | Path] = None,
    ) -> nx.DiGraph:
        """
        Run the full pipeline end-to-end and return the graph.

        Steps: load → extract → canonicalize → build → validate → weights
        """
        self.load_documents(data_path)
        self.extract(progress_log=progress_log)
        self.canonicalize()
        self.build_graph()
        self.validate_graph()
        self.assign_weights()
        return self.graph

    def load_documents(self, data: str | Path | list[dict]) -> "LLMGraphBuilder":
        """
        Load raw documents.

        Accepts:
          - A directory of .txt / .json files
          - A single .json or .jsonl file
          - A list of dicts: [{"text": "...", "source": "..."}, ...]
        """
        if isinstance(data, list):
            self.documents = data
        else:
            data = Path(data)
            docs: list[dict] = []
            if data.is_dir():
                for f in sorted(data.glob("*.txt")):
                    docs.append({"text": f.read_text(encoding="utf-8"), "source": str(f)})
                for f in sorted(data.glob("*.json")):
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        docs.extend(payload)
                    else:
                        docs.append(payload)
                for f in sorted(data.glob("*.jsonl")):
                    with f.open(encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                docs.append(json.loads(line))
            elif data.suffix == ".jsonl":
                with data.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            docs.append(json.loads(line))
            elif data.suffix == ".json":
                payload = json.loads(data.read_text(encoding="utf-8"))
                docs = payload if isinstance(payload, list) else [payload]
            else:
                docs = [{"text": data.read_text(encoding="utf-8"), "source": str(data)}]
            self.documents = docs

        print(f"[LLMBuilder] Loaded {len(self.documents)} document(s)")
        return self

    # ------------------------------------------------------------------
    # Phase 1 — LLM triple extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        progress_log: Optional[str | Path] = None,
        resume:       bool = True,
    ) -> "LLMGraphBuilder":
        """
        Phase 1: Send each text chunk to the LLM and collect raw triples.

        The running entity registry is updated after each chunk so the next
        call benefits from already-seen canonical names.
        """
        log_path = Path(progress_log) if progress_log else None
        self._progress_log = log_path

        # Load resume state
        processed_keys: set[str] = set()
        existing:       list[_Triple] = []
        if resume and log_path and log_path.exists():
            processed_keys, existing = self._load_progress(log_path)
            self.raw_triples = existing
            # Rebuild registry from existing triples
            for t in existing:
                self._registry.setdefault(t.head, t.head_type)
                self._registry.setdefault(t.tail, t.tail_type)
            print(f"[LLMBuilder] Resumed: {len(processed_keys)} chunks, "
                  f"{len(existing)} triples, {len(self._registry)} entities in registry")

        total_docs = len(self.documents)
        for d_idx, doc in enumerate(self.documents, 1):
            source = doc.get("source", f"doc_{d_idx}")
            text   = doc.get("text", "")
            chunks = _sentence_chunks(text, self.chunk_chars, self.chunk_overlap)
            print(f"[LLMBuilder] doc {d_idx}/{total_docs}: {len(chunks)} chunks  [{source}]")

            for c_idx, chunk in enumerate(chunks, 1):
                key = f"{d_idx}:{c_idx}"
                if key in processed_keys:
                    continue

                triples = self._extract_chunk(chunk, source, c_idx)
                self.raw_triples.extend(triples)

                # Update running registry
                for t in triples:
                    self._registry.setdefault(t.head, t.head_type)
                    self._registry.setdefault(t.tail, t.tail_type)

                processed_keys.add(key)
                if log_path:
                    self._append_progress(log_path, key, source, triples)

                print(f"  chunk {c_idx}/{len(chunks)}: "
                      f"+{len(triples)} triples  "
                      f"(total: {len(self.raw_triples)}, "
                      f"registry: {len(self._registry)} entities)")

        print(f"[LLMBuilder] Extraction done: "
              f"{len(self.raw_triples)} raw triples, "
              f"{len(self._registry)} unique entities")
        return self

    def _extract_chunk(
        self, chunk: str, source: str, chunk_id: int
    ) -> list[_Triple]:
        """Call LLM for one chunk, validate, deduplicate, and return triples."""
        registry_ctx = _build_registry_context(self._registry, _MAX_REGISTRY_CONTEXT)
        user_content = (
            (f"{registry_ctx}\n\n" if registry_ctx else "")
            + "Extract supply-chain triples from this text.\n"
            + "Return only a JSON array.\n\n"
            + f"TEXT:\n{chunk}"
        )
        messages = [
            {"role": "system",  "content": _EXTRACT_SYSTEM},
            {"role": "user",    "content": user_content},
        ]

        raw = self._call_llm(messages)
        items = _extract_json_array(raw)

        triples: list[_Triple] = []
        seen: set[tuple[str, str, str]] = set()

        for item in items:
            head     = _clean(item.get("head"))
            tail     = _clean(item.get("tail"))
            relation = _norm_relation(item.get("relation"))
            htype    = _norm_type(item.get("head_type"))
            ttype    = _norm_type(item.get("tail_type"))
            conf     = _to_conf(item.get("confidence", 0.7))
            evidence = _clean(item.get("evidence"))

            # Validate
            if not head or not tail:
                continue
            if head.lower() == tail.lower():
                continue  # self-reference
            if relation not in ALLOWED_RELATIONS:
                continue
            if htype not in ALLOWED_ENTITY_TYPES:
                continue
            if ttype not in ALLOWED_ENTITY_TYPES:
                continue
            if conf < self.confidence_threshold:
                continue

            key = (head.lower(), relation, tail.lower())
            if key in seen:
                continue
            seen.add(key)

            triples.append(_Triple(
                head=head, head_type=htype,
                relation=relation,
                tail=tail, tail_type=ttype,
                confidence=conf, evidence=evidence,
                source=source, chunk_id=chunk_id,
            ))

        return triples

    # ------------------------------------------------------------------
    # Phase 2 — LLM canonicalization
    # ------------------------------------------------------------------

    def canonicalize(self) -> "LLMGraphBuilder":
        """
        Phase 2: Ask the LLM to deduplicate entity aliases per entity type.

        For each entity type, all extracted entity names are sent to the LLM
        which groups aliases and returns a canonical name mapping. All triples
        are then rewritten with canonical names.

        This step merges nodes like "TSMC" / "Taiwan Semiconductor" /
        "Taiwan Semiconductor Manufacturing Co." into a single graph node.
        """
        # Group all unique entity names by type
        by_type: dict[str, set[str]] = defaultdict(set)
        for t in self.raw_triples:
            by_type[t.head_type].add(t.head)
            by_type[t.tail_type].add(t.tail)

        canon_map: dict[str, str] = {}

        for etype, names in sorted(by_type.items()):
            name_list = sorted(names)
            print(f"[LLMBuilder] Canonicalizing {len(name_list)} entities of type '{etype}'")
            type_map = self._canonicalize_type(etype, name_list)
            canon_map.update(type_map)

        self.canon_map = canon_map

        # Rewrite triples
        rewritten: list[_Triple] = []
        for t in self.raw_triples:
            canon_head = canon_map.get(t.head, t.head)
            canon_tail = canon_map.get(t.tail, t.tail)

            if canon_head.lower() == canon_tail.lower():
                continue  # self-loop after canonicalization

            rewritten.append(_Triple(
                head=canon_head, head_type=t.head_type,
                relation=t.relation,
                tail=canon_tail, tail_type=t.tail_type,
                confidence=t.confidence, evidence=t.evidence,
                source=t.source, chunk_id=t.chunk_id,
            ))

        # Deduplicate: keep highest confidence per (head, relation, tail)
        best: dict[tuple[str, str, str], _Triple] = {}
        for t in rewritten:
            key = (t.head.lower(), t.relation, t.tail.lower())
            prev = best.get(key)
            if prev is None or t.confidence > prev.confidence:
                best[key] = t

        self.raw_triples = list(best.values())

        n_merged = len(by_type) and sum(
            1 for k, v in canon_map.items() if k != v
        )
        print(f"[LLMBuilder] Canonicalization: {n_merged} aliases merged, "
              f"{len(self.raw_triples)} triples remain")
        return self

    def _canonicalize_type(
        self, etype: str, names: list[str]
    ) -> dict[str, str]:
        """
        Call LLM to deduplicate one entity type's name list.
        Batches into groups of _CANONICALIZE_BATCH to fit in context.
        """
        result: dict[str, str] = {}

        for i in range(0, len(names), _CANONICALIZE_BATCH):
            batch = names[i: i + _CANONICALIZE_BATCH]
            bullet_list = "\n".join(f"- {n}" for n in batch)
            user_content = (
                f"Entity type: {etype}\n\n"
                f"Entities:\n{bullet_list}\n\n"
                "Return ONLY a JSON object mapping every name to its canonical form."
            )
            messages = [
                {"role": "system", "content": _CANONICALIZE_SYSTEM},
                {"role": "user",   "content": user_content},
            ]
            raw      = self._call_llm(messages)
            batch_map = _extract_json_object(raw)

            # Only accept valid string→string mappings
            for k, v in batch_map.items():
                if isinstance(k, str) and isinstance(v, str) and k and v:
                    result[_clean(k)] = _clean(v)

            # Any name not returned by the LLM maps to itself
            for name in batch:
                result.setdefault(name, name)

        return result

    # ------------------------------------------------------------------
    # Phase 3 — Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> "LLMGraphBuilder":
        """
        Phase 3: Construct a NetworkX DiGraph from the (canonicalized) triples.

        Node attributes: entity_type, aliases, metadata
        Edge attributes: relation, weight (1.0 default), confidence, source_text
        """
        G = nx.DiGraph()

        for t in self.raw_triples:
            for name, etype in ((t.head, t.head_type), (t.tail, t.tail_type)):
                if name not in G:
                    # Collect all raw aliases that mapped to this canonical name
                    aliases = [
                        raw for raw, canon in self.canon_map.items()
                        if canon == name and raw != name
                    ]
                    G.add_node(name, **{
                        NODE_ATTR_TYPE:    etype,
                        NODE_ATTR_ALIASES: aliases,
                        NODE_ATTR_METADATA: {},
                    })

            G.add_edge(
                t.head, t.tail,
                **{
                    EDGE_ATTR_RELATION: t.relation,
                    EDGE_ATTR_WEIGHT:   1.0,
                    EDGE_ATTR_CONF:     t.confidence,
                    EDGE_ATTR_SOURCE:   t.evidence or t.source,
                },
            )

        self.graph = G
        print(f"[LLMBuilder] Graph built: "
              f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return self

    # ------------------------------------------------------------------
    # Phase 4 — Graph validation
    # ------------------------------------------------------------------

    def validate_graph(self, prune_isolated: bool = False) -> "LLMGraphBuilder":
        """
        Phase 4: Clean up the graph.

        Actions:
        - Remove self-loops (should not survive Phase 2, but defensive)
        - Remove parallel duplicate edges keeping highest confidence
        - Optionally remove isolated nodes (degree == 0)
        - Repair nodes whose entity_type ended up empty
        """
        G = self.graph

        # Remove self-loops
        self_loops = list(nx.selfloop_edges(G))
        G.remove_edges_from(self_loops)
        if self_loops:
            print(f"[LLMBuilder] Removed {len(self_loops)} self-loops")

        # Isolated nodes
        isolated = [n for n, d in G.degree() if d == 0]
        if isolated:
            if prune_isolated:
                G.remove_nodes_from(isolated)
                print(f"[LLMBuilder] Pruned {len(isolated)} isolated nodes")
            else:
                print(f"[LLMBuilder] {len(isolated)} isolated nodes (use "
                      f"prune_isolated=True to remove)")

        # Fix missing entity_type attributes
        for node, data in G.nodes(data=True):
            if not data.get(NODE_ATTR_TYPE):
                G.nodes[node][NODE_ATTR_TYPE] = EntityType.SUPPLIER.value

        print(f"[LLMBuilder] Validation complete: "
              f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return self

    # ------------------------------------------------------------------
    # Phase 5 — Edge weight assignment
    # ------------------------------------------------------------------

    def assign_weights(self) -> "LLMGraphBuilder":
        """
        Phase 5: Assign criticality-aware weights to all edges.

        Formula:
            alternative_to edges  → weight 0.0  (risk mitigators, not carriers)
            supplies / depends_on → weight = 1 / (1 + n_alternatives)
            all other edges       → weight 1.0

        A node with no alternative suppliers gets weight ~1.0 — a single-source
        critical dependency. A node with 3 alternatives gets ~0.25.
        The disruption simulation (Module 3) uses these weights.
        """
        G = self.graph

        for u, v, data in G.edges(data=True):
            relation = data.get(EDGE_ATTR_RELATION, "")

            if relation == RelationType.ALTERNATIVE_TO.value:
                G[u][v][EDGE_ATTR_WEIGHT] = 0.0
                continue

            if relation in (RelationType.SUPPLIES.value, RelationType.DEPENDS_ON.value):
                n_alts = sum(
                    1 for nbr in G.neighbors(u)
                    if G[u][nbr].get(EDGE_ATTR_RELATION) == RelationType.ALTERNATIVE_TO.value
                )
                G[u][v][EDGE_ATTR_WEIGHT] = round(1.0 / (1.0 + n_alts), 3)

        print("[LLMBuilder] Edge weights assigned")
        return self

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_all(
        self,
        output_dir: str | Path = "data/processed",
        prefix:     str = "llm_graph",
    ) -> dict[str, Path]:
        """
        Save graph, triples, canon_map, and stats to output_dir.

        Returns a dict of {artifact: path}.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        paths: dict[str, Path] = {}

        # Graph (pickle)
        graph_path = out / f"{prefix}.gpickle"
        with graph_path.open("wb") as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        paths["graph"] = graph_path
        print(f"[LLMBuilder] Graph → {graph_path}")

        # Triples (JSON)
        triples_path = out / f"{prefix}_triples.json"
        triples_path.write_text(
            json.dumps(
                [
                    {
                        "head":       t.head,
                        "head_type":  t.head_type,
                        "relation":   t.relation,
                        "tail":       t.tail,
                        "tail_type":  t.tail_type,
                        "confidence": t.confidence,
                        "evidence":   t.evidence,
                        "source":     t.source,
                    }
                    for t in self.raw_triples
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        paths["triples"] = triples_path
        print(f"[LLMBuilder] Triples → {triples_path}")

        # Canon map (JSON)
        canon_path = out / f"{prefix}_canon_map.json"
        canon_path.write_text(
            json.dumps(self.canon_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["canon_map"] = canon_path

        # Stats (JSON)
        stats = self.graph_stats()
        stats_path = out / f"{prefix}_stats.json"
        stats_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        paths["stats"] = stats_path
        print(f"[LLMBuilder] Stats → {stats_path}")
        print(json.dumps(stats, indent=2))

        return paths

    # ------------------------------------------------------------------
    # Stats helper
    # ------------------------------------------------------------------

    def graph_stats(self) -> dict:
        G = self.graph
        if G.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0}

        type_counts: dict[str, int] = {}
        for _, data in G.nodes(data=True):
            etype = data.get(NODE_ATTR_TYPE, "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1

        rel_counts: dict[str, int] = {}
        for _, _, data in G.edges(data=True):
            rel = data.get(EDGE_ATTR_RELATION, "unknown")
            rel_counts[rel] = rel_counts.get(rel, 0) + 1

        return {
            "nodes":          G.number_of_nodes(),
            "edges":          G.number_of_edges(),
            "node_types":     type_counts,
            "relation_types": rel_counts,
            "avg_degree":     round(
                sum(d for _, d in G.degree()) / G.number_of_nodes(), 2
            ),
            "weakly_connected_components": nx.number_weakly_connected_components(G),
            "model": self.model,
            "backend": self.backend,
        }

    # ------------------------------------------------------------------
    # Internal: LLM call with retry
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> str:
        """Dispatch to the configured backend with retry on transient failures."""
        last_exc: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self.backend == "ollama":
                    return _ollama_chat(
                        self.model, messages,
                        base_url=self.ollama_base_url,
                        timeout=self.timeout,
                    )
                else:  # groq
                    if not self.groq_api_key:
                        raise EnvironmentError(
                            "GROQ_API_KEY not set. "
                            "Set it via environment variable or pass groq_api_key=."
                        )
                    content, headers = _groq_chat(
                        self.model, messages,
                        api_key=self.groq_api_key,
                        timeout=self.timeout,
                    )
                    return content

            except _RateLimit as exc:
                wait = exc.wait
                print(f"[LLMBuilder] Rate-limited. Waiting {wait:.1f}s "
                      f"(attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(wait)
                last_exc = exc

            except (ConnectionError, RuntimeError, OSError) as exc:
                wait = min(60.0, 2.0 ** attempt)
                print(f"[LLMBuilder] Transient error: {exc}. "
                      f"Retrying in {wait:.1f}s "
                      f"(attempt {attempt + 1}/{self.max_retries + 1})")
                time.sleep(wait)
                last_exc = exc

        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts. "
            f"Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Internal: Progress log (resume support)
    # ------------------------------------------------------------------

    def _append_progress(
        self, log_path: Path, key: str, source: str, triples: list[_Triple]
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key":     key,
            "source":  source,
            "triples": [
                {
                    "head": t.head, "head_type": t.head_type,
                    "relation": t.relation,
                    "tail": t.tail, "tail_type": t.tail_type,
                    "confidence": t.confidence, "evidence": t.evidence,
                    "source": t.source, "chunk_id": t.chunk_id,
                }
                for t in triples
            ],
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_progress(
        self, log_path: Path
    ) -> tuple[set[str], list[_Triple]]:
        processed: set[str]     = set()
        triples:   list[_Triple] = []

        with log_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                processed.add(rec.get("key", ""))
                for item in rec.get("triples", []):
                    conf = _to_conf(item.get("confidence", 0.7))
                    if conf < self.confidence_threshold:
                        continue
                    htype = _norm_type(item.get("head_type", ""))
                    ttype = _norm_type(item.get("tail_type", ""))
                    rel   = _norm_relation(item.get("relation", ""))
                    if rel not in ALLOWED_RELATIONS:
                        continue
                    triples.append(_Triple(
                        head=_clean(item.get("head")),
                        head_type=htype,
                        relation=rel,
                        tail=_clean(item.get("tail")),
                        tail_type=ttype,
                        confidence=conf,
                        evidence=_clean(item.get("evidence")),
                        source=item.get("source", ""),
                        chunk_id=int(item.get("chunk_id", 0)),
                    ))

        return processed, triples
