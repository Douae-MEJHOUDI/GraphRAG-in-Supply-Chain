from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import time
from pathlib import Path
from typing import Any

import networkx as nx
import requests

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


DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_ENV_FILE = ".env"

ALLOWED_ENTITY_TYPES = {e.value for e in EntityType}
ALLOWED_RELATIONS = {
    RelationType.SUPPLIES.value,
    RelationType.DEPENDS_ON.value,
    RelationType.LOCATED_IN.value,
    RelationType.AFFECTED_BY.value,
    RelationType.SHIPS_THROUGH.value,
    RelationType.ALTERNATIVE_TO.value,
    RelationType.SELLS_TO.value,
    RelationType.CONNECTS.value,
}

SYSTEM_PROMPT = f"""
You extract supply-chain triples from text for graph construction.

Goal:
- Extract only atomic one-hop relations (2 nodes per triple).
- Do NOT infer transitive/multi-hop relations.
- Multi-level propagation is handled by graph traversal later, not by you.
- Build reliable location links so disruption-at-location can propagate to entities.

Output format:
- Return ONLY a JSON array.
- Every item must be:
  {{
    "head": "...",
    "head_type": "...",
    "relation": "...",
    "tail": "...",
    "tail_type": "...",
    "confidence": 0.0-1.0,
    "evidence": "short quote or sentence"
  }}

Rules:
- Keep only explicit or strongly implied one-hop facts.
- If text supports A->B and B->C, output both edges separately.
- Never add A->C unless the text explicitly states A->C.
- Do NOT invent entities.
- Keep entity names concise and canonical.
- Prefer canonical location entities (city/region/country as `Region` type).
- Direction matters:
  - supplies: supplier -> customer
  - depends_on: dependent -> dependency
  - affected_by: impacted entity -> disruption event
  - located_in: entity -> location
- For location-triggered propagation, emit:
  - company/supplier/port/route/disruption_event -> located_in -> location
  - location hierarchy when explicit (city -> located_in -> country/region)
- If a disruption is stated in a place ("war in X", "earthquake in Y"),
  model the disruption event node and add disruption_event -> located_in -> place.
- Allowed head_type/tail_type: {sorted(ALLOWED_ENTITY_TYPES)}
- Allowed relation: {sorted(ALLOWED_RELATIONS)}
- If no valid triples, return [].
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract supply-chain triples with Groq Llama 3 and build a graph."
    )
    parser.add_argument(
        "--input",
        default="data/interim/sp500_company_graph_knowledge.jsonl",
        help="Input JSON/JSONL with text fields.",
    )
    parser.add_argument(
        "--output-triples",
        default="data/processed/sp500_graph_knowledge_triples_llama3.json",
        help="Output triples JSON path.",
    )
    parser.add_argument(
        "--output-graph",
        default="data/processed/sp500_graph_knowledge_graph_llama3.gpickle",
        help="Output graph pickle path.",
    )
    parser.add_argument(
        "--output-stats",
        default="data/processed/sp500_graph_knowledge_stats_llama3.json",
        help="Output stats JSON path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Groq model name (default: llama-3.3-70b-versatile).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Groq OpenAI-compatible base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Groq API key. If empty, reads GROQ_API_KEY from env or .env.",
    )
    parser.add_argument(
        "--env-file",
        default=DEFAULT_ENV_FILE,
        help="Path to .env file used to load GROQ_API_KEY.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Optional cap on number of docs (0 = all).",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=2400,
        help="Approx max chars per chunk sent to LLM.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="Overlap chars between consecutive chunks.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between requests in seconds.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print ETA milestone every N docs (default: 5).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries for rate-limited or transient failures.",
    )
    parser.add_argument(
        "--retry-backoff-base",
        type=float,
        default=1.5,
        help="Base backoff seconds for retries.",
    )
    parser.add_argument(
        "--retry-backoff-max",
        type=float,
        default=45.0,
        help="Max backoff seconds for retries.",
    )
    parser.add_argument(
        "--min-remaining-tokens",
        type=int,
        default=1200,
        help="If remaining TPM drops below this, wait until token reset.",
    )
    parser.add_argument(
        "--service-tier",
        default="",
        help='Optional Groq service tier (example: "flex"). Empty = default tier.',
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing progress log/output (default: true).",
    )
    parser.add_argument(
        "--progress-log",
        default="",
        help="Path to append per-response progress JSONL. Default: derived from output-triples.",
    )
    parser.add_argument(
        "--save-every-docs",
        type=int,
        default=1,
        help="Persist merged triples/graph/stats every N docs (default: 1).",
    )
    return parser.parse_args()


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def _load_docs(path: Path, max_docs: int = 0) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)
                docs.append(
                    {
                        "source": obj.get("source", f"doc_{i}"),
                        "text": obj.get("text", ""),
                    }
                )
                if max_docs and len(docs) >= max_docs:
                    break
        return docs

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        for i, obj in enumerate(payload):
            docs.append(
                {
                    "source": obj.get("source", f"doc_{i}"),
                    "text": obj.get("text", ""),
                }
            )
            if max_docs and len(docs) >= max_docs:
                break
    else:
        docs.append(
            {
                "source": payload.get("source", "doc_0"),
                "text": payload.get("text", ""),
            }
        )
    return docs


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    step = max(200, max_chars - overlap)
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        if end < n:
            split = text.rfind(". ", start, end)
            if split > start + 400:
                end = split + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start += step
    return chunks


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    raw = raw.replace("```json", "```")
    if raw.startswith("```"):
        raw = raw.strip("`").strip()

    try:
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return [x for x in loaded if isinstance(x, dict)]
    except Exception:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        sub = raw[start : end + 1]
        try:
            loaded = json.loads(sub)
            if isinstance(loaded, list):
                return [x for x in loaded if isinstance(x, dict)]
        except Exception:
            return []
    return []


def _clean_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;:.")


def _clean_type(value: Any) -> str:
    text = _clean_name(value)
    mapping = {
        "supplier": EntityType.SUPPLIER.value,
        "manufacturer": EntityType.MANUFACTURER.value,
        "part": EntityType.PART.value,
        "port": EntityType.PORT.value,
        "region": EntityType.REGION.value,
        "logisticsroute": EntityType.LOGISTICS_ROUTE.value,
        "disruptionevent": EntityType.DISRUPTION.value,
        "customer": EntityType.CUSTOMER.value,
    }
    key = text.lower().replace(" ", "")
    return mapping.get(key, text)


def _clean_relation(value: Any) -> str:
    text = _clean_name(value).lower()
    text = text.replace("-", "_").replace(" ", "_")
    return text


def _to_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.7
    return max(0.0, min(1.0, conf))


def _parse_reset_seconds(value: str | None) -> float:
    """
    Parse Groq reset headers like:
    - "7.66s"
    - "2m59.56s"
    """
    if not value:
        return 0.0
    text = value.strip().lower()
    if not text:
        return 0.0

    # fast path: plain numeric seconds
    try:
        return max(0.0, float(text))
    except Exception:
        pass

    total = 0.0
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)m", text)
    s = re.search(r"([0-9]+(?:\.[0-9]+)?)s", text)
    if m:
        total += 60.0 * float(m.group(1))
    if s:
        total += float(s.group(1))
    return max(0.0, total)


class _RateLimitError(RuntimeError):
    def __init__(self, wait_seconds: float, details: str = ""):
        super().__init__(details or "rate limited")
        self.wait_seconds = max(0.0, wait_seconds)


def _validate_triple(item: dict[str, Any]) -> dict[str, Any] | None:
    head = _clean_name(item.get("head"))
    tail = _clean_name(item.get("tail"))
    relation = _clean_relation(item.get("relation"))
    head_type = _clean_type(item.get("head_type"))
    tail_type = _clean_type(item.get("tail_type"))
    evidence = _clean_name(item.get("evidence"))
    conf = _to_confidence(item.get("confidence", 0.7))

    if not head or not tail:
        return None
    if head.lower() == tail.lower():
        return None
    if relation not in ALLOWED_RELATIONS:
        return None
    if head_type not in ALLOWED_ENTITY_TYPES:
        return None
    if tail_type not in ALLOWED_ENTITY_TYPES:
        return None

    return {
        "head": head,
        "head_type": head_type,
        "relation": relation,
        "tail": tail,
        "tail_type": tail_type,
        "confidence": conf,
        "evidence": evidence,
    }


def _groq_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    service_tier: str = "",
) -> tuple[str, requests.structures.CaseInsensitiveDict[str]]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.0,
        "top_p": 0.9,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if service_tier.strip():
        payload["service_tier"] = service_tier.strip()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    if resp.status_code == 429:
        wait_retry = _parse_reset_seconds(resp.headers.get("retry-after"))
        wait_tokens = _parse_reset_seconds(resp.headers.get("x-ratelimit-reset-tokens"))
        wait = max(wait_retry, wait_tokens, 1.0)
        raise _RateLimitError(wait_seconds=wait, details="HTTP 429 Too Many Requests")
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        return "", resp.headers
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return " ".join(x for x in parts if x).strip(), resp.headers
    return str(content).strip(), resp.headers


def _build_prompt(source: str, chunk_id: int, chunk_text: str) -> str:
    return (
        f"Source: {source}\n"
        f"Chunk: {chunk_id}\n\n"
        "Extract only atomic one-hop supply-chain triples from this text.\n"
        "Do not infer transitive links.\n"
        "If there is a chain, emit each local edge separately.\n"
        "Capture location edges so location shocks can be propagated later.\n"
        "Return only a JSON array.\n\n"
        f"TEXT:\n{chunk_text}"
    )


def _attach_dependency_alternatives(
    triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add OR-choice metadata on dependency edges.

    Policy:
    - For `supplies`: suppliers are alternative options when they share the
      same buyer tail (A->Y, B->Y).
    - For `depends_on`: dependencies are alternative options when they share
      the same dependent head (Y->A, Y->B).

    Each eligible edge gets:
    - option_group_id
    - alternatives: list of all other options in the same group
    - selection_mode: "OR"
    """
    option_map: dict[tuple[str, str], set[str]] = {}
    triple_to_slot: list[tuple[dict[str, Any], tuple[str, str], str] | None] = []

    for t in triples:
        rel = str(t.get("relation", ""))
        if rel == RelationType.SUPPLIES.value:
            slot_entity = str(t.get("tail", ""))
            option_entity = str(t.get("head", ""))
        elif rel == RelationType.DEPENDS_ON.value:
            slot_entity = str(t.get("head", ""))
            option_entity = str(t.get("tail", ""))
        else:
            triple_to_slot.append(None)
            continue

        slot_key = (rel, slot_entity.lower())
        option_map.setdefault(slot_key, set()).add(option_entity)
        triple_to_slot.append((t, slot_key, slot_entity))

    for item in triple_to_slot:
        if item is None:
            continue

        t, slot_key, slot_entity = item
        rel = str(t.get("relation", ""))
        current_option = str(t.get("head", "")) if rel == RelationType.SUPPLIES.value else str(t.get("tail", ""))
        all_options = sorted(option_map.get(slot_key, set()))
        alternatives = [x for x in all_options if x.lower() != current_option.lower()]

        t["option_group_id"] = f"{rel}:{slot_entity}"
        t["selection_mode"] = "OR"
        t["alternatives"] = alternatives

    return triples


def _build_graph(triples: list[dict[str, Any]]) -> nx.DiGraph:
    G = nx.DiGraph()
    for t in triples:
        h = t["head"]
        r = t["relation"]
        ta = t["tail"]
        h_type = t["head_type"]
        t_type = t["tail_type"]

        if h not in G:
            G.add_node(
                h,
                **{
                    NODE_ATTR_TYPE: h_type,
                    NODE_ATTR_ALIASES: [],
                    NODE_ATTR_METADATA: {},
                },
            )
        if ta not in G:
            G.add_node(
                ta,
                **{
                    NODE_ATTR_TYPE: t_type,
                    NODE_ATTR_ALIASES: [],
                    NODE_ATTR_METADATA: {},
                },
            )

        G.add_edge(
            h,
            ta,
            **{
                EDGE_ATTR_RELATION: r,
                EDGE_ATTR_CONF: t["confidence"],
                EDGE_ATTR_SOURCE: t.get("evidence", ""),
                EDGE_ATTR_WEIGHT: 1.0,
                "option_group_id": t.get("option_group_id", ""),
                "selection_mode": t.get("selection_mode", ""),
                "alternatives": t.get("alternatives", []),
            },
        )
    return G


def _format_duration(seconds: float) -> str:
    s = int(max(0, round(seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h {m}m {sec}s"
    if m > 0:
        return f"{m}m {sec}s"
    return f"{sec}s"


def _graph_stats(G: nx.DiGraph) -> dict[str, Any]:
    node_types: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        k = str(data.get(NODE_ATTR_TYPE, "Unknown"))
        node_types[k] = node_types.get(k, 0) + 1

    rel_types: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        k = str(data.get(EDGE_ATTR_RELATION, "unknown"))
        rel_types[k] = rel_types.get(k, 0) + 1

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": node_types,
        "relation_types": rel_types,
    }


def _default_progress_log_path(out_triples: Path) -> Path:
    return out_triples.with_name(f"{out_triples.stem}.progress.jsonl")


def _chunk_key(doc_index: int, chunk_index: int) -> str:
    return f"{doc_index}:{chunk_index}"


def _load_existing_triples(out_triples: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not out_triples.exists():
        return merged
    try:
        payload = json.loads(out_triples.read_text(encoding="utf-8"))
    except Exception:
        return merged
    if not isinstance(payload, list):
        return merged
    for item in payload:
        if not isinstance(item, dict):
            continue
        validated = _validate_triple(item)
        if not validated:
            continue
        key = (
            validated["head"].lower(),
            validated["relation"],
            validated["tail"].lower(),
        )
        prev = merged.get(key)
        if prev is None or validated["confidence"] > prev["confidence"]:
            merged[key] = {
                **validated,
                "source": item.get("source", ""),
                "chunk_id": item.get("chunk_id", 0),
            }
    return merged


def _load_progress_state(
    progress_log: Path,
) -> tuple[set[str], dict[tuple[str, str, str], dict[str, Any]]]:
    processed_chunks: set[str] = set()
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not progress_log.exists():
        return processed_chunks, merged

    with progress_log.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            doc_index = int(rec.get("doc_index", 0))
            chunk_index = int(rec.get("chunk_index", 0))
            if doc_index > 0 and chunk_index > 0:
                processed_chunks.add(_chunk_key(doc_index, chunk_index))

            triples = rec.get("triples", [])
            if not isinstance(triples, list):
                continue
            for item in triples:
                if not isinstance(item, dict):
                    continue
                validated = _validate_triple(item)
                if not validated:
                    continue
                key = (
                    validated["head"].lower(),
                    validated["relation"],
                    validated["tail"].lower(),
                )
                prev = merged.get(key)
                if prev is None or validated["confidence"] > prev["confidence"]:
                    merged[key] = {
                        **validated,
                        "source": item.get("source", rec.get("source", "")),
                        "chunk_id": item.get("chunk_id", chunk_index),
                    }
    return processed_chunks, merged


def _append_progress_record(progress_log: Path, record: dict[str, Any]) -> None:
    progress_log.parent.mkdir(parents=True, exist_ok=True)
    with progress_log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _persist_snapshot(
    merged: dict[tuple[str, str, str], dict[str, Any]],
    out_triples: Path,
    out_graph: Path,
    out_stats: Path,
    docs_count: int,
    processed_chunks_count: int,
    model: str,
    snapshot_tag: str,
) -> None:
    triples = sorted(
        merged.values(),
        key=lambda x: (x["head"].lower(), x["relation"], x["tail"].lower()),
    )
    triples = _attach_dependency_alternatives(triples)
    out_triples.write_text(json.dumps(triples, indent=2, ensure_ascii=False), encoding="utf-8")

    G = _build_graph(triples)
    with out_graph.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    stats = _graph_stats(G)
    stats["docs"] = docs_count
    stats["chunks"] = processed_chunks_count
    stats["model"] = model
    stats["snapshot"] = snapshot_tag
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[llama3-groq] snapshot={snapshot_tag} "
        f"triples={len(triples)} nodes={stats['nodes']} edges={stats['edges']}"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_triples = Path(args.output_triples)
    out_graph = Path(args.output_graph)
    out_stats = Path(args.output_stats)
    progress_log = Path(args.progress_log) if args.progress_log else _default_progress_log_path(out_triples)
    env_values = _load_env_file(Path(args.env_file))
    api_key = args.api_key or os.getenv("GROQ_API_KEY") or env_values.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit(
            "Missing GROQ_API_KEY. Set it in environment, .env, or pass --api-key."
        )

    out_triples.parent.mkdir(parents=True, exist_ok=True)
    out_graph.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_docs(input_path, max_docs=args.max_docs)
    print(f"[llama3-groq] docs={len(docs)} model={args.model} resume={args.resume}")
    run_started = time.time()

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    processed_chunks: set[str] = set()
    if args.resume:
        processed_chunks, merged = _load_progress_state(progress_log)
        if not merged and out_triples.exists():
            merged = _load_existing_triples(out_triples)
        print(
            f"[llama3-groq] resume loaded progress_chunks={len(processed_chunks)} "
            f"merged_triples={len(merged)} progress_log={progress_log}"
        )
    else:
        if progress_log.exists():
            progress_log.unlink()
        print(f"[llama3-groq] fresh run, progress log reset -> {progress_log}")

    total_chunks = 0  # chunks requested in this run
    for d_idx, doc in enumerate(docs, 1):
        doc_started = time.time()
        source = doc["source"]
        chunks = _chunk_text(
            doc.get("text", ""),
            max_chars=args.chunk_chars,
            overlap=args.chunk_overlap,
        )
        print(
            f"[llama3-groq] doc {d_idx}/{len(docs)} chunks={len(chunks)} source={source}"
        )

        for c_idx, chunk in enumerate(chunks, 1):
            ck = _chunk_key(d_idx, c_idx)
            if ck in processed_chunks:
                continue

            total_chunks += 1
            prompt = _build_prompt(source, c_idx, chunk)
            raw = ""
            resp_headers: requests.structures.CaseInsensitiveDict[str] | None = None
            attempt = 0
            while attempt <= max(0, args.max_retries):
                try:
                    raw, resp_headers = _groq_chat(
                        base_url=args.base_url,
                        api_key=api_key,
                        model=args.model,
                        user_prompt=prompt,
                        timeout_s=args.timeout,
                        service_tier=args.service_tier,
                    )
                    break
                except _RateLimitError as exc:
                    wait = max(
                        exc.wait_seconds,
                        min(args.retry_backoff_max, args.retry_backoff_base * (2 ** attempt)),
                    )
                    print(
                        f"[llama3-groq] 429 doc={d_idx} chunk={c_idx} "
                        f"attempt={attempt + 1}/{args.max_retries + 1} "
                        f"sleep={_format_duration(wait)}"
                    )
                    time.sleep(wait)
                except Exception as exc:
                    wait = min(args.retry_backoff_max, args.retry_backoff_base * (2 ** attempt))
                    print(
                        f"[llama3-groq] retry doc={d_idx} chunk={c_idx} "
                        f"attempt={attempt + 1}/{args.max_retries + 1} "
                        f"sleep={_format_duration(wait)} err={exc}"
                    )
                    time.sleep(wait)
                attempt += 1

            if resp_headers is None:
                print(
                    f"[llama3-groq] warning: failed doc={d_idx} chunk={c_idx} after retries"
                )
                continue

            items = _extract_json_array(raw)
            kept = 0
            response_triples: list[dict[str, Any]] = []
            for item in items:
                validated = _validate_triple(item)
                if not validated:
                    continue
                key = (
                    validated["head"].lower(),
                    validated["relation"],
                    validated["tail"].lower(),
                )
                prev = merged.get(key)
                if prev is None or validated["confidence"] > prev["confidence"]:
                    merged[key] = {
                        **validated,
                        "source": source,
                        "chunk_id": c_idx,
                    }
                response_triples.append(
                    {
                        **validated,
                        "source": source,
                        "chunk_id": c_idx,
                    }
                )
                kept += 1

            processed_chunks.add(ck)
            _append_progress_record(
                progress_log,
                {
                    "ts": int(time.time()),
                    "doc_index": d_idx,
                    "chunk_index": c_idx,
                    "source": source,
                    "raw_items": len(items),
                    "kept": kept,
                    "triples": response_triples,
                },
            )

            print(
                f"[llama3-groq] doc={d_idx} chunk={c_idx}/{len(chunks)} "
                f"raw={len(items)} kept={kept} merged_total={len(merged)} "
                f"done_chunks={len(processed_chunks)}"
            )

            remaining_tokens_s = str(resp_headers.get("x-ratelimit-remaining-tokens", "")).strip()
            reset_tokens_s = str(resp_headers.get("x-ratelimit-reset-tokens", "")).strip()
            try:
                remaining_tokens = int(float(remaining_tokens_s)) if remaining_tokens_s else -1
            except Exception:
                remaining_tokens = -1
            reset_tokens = _parse_reset_seconds(reset_tokens_s)

            if remaining_tokens >= 0 and remaining_tokens < max(0, args.min_remaining_tokens):
                wait = max(0.5, reset_tokens)
                print(
                    f"[llama3-groq] pacing: remaining_tpm={remaining_tokens} "
                    f"reset={_format_duration(wait)} sleep={_format_duration(wait)}"
                )
                time.sleep(wait)

            if args.sleep > 0:
                time.sleep(args.sleep)

        doc_elapsed = time.time() - doc_started
        run_elapsed = time.time() - run_started
        docs_done = d_idx
        docs_left = max(0, len(docs) - docs_done)
        avg_doc_s = run_elapsed / max(1, docs_done)
        eta_s = docs_left * avg_doc_s
        docs_per_min = 60.0 / avg_doc_s if avg_doc_s > 0 else 0.0

        if (
            docs_done % max(1, args.progress_every) == 0
            or docs_done == len(docs)
        ):
            print(
                f"[llama3-groq] progress docs={docs_done}/{len(docs)} "
                f"doc_time={_format_duration(doc_elapsed)} "
                f"avg_doc={_format_duration(avg_doc_s)} "
                f"rate={docs_per_min:.2f} docs/min "
                f"eta={_format_duration(eta_s)} "
                f"chunks={total_chunks} "
                f"triples={len(merged)}"
            )

        if docs_done % max(1, args.save_every_docs) == 0:
            _persist_snapshot(
                merged=merged,
                out_triples=out_triples,
                out_graph=out_graph,
                out_stats=out_stats,
                docs_count=len(docs),
                processed_chunks_count=len(processed_chunks),
                model=args.model,
                snapshot_tag=f"doc_{docs_done}",
            )

    _persist_snapshot(
        merged=merged,
        out_triples=out_triples,
        out_graph=out_graph,
        out_stats=out_stats,
        docs_count=len(docs),
        processed_chunks_count=len(processed_chunks),
        model=args.model,
        snapshot_tag="final",
    )
    final_stats = json.loads(out_stats.read_text(encoding="utf-8"))
    print(f"[llama3-groq] triples saved -> {out_triples}")
    print(f"[llama3-groq] graph saved -> {out_graph}")
    print(f"[llama3-groq] stats saved -> {out_stats}")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
