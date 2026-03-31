from __future__ import annotations

import argparse
import json
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


DEFAULT_MODEL = "qwen2.5:32b"
DEFAULT_BASE_URL = "http://localhost:11434"

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
    - affected_by: disruption_event -> impacted entity
    - located_in:
        - disruption_event -> location
        - location -> contained entity (company/supplier/manufacturer/part/port/route/customer)
- Alternative policy (strict):
    - Do NOT mark complementary dependencies as alternatives (example: milk and chocolate used together are not alternatives).
    - Propagation rule: if x `alternative_to` y, then for dependency/supply edge x -> z, include y in x->z alternatives (and symmetrically include x for y -> z).
    - Even if a new alternative for x and y appears , we need to have for each edge list of all alterantives.
- For BFS-friendly geopolitical traversal, prefer country/region as the head node when linking
    non-event entities to locations (for example: Iran -> located_in -> Bandar Abbas Port).
- Geopolitical disruption templates (apply consistently across countries, e.g., Iran, India):
    - If text mentions war/sanctions/blockade/export controls/conflict in place X,
        create a DisruptionEvent node and add disruption_event -> located_in -> X.
    - If text mentions ports, factories, suppliers, routes, or chokepoints in X,
        add those entities and connect X -> located_in -> entity.
    - Model chokepoints (straits, canals, shipping lanes, sea corridors) as LogisticsRoute.
    - If text states closure, delay, rerouting, congestion, insurance spikes, or shipment disruption,
        add affected_by from disruption event to impacted port/route/company/supplier.
    - If text states route structure, connect route/port/region with connects edges.
- Do not use outside knowledge; only use what is present in the given text.
- For location-triggered propagation, emit:
    - disruption_event -> located_in -> location
    - location -> located_in -> company/supplier/port/route/customer/manufacturer/part
    - broader location -> located_in -> narrower location when explicit
- If a disruption is stated in a place ("war in X", "earthquake in Y"),
  model the disruption event node and add disruption_event -> located_in -> place.
- Allowed head_type/tail_type: {sorted(ALLOWED_ENTITY_TYPES)}
- Allowed relation: {sorted(ALLOWED_RELATIONS)}
- If no valid triples, return [].
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract supply-chain triples with local Ollama Qwen2.5 and build a graph."
    )
    parser.add_argument(
        "--input",
        default="data/sp500_company_graph_knowledge.jsonl",
        help="Input JSON/JSONL with text fields.",
    )
    parser.add_argument(
        "--output-triples",
        default="data/processed/sp500_graph_knowledge_triples_qwen25.json",
        help="Output triples JSON path.",
    )
    parser.add_argument(
        "--output-graph",
        default="data/processed/sp500_graph_knowledge_graph_qwen25.gpickle",
        help="Output graph pickle path.",
    )
    parser.add_argument(
        "--output-stats",
        default="data/processed/sp500_graph_knowledge_stats_qwen25.json",
        help="Output stats JSON path.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name (default: qwen2.5:32b).",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Generation top-p.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=1024,
        help="Maximum output tokens per chunk response.",
    )
    parser.add_argument(
        "--keep-alive",
        default="30m",
        help="Ollama model keep_alive value.",
    )
    parser.add_argument(
        "--skip-model-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip Ollama model availability check at startup.",
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
        default=2200,
        help="Approx max chars per chunk sent to LLM.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=300,
        help="Overlap chars between consecutive chunks.",
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
        default=4,
        help="Max retries for transient failures.",
    )
    parser.add_argument(
        "--retry-backoff-base",
        type=float,
        default=2.0,
        help="Base backoff seconds for retries.",
    )
    parser.add_argument(
        "--retry-backoff-max",
        type=float,
        default=60.0,
        help="Max backoff seconds for retries.",
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
        "city": EntityType.REGION.value,
        "country": EntityType.REGION.value,
        "province": EntityType.REGION.value,
        "state": EntityType.REGION.value,
        "location": EntityType.REGION.value,
        "place": EntityType.REGION.value,
        "gpe": EntityType.REGION.value,
        "region": EntityType.REGION.value,
        "route": EntityType.LOGISTICS_ROUTE.value,
        "shippinglane": EntityType.LOGISTICS_ROUTE.value,
        "shippingroute": EntityType.LOGISTICS_ROUTE.value,
        "searoute": EntityType.LOGISTICS_ROUTE.value,
        "waterway": EntityType.LOGISTICS_ROUTE.value,
        "strait": EntityType.LOGISTICS_ROUTE.value,
        "canal": EntityType.LOGISTICS_ROUTE.value,
        "corridor": EntityType.LOGISTICS_ROUTE.value,
        "logisticsroute": EntityType.LOGISTICS_ROUTE.value,
        "event": EntityType.DISRUPTION.value,
        "incident": EntityType.DISRUPTION.value,
        "conflict": EntityType.DISRUPTION.value,
        "war": EntityType.DISRUPTION.value,
        "sanction": EntityType.DISRUPTION.value,
        "blockade": EntityType.DISRUPTION.value,
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


def _flip_triple_direction(triple: dict[str, Any]) -> dict[str, Any]:
    return {
        "head": triple["tail"],
        "head_type": triple["tail_type"],
        "relation": triple["relation"],
        "tail": triple["head"],
        "tail_type": triple["head_type"],
        "confidence": triple["confidence"],
        "evidence": triple.get("evidence", ""),
    }


def _normalize_located_in_direction(triple: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize located_in edge direction for geopolitical BFS:
    - disruption_event -> location
    - location -> non-location entity
    - for location <-> location hierarchy, keep both directions to preserve reachability
    """
    if triple.get("relation") != RelationType.LOCATED_IN.value:
        return [triple]

    head_type = str(triple.get("head_type", ""))
    tail_type = str(triple.get("tail_type", ""))
    head_is_region = head_type == EntityType.REGION.value
    tail_is_region = tail_type == EntityType.REGION.value
    head_is_disruption = head_type == EntityType.DISRUPTION.value
    tail_is_disruption = tail_type == EntityType.DISRUPTION.value

    if head_is_disruption and tail_is_region:
        return [triple]

    if tail_is_disruption and head_is_region:
        return [_flip_triple_direction(triple)]

    if head_is_region and not tail_is_region:
        return [triple]

    if tail_is_region and not head_is_region:
        return [_flip_triple_direction(triple)]

    if head_is_region and tail_is_region:
        flipped = _flip_triple_direction(triple)
        return [triple, flipped]

    return [triple]


def _normalize_affected_by_direction(triple: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize affected_by edge direction for event-first directed BFS:
    - disruption_event -> impacted entity
    """
    if triple.get("relation") != RelationType.AFFECTED_BY.value:
        return [triple]

    head_type = str(triple.get("head_type", ""))
    tail_type = str(triple.get("tail_type", ""))
    head_is_disruption = head_type == EntityType.DISRUPTION.value
    tail_is_disruption = tail_type == EntityType.DISRUPTION.value

    if head_is_disruption and not tail_is_disruption:
        return [triple]

    if tail_is_disruption and not head_is_disruption:
        return [_flip_triple_direction(triple)]

    return [triple]


def _normalize_direction_for_bfs(triple: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in _normalize_located_in_direction(triple):
        normalized.extend(_normalize_affected_by_direction(item))
    return normalized


def _normalize_model_name(name: str) -> str:
    return str(name or "").strip().lower()


def _assert_ollama_ready(base_url: str, model: str, timeout_s: int) -> None:
    version_url = f"{base_url.rstrip('/')}/api/version"
    tags_url = f"{base_url.rstrip('/')}/api/tags"

    try:
        version_resp = requests.get(version_url, timeout=timeout_s)
        version_resp.raise_for_status()
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {base_url}. Start Ollama and retry. Details: {exc}"
        )

    try:
        tags_resp = requests.get(tags_url, timeout=timeout_s)
        tags_resp.raise_for_status()
        payload = tags_resp.json()
    except Exception as exc:
        raise SystemExit(
            f"Ollama is reachable but model listing failed at {tags_url}. Details: {exc}"
        )

    models: list[str] = []
    for item in payload.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model")
        if isinstance(name, str) and name.strip():
            models.append(name.strip())

    requested = _normalize_model_name(model)
    available = {_normalize_model_name(m) for m in models}
    available_bases = {m.split(":", 1)[0] for m in available}
    requested_base = requested.split(":", 1)[0]

    if requested not in available and requested_base not in available_bases:
        sample = ", ".join(models[:8]) if models else "none"
        raise SystemExit(
            f"Model '{model}' not found in Ollama. Available models: {sample}. "
            f"Run: ollama pull {model}"
        )


def _ollama_chat(
    base_url: str,
    model: str,
    user_prompt: str,
    timeout_s: int,
    temperature: float,
    top_p: float,
    num_predict: int,
    keep_alive: str,
) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"

    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
    }
    if num_predict > 0:
        options["num_predict"] = num_predict

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": options,
        "keep_alive": keep_alive,
    }

    resp = requests.post(url, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama error: {data.get('error')}")

    message = data.get("message", {}) if isinstance(data, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return str(content).strip()


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


def _build_alternative_index(
    triples: list[dict[str, Any]],
) -> dict[str, set[str]]:
    """
    Build connected components of explicit alternative_to statements.

    Returns:
      lowercased entity name -> set of canonical entity names in same alt component
    """
    alt_graph = nx.Graph()
    canonical_name: dict[str, str] = {}

    for t in triples:
        if str(t.get("relation", "")) != RelationType.ALTERNATIVE_TO.value:
            continue

        left = _clean_name(t.get("head"))
        right = _clean_name(t.get("tail"))
        if not left or not right:
            continue
        if left.lower() == right.lower():
            continue

        left_l = left.lower()
        right_l = right.lower()
        canonical_name.setdefault(left_l, left)
        canonical_name.setdefault(right_l, right)
        alt_graph.add_edge(left_l, right_l)

    alt_index: dict[str, set[str]] = {}
    for component in nx.connected_components(alt_graph):
        members = {canonical_name.get(node_l, node_l) for node_l in component}
        member_lowers = {x.lower() for x in members}
        for node_l in member_lowers:
            alt_index[node_l] = set(members)

    return alt_index


def _attach_dependency_alternatives(
    triples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add OR-choice metadata on dependency edges.

    Policy:
    - Alternatives are taken from extracted `alternative_to` links.
    - For each dependency edge x->z (`supplies` head->tail or `depends_on` head->tail),
      every alternative y of x is attached to that edge as an OR option, even if y->z
      is not explicitly present.
    - This makes event-time reasoning easier: if x fails, y is discoverable directly
      from the affected edge metadata.

    Each eligible edge gets:
    - option_group_id
    - alternatives: list of all other options in the same group
    - selection_mode: "OR"
    """
    alt_index = _build_alternative_index(triples)
    for t in triples:
        rel = str(t.get("relation", ""))
        if rel == RelationType.SUPPLIES.value:
            slot_entity = str(t.get("tail", ""))
            current_option = str(t.get("head", ""))
        elif rel == RelationType.DEPENDS_ON.value:
            slot_entity = str(t.get("head", ""))
            current_option = str(t.get("tail", ""))
        else:
            continue

        explicit_alts = alt_index.get(current_option.lower(), set())
        alternatives = sorted(
            [
                x
                for x in explicit_alts
                if x.lower() != current_option.lower()
            ],
            key=lambda x: x.lower(),
        )

        if not alternatives:
            t.pop("option_group_id", None)
            t.pop("selection_mode", None)
            t.pop("alternatives", None)
            continue

        group_members = sorted([current_option, *alternatives], key=lambda x: x.lower())
        t["option_group_id"] = f"{rel}:{slot_entity}:{'|'.join(group_members)}"
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
        normalized = _normalize_direction_for_bfs(validated)
        for triple in normalized:
            key = (
                triple["head"].lower(),
                triple["relation"],
                triple["tail"].lower(),
            )
            prev = merged.get(key)
            if prev is None or triple["confidence"] > prev["confidence"]:
                merged[key] = {
                    **triple,
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
                normalized = _normalize_direction_for_bfs(validated)
                for triple in normalized:
                    key = (
                        triple["head"].lower(),
                        triple["relation"],
                        triple["tail"].lower(),
                    )
                    prev = merged.get(key)
                    if prev is None or triple["confidence"] > prev["confidence"]:
                        merged[key] = {
                            **triple,
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
        f"[qwen25-ollama] snapshot={snapshot_tag} "
        f"triples={len(triples)} nodes={stats['nodes']} edges={stats['edges']}"
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    out_triples = Path(args.output_triples)
    out_graph = Path(args.output_graph)
    out_stats = Path(args.output_stats)
    progress_log = Path(args.progress_log) if args.progress_log else _default_progress_log_path(out_triples)

    if not args.skip_model_check:
        _assert_ollama_ready(args.base_url, args.model, timeout_s=max(5, args.timeout))

    out_triples.parent.mkdir(parents=True, exist_ok=True)
    out_graph.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    progress_log.parent.mkdir(parents=True, exist_ok=True)

    docs = _load_docs(input_path, max_docs=args.max_docs)
    run_started = time.time()
    print(
        f"[qwen25-ollama] run_start docs={len(docs)} model={args.model} "
        f"input={input_path} resume={args.resume}"
    )

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    processed_chunks: set[str] = set()
    initial_processed_chunks = 0
    if args.resume:
        processed_chunks, merged = _load_progress_state(progress_log)
        if not merged and out_triples.exists():
            merged = _load_existing_triples(out_triples)
        print(
            f"[qwen25-ollama] resume loaded progress_chunks={len(processed_chunks)} "
            f"merged_triples={len(merged)} progress_log={progress_log}"
        )
        initial_processed_chunks = len(processed_chunks)
    else:
        if progress_log.exists():
            progress_log.unlink()
        print(f"[qwen25-ollama] fresh run, progress log reset -> {progress_log}")
        initial_processed_chunks = 0

    skipped_chunks = 0
    failed_chunks = 0
    new_chunks_done = 0

    total_chunks = 0
    for d_idx, doc in enumerate(docs, 1):
        doc_started = time.time()
        source = doc["source"]
        chunks = _chunk_text(
            doc.get("text", ""),
            max_chars=args.chunk_chars,
            overlap=args.chunk_overlap,
        )
        run_elapsed = time.time() - run_started
        print(
            f"[qwen25-ollama] doc_start doc={d_idx}/{len(docs)} "
            f"chunks={len(chunks)} source={source} elapsed={_format_duration(run_elapsed)}"
        )

        for c_idx, chunk in enumerate(chunks, 1):
            ck = _chunk_key(d_idx, c_idx)
            if ck in processed_chunks:
                skipped_chunks += 1
                if skipped_chunks <= 5 or skipped_chunks % 25 == 0:
                    print(
                        f"[qwen25-ollama] chunk_skip doc={d_idx} chunk={c_idx}/{len(chunks)} "
                        f"reason=resume skipped_total={skipped_chunks}"
                    )
                continue

            total_chunks += 1
            chunk_started = time.time()
            print(
                f"[qwen25-ollama] chunk_start doc={d_idx} chunk={c_idx}/{len(chunks)} "
                f"chars={len(chunk)} merged_before={len(merged)}"
            )
            prompt = _build_prompt(source, c_idx, chunk)
            raw = ""
            attempt = 0
            while attempt <= max(0, args.max_retries):
                try:
                    raw = _ollama_chat(
                        base_url=args.base_url,
                        model=args.model,
                        user_prompt=prompt,
                        timeout_s=args.timeout,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        num_predict=args.num_predict,
                        keep_alive=args.keep_alive,
                    )
                    break
                except Exception as exc:
                    wait = min(args.retry_backoff_max, args.retry_backoff_base * (2 ** attempt))
                    print(
                        f"[qwen25-ollama] retry doc={d_idx} chunk={c_idx} "
                        f"attempt={attempt + 1}/{args.max_retries + 1} "
                        f"sleep={_format_duration(wait)} err={exc}"
                    )
                    time.sleep(wait)
                attempt += 1

            if not raw:
                failed_chunks += 1
                print(
                    f"[qwen25-ollama] warning: empty response for doc={d_idx} chunk={c_idx} "
                    f"failed_chunks={failed_chunks}"
                )

            items = _extract_json_array(raw)
            kept = 0
            response_triples: list[dict[str, Any]] = []
            for item in items:
                validated = _validate_triple(item)
                if not validated:
                    continue
                normalized = _normalize_direction_for_bfs(validated)
                for triple in normalized:
                    key = (
                        triple["head"].lower(),
                        triple["relation"],
                        triple["tail"].lower(),
                    )
                    prev = merged.get(key)
                    if prev is None or triple["confidence"] > prev["confidence"]:
                        merged[key] = {
                            **triple,
                            "source": source,
                            "chunk_id": c_idx,
                        }
                    response_triples.append(
                        {
                            **triple,
                            "source": source,
                            "chunk_id": c_idx,
                        }
                    )
                    kept += 1

            processed_chunks.add(ck)
            new_chunks_done = max(0, len(processed_chunks) - initial_processed_chunks)
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

            chunk_elapsed = time.time() - chunk_started
            run_elapsed = time.time() - run_started
            chunks_per_min = (60.0 * new_chunks_done / run_elapsed) if run_elapsed > 0 else 0.0
            print(
                f"[qwen25-ollama] doc={d_idx} chunk={c_idx}/{len(chunks)} "
                f"raw={len(items)} kept={kept} merged_total={len(merged)} "
                f"done_chunks_total={len(processed_chunks)} new_chunks={new_chunks_done} "
                f"chunk_time={_format_duration(chunk_elapsed)} "
                f"elapsed={_format_duration(run_elapsed)} rate={chunks_per_min:.2f} chunks/min "
                f"skipped={skipped_chunks} failed={failed_chunks}"
            )

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
                f"[qwen25-ollama] progress docs={docs_done}/{len(docs)} "
                f"doc_time={_format_duration(doc_elapsed)} "
                f"avg_doc={_format_duration(avg_doc_s)} "
                f"rate={docs_per_min:.2f} docs/min "
                f"eta={_format_duration(eta_s)} "
                f"chunks_started={total_chunks} new_chunks={new_chunks_done} "
                f"skipped={skipped_chunks} failed={failed_chunks} "
                f"triples={len(merged)} elapsed={_format_duration(run_elapsed)}"
            )

        print(
            f"[qwen25-ollama] doc_done doc={docs_done}/{len(docs)} "
            f"source={source} doc_time={_format_duration(doc_elapsed)} "
            f"elapsed={_format_duration(run_elapsed)} triples={len(merged)}"
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
    total_elapsed = time.time() - run_started
    print(f"[qwen25-ollama] triples saved -> {out_triples}")
    print(f"[qwen25-ollama] graph saved -> {out_graph}")
    print(f"[qwen25-ollama] stats saved -> {out_stats}")
    print(
        f"[qwen25-ollama] run_summary elapsed={_format_duration(total_elapsed)} "
        f"docs={len(docs)} new_chunks={new_chunks_done} "
        f"resume_skips={skipped_chunks} failed_chunks={failed_chunks} "
        f"triples={len(merged)}"
    )
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    main()
