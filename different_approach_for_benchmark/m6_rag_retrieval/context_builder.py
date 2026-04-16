import re
from typing import Any


_COMPANIES = {
    "tsmc", "taiwan semiconductor", "apple", "amd", "nvidia", "intel",
    "qualcomm", "samsung", "broadcom", "asml", "foxconn", "sk hynix",
    "micron", "texas instruments", "murata", "kyocera", "tdk",
}

_COUNTRIES = {
    "china", "taiwan", "south korea", "korea", "japan", "malaysia",
    "thailand", "vietnam", "singapore", "united states", "usa", "us",
    "germany", "netherlands",
}

_MINERALS = {
    "gallium", "germanium", "cobalt", "lithium", "rare earths", "rare earth",
    "nickel", "copper", "palladium", "platinum", "indium",
}

_PRODUCTS = {
    "iphone", "macbook", "ipad", "gpu", "cpu", "semiconductor", "chip",
    "wafer", "battery", "solar panel", "playstation", "xbox", "galaxy",
    "pixel", "switch", "macpro", "imac",
}

_DISRUPTION_WORDS = re.compile(
    r"\b(stops?|halt(?:s|ed)?|shut(?:s|ting|down)?|bans?|sanctions?|conflict|war|"
    r"earthquake|flood|strikes?|export control|restricts?|disruption|disrupt(?:s|ed|ing)?|"
    r"shortage|crisis|collapse|cut(?:s|ting)?|invade[sd]?|invasion|block(?:s|ed)?|"
    r"attack(?:s|ed)?|sanction(?:s|ed)?)\b",
    re.IGNORECASE,
)


def parse_query(query: str) -> dict:
    q = query.strip()
    q_lower = q.lower()

    entity = _find_entity(q_lower)
    entity_type = _classify_entity(entity) if entity else "unknown"

    has_disruption = bool(_DISRUPTION_WORDS.search(q))

    if entity_type == "company" and has_disruption:
        itype = "company_disruption"
    elif entity_type == "country" and has_disruption:
        itype = "country_disruption"
    elif entity_type == "mineral" and has_disruption:
        itype = "mineral_disruption"
    elif entity_type == "product":
        itype = "product_supply"
    else:
        itype = "general"

    return {"type": itype, "entity": entity, "entity_type": entity_type, "raw": q}


def _find_entity(q_lower: str) -> str:
    best = ""
    for name in sorted(_COMPANIES | _COUNTRIES | _MINERALS | _PRODUCTS, key=len, reverse=True):
        if name in q_lower:
            best = name
            break
    if not best:
        _SKIP_WORDS = {
            "what", "which", "where", "when", "who", "why", "how",
            "if", "is", "are", "the", "a", "an", "and", "or", "of",
            "for", "in", "on", "at", "to", "by", "has", "have",
            "supply", "supplies", "supplier", "suppliers", "companies",
            "company", "components", "component", "disrupted", "disruption",
        }
        words = q_lower.title().split()
        noun_words = []
        candidates = []
        for w in words:
            clean = re.sub(r"[^a-zA-Z0-9]", "", w)
            if clean and clean[0].isupper() and clean.lower() not in _SKIP_WORDS:
                noun_words.append(clean)
            else:
                if noun_words:
                    candidates.append(" ".join(noun_words))
                noun_words = []
        if noun_words:
            candidates.append(" ".join(noun_words))
        if candidates:
            best = max(candidates, key=len)
    return best


def _classify_entity(entity: str) -> str:
    e = entity.lower()
    if e in _COMPANIES:
        return "company"
    if e in _COUNTRIES:
        return "country"
    if e in _MINERALS:
        return "mineral"
    if e in _PRODUCTS:
        return "product"
    return "unknown"


def build_context(subgraph: dict) -> str:
    lines = []
    entity = subgraph.get("query_entity", "")
    itype  = subgraph.get("intent_type", "")

    lines.append(f"=== SUPPLY CHAIN KNOWLEDGE GRAPH CONTEXT ===")
    lines.append(f"Query entity : {entity}")
    lines.append(f"Intent       : {itype}")
    lines.append("")

    supply_edges = subgraph.get("supply_edges", [])
    seen_supply: set = set()
    if supply_edges:
        lines.append("--- SUPPLY RELATIONSHIPS ---")
        for e in supply_edges:
            key = (e["from"], e["to"], e["type"])
            if key in seen_supply:
                continue
            seen_supply.add(key)
            rel = e["type"].replace("_", " ")
            ev  = (e.get("evidence") or "")[:150]
            conf = e.get("confidence", 1.0)
            lines.append(f"  {e['from']}  --[{rel}]-->  {e['to']}  (confidence={conf:.2f})")
            if ev:
                lines.append(f"    Evidence: {ev}")
        lines.append("")

    dep_edges = subgraph.get("depends_edges", [])
    seen_dep: set = set()
    if dep_edges:
        lines.append("--- PRODUCT DEPENDENCIES ---")
        for e in dep_edges:
            key = (e["product"], e["company"])
            if key in seen_dep:
                continue
            seen_dep.add(key)
            ev = (e.get("evidence") or "")[:150]
            lines.append(f"  {e['product']}  depends on  {e['company']}")
            if ev:
                lines.append(f"    Evidence: {ev}")
        lines.append("")

    prod_edges = subgraph.get("produces_edges", [])
    if prod_edges:
        lines.append("--- MINERAL PRODUCTION ---")
        for e in prod_edges:
            prod = e.get("production_mt_2023") or "unknown"
            res  = e.get("reserves_mt") or "unknown"
            lines.append(
                f"  {e['country']}  produces  {e['mineral']}  "
                f"(2023 output: {prod}, reserves: {res})"
            )
        lines.append("")

    companies = subgraph.get("companies", [])
    if companies:
        names = ", ".join(c["name"] for c in companies)
        lines.append(f"--- COMPANIES INVOLVED ---")
        lines.append(f"  {names}")
        lines.append("")

    countries = subgraph.get("countries", [])
    if countries:
        names = ", ".join(c["name"] for c in countries)
        lines.append(f"--- COUNTRIES INVOLVED ---")
        lines.append(f"  {names}")
        lines.append("")

    seen_events: set = set()
    unique_events = []
    for ev in subgraph.get("risk_events", []):
        key = (ev.get("name", ""), ev.get("date", ""), ev.get("affects", ""))
        if key not in seen_events:
            seen_events.add(key)
            unique_events.append(ev)

    risk_events = sorted(unique_events, key=lambda x: float(x.get("goldstein", 0) or 0))
    if risk_events:
        lines.append("--- RECENT RISK EVENTS (most severe first) ---")
        for ev in risk_events[:15]:
            g = ev.get("goldstein", 0)
            lines.append(
                f"  [{ev.get('date', '?')}]  {ev.get('event_label', '?')}  "
                f"affecting {ev.get('affects', '?')}  (Goldstein={g:.1f})"
            )
            name = ev.get("name", "")
            if name:
                lines.append(f"    Event: {name[:120]}")
        lines.append("")

    if len(lines) <= 5:
        lines.append("  [No relevant graph data found for this query]")

    return "\n".join(lines)
