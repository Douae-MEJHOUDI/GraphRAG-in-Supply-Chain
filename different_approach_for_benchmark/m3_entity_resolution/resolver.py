import json
import logging
import re
import uuid
from pathlib import Path

from rapidfuzz import fuzz, process as fuzz_process

from m3_entity_resolution.aliases import COMPANY_LOOKUP, COUNTRY_LOOKUP

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
CANONICAL_DIR = Path("data/canonical")

FUZZY_THRESHOLD = {
    "company": 88,
    "country": 92,
    "mineral": 90,
    "product": 85,
    "risk_event": 100,
}

_LEGAL_SUFFIXES = re.compile(
    r"\s+(inc\.?|corp\.?|co\.?|ltd\.?|llc\.?|l\.p\.?|plc\.?|n\.v\.?|ag\.?|"
    r"gmbh\.?|s\.a\.?|holdings?|group|limited|incorporated|corporation|company)\.?$",
    re.IGNORECASE,
)

_LIST_FRAGMENT = re.compile(r"^[a-z]\.\s+\w", re.IGNORECASE)
_NOISE_NAME    = re.compile(r"^[^a-zA-Z]+")

_NOT_COMPANIES = {
    "healthcare", "third-party", "third party", "inventory", "licensing",
    "research", "development", "operations", "services", "solutions",
    "management", "technology", "technologies", "systems", "network",
    "networks", "capital", "financial", "insurance", "logistics",
    "manufacturing", "assembly", "production", "distribution",
    "government", "authority", "ministry", "department", "agency",
    "committee", "commission", "board", "council", "union",
    "the court", "court", "congress", "senate", "parliament",
    "international", "global", "worldwide", "domestic",
    "first", "second", "third", "fourth", "north", "south", "east", "west",
    "adss", "ai nic", "asic", "atmp", "bbb", "bbb+", "bbb-",
    "supplier trust", "purchase obligations", "purchase obligation",
    "deemed repatriation tax payable", "supply concentrations",
    "unconditional purchase obligations", "the audit and finance committee",
}


def _valid_canonical_name(name: str) -> bool:
    name = name.strip()
    if len(name) < 3:
        return False
    if _LIST_FRAGMENT.match(name):
        return False
    if _NOISE_NAME.match(name):
        return False
    if name.lower() in _NOT_COMPANIES:
        return False
    if name.startswith("http") or name.startswith("link|"):
        return False
    if len(name) <= 4 and name.isupper():
        return False
    alpha = sum(c.isalpha() for c in name)
    if alpha / len(name) < 0.4:
        return False
    return True


def normalize(name: str) -> str:
    name = name.strip()
    name = _LEGAL_SUFFIXES.sub("", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


class EntityResolver:

    def __init__(self):
        self.registry: dict[str, dict] = {}
        self._norm_index: dict[str, str] = {}
        self.ambiguous: list[dict] = []

    def resolve_all(self) -> tuple[list[dict], list[dict]]:
        all_entities      = []
        all_relationships = []

        for source_dir in sorted(PROCESSED_DIR.iterdir()):
            if not source_dir.is_dir():
                continue
            source = source_dir.name

            ent_file = source_dir / "entities.json"
            rel_file = source_dir / "relationships.json"

            if ent_file.exists():
                with open(ent_file, encoding="utf-8") as f:
                    ents = json.load(f)
                logger.info(f"[M3] Loaded {len(ents)} entities from {source}")
                all_entities.extend(ents)

            if rel_file.exists():
                with open(rel_file, encoding="utf-8") as f:
                    rels = json.load(f)
                logger.info(f"[M3] Loaded {len(rels)} relationships from {source}")
                all_relationships.extend(rels)

        logger.info(f"[M3] Total input: {len(all_entities)} entities, {len(all_relationships)} relationships")

        for ent in all_entities:
            self._resolve_entity(ent["name"], ent["type"], ent.get("source", ""), ent.get("properties", {}))

        logger.info(f"[M3] Registry: {len(self.registry)} canonical entities")

        resolved_rels = []
        skipped = 0
        for rel in all_relationships:
            src = self._get_canonical_name(rel["source_entity"])
            tgt = self._get_canonical_name(rel["target_entity"])

            if src is None or tgt is None:
                skipped += 1
                continue
            if src == tgt:
                skipped += 1
                continue

            resolved_rels.append({
                **rel,
                "source_entity": src,
                "target_entity": tgt,
            })

        logger.info(f"[M3] Relationships: {len(resolved_rels)} resolved, {skipped} skipped (unresolvable or self-loops)")
        logger.info(f"[M3] Ambiguous: {len(self.ambiguous)} entities flagged for review")

        canonical_list = list(self.registry.values())
        return canonical_list, resolved_rels

    def _resolve_entity(self, name: str, etype: str, source: str, properties: dict) -> str | None:
        if not _valid_canonical_name(name):
            return None
        canonical_name = self._lookup(name, etype)

        if canonical_name and canonical_name in self.registry:
            cid = self.registry[canonical_name]["id"]
            if name not in self.registry[canonical_name]["aliases"]:
                self.registry[canonical_name]["aliases"].append(name)
            if source and source not in self.registry[canonical_name]["sources"]:
                self.registry[canonical_name]["sources"].append(source)
            return cid

        cid = str(uuid.uuid4())
        display_name = canonical_name or name
        self.registry[display_name] = {
            "id":         cid,
            "name":       display_name,
            "type":       etype,
            "aliases":    [name] if name != display_name else [],
            "sources":    [source] if source else [],
            "properties": properties,
        }
        norm = normalize(display_name)
        self._norm_index[norm] = display_name
        return cid

    def _lookup(self, name: str, etype: str) -> str | None:
        name_lower = name.lower().strip()

        if etype == "company" and name_lower in COMPANY_LOOKUP:
            return COMPANY_LOOKUP[name_lower]
        if etype == "country" and name_lower in COUNTRY_LOOKUP:
            return COUNTRY_LOOKUP[name_lower]

        norm = normalize(name)
        if norm in self._norm_index:
            return self._norm_index[norm]

        threshold = FUZZY_THRESHOLD.get(etype, 90)
        if threshold == 100:
            return None

        candidates = [
            canon for canon, data in self.registry.items()
            if data["type"] == etype
        ]
        if not candidates:
            return None

        match = fuzz_process.extractOne(
            norm,
            [normalize(c) for c in candidates],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )

        if match:
            matched_norm, score, idx = match
            canonical = candidates[idx]
            logger.debug(f"[M3] Fuzzy: '{name}' → '{canonical}' (score={score:.0f})")
            return canonical

        if etype in ("company", "country", "mineral"):
            try:
                from m6_rag_retrieval.vector_store import get_store
                result = get_store().find_entity(name, etype, threshold=0.82)
                if result and result in self.registry:
                    logger.debug(f"[M3] Vector: '{name}' → '{result}'")
                    return result
            except Exception:
                pass

        return None

    def _get_canonical_name(self, name: str) -> str | None:
        if name in self.registry:
            return name

        for canon, data in self.registry.items():
            if name in data.get("aliases", []):
                return canon

        norm = normalize(name)
        if norm in self._norm_index:
            return self._norm_index[norm]

        candidates = list(self.registry.keys())
        if not candidates:
            return None

        match = fuzz_process.extractOne(
            norm,
            [normalize(c) for c in candidates],
            scorer=fuzz.token_sort_ratio,
            score_cutoff=85,
        )
        if match:
            _, score, idx = match
            return candidates[idx]

        return None
