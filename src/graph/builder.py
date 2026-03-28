"""
builder.py
----------
KnowledgeGraphBuilder — the core of Module 1.

Orchestrates the full pipeline from raw text documents to a populated,
validated NetworkX DiGraph:

  Raw text
    → NER (spaCy + rule-based SC patterns)
    → Triple extraction (template matching + LLM zero-shot)
    → Entity disambiguation
    → Graph construction
    → Edge weight assignment
    → Validation & statistics

The graph is the single source of truth for all downstream modules.
Every node and edge carries typed attributes defined in schema.py.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import networkx as nx
import spacy

from src.graph.schema import (
    Entity,
    EntityType,
    RelationType,
    Triple,
    NODE_ATTR_TYPE,
    NODE_ATTR_ALIASES,
    NODE_ATTR_METADATA,
    EDGE_ATTR_RELATION,
    EDGE_ATTR_WEIGHT,
    EDGE_ATTR_CONF,
    EDGE_ATTR_SOURCE,
)
from src.graph.disambiguator import EntityDisambiguator


# ---------------------------------------------------------------------------
# spaCy model — lazy-loaded so the class can be instantiated without it
# ---------------------------------------------------------------------------

_SPACY_MODEL: Optional[spacy.Language] = None

def _get_spacy() -> spacy.Language:
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        try:
            _SPACY_MODEL = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _SPACY_MODEL


# ---------------------------------------------------------------------------
# Canonical entity whitelist
#
# These entities have an unambiguous type that must not be overridden by
# sentence-level signal matching. Without this, a sentence like
# "TSMC supplies advanced chips..." causes "TSMC" to be typed as PART
# because "chip" appears in the same sentence.
#
# Priority order in _classify_entity_type:
#   1. This whitelist  (always wins)
#   2. spaCy label rules (GPE → Region, EVENT → Disruption)
#   3. Entity-name keyword check  (entity name itself contains "port", etc.)
#   4. ORG default → Supplier
# ---------------------------------------------------------------------------

CANONICAL_ENTITY_TYPES: dict[str, EntityType] = {
    # Semiconductor
    "TSMC":                               EntityType.SUPPLIER,
    "Taiwan Semiconductor Manufacturing": EntityType.SUPPLIER,
    "ASML":                               EntityType.SUPPLIER,
    "GlobalFoundries":                    EntityType.SUPPLIER,
    "Samsung Foundry":                    EntityType.SUPPLIER,
    "Samsung":                            EntityType.SUPPLIER,
    "Intel":                              EntityType.SUPPLIER,
    # EV / Consumer electronics manufacturers
    "Tesla":                              EntityType.MANUFACTURER,
    "Apple":                              EntityType.MANUFACTURER,
    "BMW":                                EntityType.MANUFACTURER,
    "Volkswagen":                         EntityType.MANUFACTURER,
    "Audi":                               EntityType.MANUFACTURER,
    "Ford":                               EntityType.MANUFACTURER,
    "Foxconn":                            EntityType.MANUFACTURER,
    # Battery & component suppliers
    "CATL":                               EntityType.SUPPLIER,
    "Panasonic Energy":                   EntityType.SUPPLIER,
    "Panasonic":                          EntityType.SUPPLIER,
    "LG Energy":                          EntityType.SUPPLIER,
    "POSCO":                              EntityType.SUPPLIER,
    "Glencore":                           EntityType.SUPPLIER,
    "Ganfeng Lithium":                    EntityType.SUPPLIER,
    "Pilbara Minerals":                   EntityType.SUPPLIER,
    # Logistics
    "Maersk":                             EntityType.SUPPLIER,
    # Chip / component products (not companies)
    "AMD":                                EntityType.PART,
    "NVIDIA":                             EntityType.PART,
}

# ---------------------------------------------------------------------------
# Entity-name keyword sets used in _classify_entity_type
# These fire on the entity's OWN name, not the surrounding sentence.
# This correctly types "Port of Shanghai" as Port without contaminating
# every other entity in the same sentence.
# ---------------------------------------------------------------------------

_PORT_KEYWORDS       = {"port", "harbor", "harbour", "terminal", "hub"}
_ROUTE_KEYWORDS      = {"route", "lane", "corridor", "canal", "strait", "passage"}
_DISRUPTION_KEYWORDS = {
    "earthquake", "strike", "flood", "fire", "closure", "sanction",
    "shortage", "disruption", "lockdown", "outage", "explosion",
    "hurricane", "typhoon", "tsunami", "pandemic",
}

# Context signals used ONLY to distinguish Manufacturer from generic Supplier
# for ORG spans not in the whitelist.
_MANUFACTURER_CONTEXT_SIGNALS = {
    "assembles", "manufactures", "factory", "plant",
    "production facility", "makes electric", "designs consumer",
}

# spaCy labels we accept for entity extraction
_ACCEPTED_LABELS = {"ORG", "GPE", "FAC", "PRODUCT", "EVENT", "LOC", "NORP"}


# ---------------------------------------------------------------------------
# Relation trigger phrases
# ---------------------------------------------------------------------------

RELATION_PATTERNS: list[tuple[re.Pattern, RelationType]] = [
    (re.compile(r"suppl(?:ies|y|ied|ier)", re.I),   RelationType.SUPPLIES),
    (re.compile(r"source[sd]? from",        re.I),   RelationType.SUPPLIES),
    (re.compile(r"depend[s]? on",           re.I),   RelationType.DEPENDS_ON),
    (re.compile(r"require[sd]?",            re.I),   RelationType.DEPENDS_ON),
    (re.compile(r"ships? through",          re.I),   RelationType.SHIPS_THROUGH),
    (re.compile(r"routes? through",         re.I),   RelationType.SHIPS_THROUGH),
    (re.compile(r"located in",              re.I),   RelationType.LOCATED_IN),
    (re.compile(r"based in",               re.I),   RelationType.LOCATED_IN),
    (re.compile(r"headquartered in",        re.I),   RelationType.LOCATED_IN),
    (re.compile(r"affect(?:ed|s)? by",     re.I),   RelationType.AFFECTED_BY),
    (re.compile(r"impacted by",             re.I),   RelationType.AFFECTED_BY),
    (re.compile(r"alternative(?:ly)?",     re.I),   RelationType.ALTERNATIVE_TO),
    (re.compile(r"sell[s]? to",            re.I),   RelationType.SELLS_TO),
    (re.compile(r"deliver[s]? to",         re.I),   RelationType.SELLS_TO),
]


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------

class KnowledgeGraphBuilder:
    """
    Builds a supply chain knowledge graph from raw text documents.

    Workflow
    --------
    1. load_documents(path)        — read raw text/JSON files
    2. extract_entities()          — NER to find SC entities
    3. extract_triples()           — find (head, rel, tail) facts
    4. disambiguate()              — merge aliases to canonical names
    5. build_graph()               — construct NetworkX DiGraph
    6. assign_edge_weights()       — criticality-aware weighting

    Alternatively: run_pipeline(path) does all six steps in order.

    Parameters
    ----------
    spacy_model : str
        spaCy model name. Defaults to "en_core_web_sm".
    confidence_threshold : float
        Minimum triple confidence to include in the graph. Default 0.5.
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.5,
    ):
        self._spacy_model_name    = spacy_model
        self.confidence_threshold = confidence_threshold
        # Lazy init so graph construction can still run in restricted environments
        # where sentence-transformer model downloads are unavailable.
        self.disambiguator: Optional[EntityDisambiguator] = None

        self.documents:           list[dict]   = []
        self.raw_entities:        list[Entity] = []
        self.canonical_entities:  list[Entity] = []
        self.triples:             list[Triple] = []
        self.graph:               nx.DiGraph   = nx.DiGraph()

    # ------------------------------------------------------------------
    # Step 1 — Load documents
    # ------------------------------------------------------------------

    def load_documents(self, data: list[dict] | str | Path) -> "KnowledgeGraphBuilder":
        """
        Load raw text documents for NLP processing.

        Accepts either:
          - A list of dicts: [{"text": "...", "source": "..."}, ...]
          - A path to a directory containing .txt or .json files
          - A path to a single .json file with a list of documents

        Returns self for method chaining.
        """
        if isinstance(data, (str, Path)):
            data = Path(data)
            if data.is_dir():
                docs = []
                for f in sorted(data.glob("*.txt")):
                    docs.append({"text": f.read_text(encoding="utf-8"),
                                 "source": str(f)})
                for f in sorted(data.glob("*.json")):
                    loaded = json.loads(f.read_text(encoding="utf-8"))
                    if isinstance(loaded, list):
                        docs.extend(loaded)
                    else:
                        docs.append(loaded)
                self.documents = docs
            elif data.suffix == ".json":
                self.documents = json.loads(data.read_text(encoding="utf-8"))
            else:
                self.documents = [{"text": data.read_text(encoding="utf-8"),
                                   "source": str(data)}]
        else:
            self.documents = data

        print(f"[Builder] Loaded {len(self.documents)} document(s)")
        return self

    # ------------------------------------------------------------------
    # Step 2 — Entity extraction
    # ------------------------------------------------------------------

    def extract_entities(self) -> "KnowledgeGraphBuilder":
        """
        Run NER over all documents to find SC-specific entities.

        Uses spaCy for base entity detection then applies a three-tier
        classification strategy:
          1. Canonical whitelist  — known entities always get the right type
          2. spaCy label rules    — GPE/LOC → Region, EVENT → Disruption
          3. Entity-name keywords — "Port of X" → Port, regardless of context
          4. ORG default          — any unmatched ORG → Supplier

        Returns self for method chaining.
        """
        nlp = _get_spacy()
        raw: list[Entity] = []

        for doc_dict in self.documents:
            text   = doc_dict.get("text", "")
            source = doc_dict.get("source", "")
            spacy_doc = nlp(text)

            for sent in spacy_doc.sents:
                sent_text = sent.text.lower()
                sent_raw  = sent.text

                for ent in sent.ents:
                    if ent.label_ not in _ACCEPTED_LABELS:
                        continue

                    etype = self._classify_entity_type(
                        ent.label_, sent_text, ent.text
                    )
                    if etype is None:
                        continue

                    raw.append(Entity(
                        name=ent.text,
                        entity_type=etype,
                        source_text=sent_raw,
                        metadata={"spacy_label": ent.label_, "source": source},
                    ))

        self.raw_entities = raw
        print(f"[Builder] Extracted {len(raw)} raw entities")
        return self

    def _classify_entity_type(
        self,
        spacy_label: str,
        context:     str,   # full sentence, lowercased
        entity_name: str,   # the entity span text as extracted by spaCy
    ) -> Optional[EntityType]:
        """
        Map a spaCy label + entity name + sentence context to a SC EntityType.

        Priority (first match wins):
          1. Canonical whitelist   — exact name lookup, always correct
          2. spaCy label rules     — GPE/LOC/NORP → Region, EVENT → Disruption
          3. Entity-name keywords  — entity's own text contains "port", "earthquake", etc.
             Fires on the ENTITY NAME, not the surrounding sentence, which prevents
             "TSMC" from being typed as PART just because "chip" appears nearby.
          4. Context signal        — only used to distinguish Manufacturer from
             Supplier for ORG spans not in the whitelist.
          5. ORG default           — Supplier (safe fallback for unknown companies)
        """
        # 1. Canonical whitelist
        if entity_name in CANONICAL_ENTITY_TYPES:
            return CANONICAL_ENTITY_TYPES[entity_name]

        # 2. spaCy label rules
        if spacy_label in {"GPE", "LOC", "NORP"}:
            return EntityType.REGION

        if spacy_label == "EVENT":
            return EntityType.DISRUPTION

        # 3. Entity-name keyword checks
        name_lower = entity_name.lower()

        if any(kw in name_lower for kw in _DISRUPTION_KEYWORDS):
            return EntityType.DISRUPTION

        if any(kw in name_lower for kw in _PORT_KEYWORDS):
            return EntityType.PORT

        if any(kw in name_lower for kw in _ROUTE_KEYWORDS):
            return EntityType.LOGISTICS_ROUTE

        # 4. Context signal — Manufacturer vs Supplier (ORG only)
        if spacy_label in {"ORG", "FAC"}:
            if any(sig in context for sig in _MANUFACTURER_CONTEXT_SIGNALS):
                return EntityType.MANUFACTURER
            return EntityType.SUPPLIER

        # 5. PRODUCT label → Part
        if spacy_label == "PRODUCT":
            return EntityType.PART

        return None

    # ------------------------------------------------------------------
    # Step 3 — Triple extraction
    # ------------------------------------------------------------------

    def extract_triples(self) -> "KnowledgeGraphBuilder":
        """
        Extract (head, relation, tail) triples from document sentences.

        Strategy: for each sentence, find two or more entities, then check
        whether the text between (or around) them matches a relation pattern.
        Confidence is a simple heuristic based on pattern specificity.

        Returns self for method chaining.
        """
        if not self.raw_entities:
            self.extract_entities()

        nlp = _get_spacy()
        triples: list[Triple] = []

        for doc_dict in self.documents:
            text      = doc_dict.get("text", "")
            spacy_doc = nlp(text)

            for sent in spacy_doc.sents:
                sent_ents = [
                    e for e in sent.ents
                    if e.label_ in _ACCEPTED_LABELS
                ]
                if len(sent_ents) < 2:
                    continue

                sent_text = sent.text

                for i, head_ent in enumerate(sent_ents):
                    for tail_ent in sent_ents[i + 1:]:
                        triple = self._match_relation(
                            head_ent.text, tail_ent.text, sent_text
                        )
                        if triple is not None:
                            triples.append(triple)

        self.triples = [
            t for t in triples
            if t.confidence >= self.confidence_threshold
        ]
        print(
            f"[Builder] Extracted {len(self.triples)} triples "
            f"(threshold >= {self.confidence_threshold})"
        )
        return self

    def _match_relation(
        self, head: str, tail: str, sentence: str
    ) -> Optional[Triple]:
        """
        Scan the sentence for a relation pattern between head and tail.
        Returns the highest-confidence matching Triple, or None.
        """
        best_triple = None
        best_conf   = 0.0

        for pattern, rel_type in RELATION_PATTERNS:
            if not pattern.search(sentence):
                continue

            head_pos = sentence.lower().find(head.lower())
            tail_pos = sentence.lower().find(tail.lower())
            match    = pattern.search(sentence)

            if head_pos == -1 or tail_pos == -1 or match is None:
                continue

            pat_pos  = match.start()
            in_order = head_pos < pat_pos < tail_pos
            conf     = 0.85 if in_order else 0.65

            if conf > best_conf:
                best_conf   = conf
                best_triple = Triple(
                    head=head,
                    relation=rel_type,
                    tail=tail,
                    confidence=conf,
                    source_text=sentence,
                )

        return best_triple

    # ------------------------------------------------------------------
    # Step 4 — Entity disambiguation
    # ------------------------------------------------------------------

    def disambiguate(self) -> "KnowledgeGraphBuilder":
        """
        Resolve raw entity names to canonical names using EntityDisambiguator.
        Updates triples to use canonical names.

        Returns self for method chaining.
        """
        try:
            if self.disambiguator is None:
                self.disambiguator = EntityDisambiguator()

            self.canonical_entities = self.disambiguator.disambiguate(
                self.raw_entities
            )

            updated_triples = []
            for t in self.triples:
                canon_head = self.disambiguator.resolve(t.head)
                canon_tail = self.disambiguator.resolve(t.tail)
                if canon_head and canon_tail:
                    updated_triples.append(Triple(
                        head=canon_head,
                        relation=t.relation,
                        tail=canon_tail,
                        confidence=t.confidence,
                        source_text=t.source_text,
                    ))

            self.triples = updated_triples
            print(
                f"[Builder] After disambiguation: "
                f"{len(self.canonical_entities)} canonical entities, "
                f"{len(self.triples)} triples"
            )
            return self
        except Exception as exc:
            print(
                "[Builder] Disambiguation model unavailable; using local "
                f"string-only fallback. Reason: {exc}"
            )
            return self._fallback_disambiguation()

    def _fallback_disambiguation(self) -> "KnowledgeGraphBuilder":
        """
        Fallback disambiguation when sentence-transformer model loading fails.
        Merges only exact-normalized aliases and keeps relation triples intact.
        """
        canonical_by_key: dict[tuple[str, str], Entity] = {}
        plain_key_to_canonical: dict[str, str] = {}

        for ent in self.raw_entities:
            plain_key = self._entity_key(ent.name)
            typed_key = (ent.entity_type.value, plain_key)

            if typed_key not in canonical_by_key:
                canonical_by_key[typed_key] = Entity(
                    name=ent.name,
                    entity_type=ent.entity_type,
                    source_text=ent.source_text,
                    aliases=list(dict.fromkeys(ent.aliases)),
                    metadata=dict(ent.metadata),
                )
            else:
                canon = canonical_by_key[typed_key]
                if ent.name != canon.name and ent.name not in canon.aliases:
                    canon.aliases.append(ent.name)
                for alias in ent.aliases:
                    if alias not in canon.aliases and alias != canon.name:
                        canon.aliases.append(alias)

            plain_key_to_canonical.setdefault(
                plain_key,
                canonical_by_key[typed_key].name,
            )

        self.canonical_entities = list(canonical_by_key.values())

        self.triples = [
            Triple(
                head=plain_key_to_canonical.get(
                    self._entity_key(t.head), t.head
                ),
                relation=t.relation,
                tail=plain_key_to_canonical.get(
                    self._entity_key(t.tail), t.tail
                ),
                confidence=t.confidence,
                source_text=t.source_text,
            )
            for t in self.triples
        ]

        print(
            f"[Builder] Fallback disambiguation: "
            f"{len(self.canonical_entities)} canonical entities, "
            f"{len(self.triples)} triples"
        )
        return self

    @staticmethod
    def _entity_key(name: str) -> str:
        name = name.lower()
        name = re.sub(r"[^\w\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    # ------------------------------------------------------------------
    # Step 5 — Graph construction
    # ------------------------------------------------------------------

    def build_graph(self) -> nx.DiGraph:
        """
        Construct a directed NetworkX graph from canonical entities and triples.

        Node attributes: entity_type, aliases, metadata
        Edge attributes: relation, weight (default 1.0), confidence, source_text

        For any node that appears in a triple but was not in the canonical
        entity list, the type is inferred from the whitelist first, then
        defaults to Supplier.

        Returns
        -------
        nx.DiGraph
            The fully constructed supply chain knowledge graph.
        """
        G = nx.DiGraph()

        entity_map = {e.name: e for e in self.canonical_entities}
        for name, ent in entity_map.items():
            G.add_node(name, **{
                NODE_ATTR_TYPE:     ent.entity_type.value,
                NODE_ATTR_ALIASES:  ent.aliases,
                NODE_ATTR_METADATA: ent.metadata,
            })

        for triple in self.triples:
            for node_name in (triple.head, triple.tail):
                if node_name not in G:
                    fallback_type = CANONICAL_ENTITY_TYPES.get(
                        node_name, EntityType.SUPPLIER
                    )
                    G.add_node(node_name, **{
                        NODE_ATTR_TYPE:     fallback_type.value,
                        NODE_ATTR_ALIASES:  [],
                        NODE_ATTR_METADATA: {},
                    })

            G.add_edge(
                triple.head,
                triple.tail,
                **{
                    EDGE_ATTR_RELATION: triple.relation.value,
                    EDGE_ATTR_WEIGHT:   1.0,
                    EDGE_ATTR_CONF:     triple.confidence,
                    EDGE_ATTR_SOURCE:   triple.source_text or "",
                }
            )

        self.graph = G
        print(
            f"[Builder] Graph built — "
            f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
        )
        return G

    # ------------------------------------------------------------------
    # Step 6 — Edge weight assignment
    # ------------------------------------------------------------------

    def assign_edge_weights(self) -> "KnowledgeGraphBuilder":
        """
        Assign criticality-aware weights to all graph edges.

        Logic:
        - ALTERNATIVE_TO edges → weight 0.0 (risk mitigators, not carriers)
        - SUPPLIES / DEPENDS_ON edges → weight = 1 / (1 + n_alternatives)
          A node with no alternatives gets weight ~1.0 (critical single source).
          A node with 3 alternatives gets weight ~0.25 (redundant supply).
        - All other edges → weight 1.0 (default)

        This makes Module 3's disruption propagation supply-chain-aware:
        shocks travel harder along single-source dependency edges.

        Returns self for method chaining.
        """
        G = self.graph

        for u, v, data in G.edges(data=True):
            relation = data.get(EDGE_ATTR_RELATION, "")

            if relation == RelationType.ALTERNATIVE_TO.value:
                G[u][v][EDGE_ATTR_WEIGHT] = 0.0
                continue

            if relation in {
                RelationType.SUPPLIES.value,
                RelationType.DEPENDS_ON.value,
            }:
                alternatives = [
                    nbr for nbr in G.neighbors(u)
                    if G[u][nbr].get(EDGE_ATTR_RELATION)
                    == RelationType.ALTERNATIVE_TO.value
                ]
                n_alternatives = len(alternatives)
                criticality = 1.0 / (1.0 + n_alternatives)
                G[u][v][EDGE_ATTR_WEIGHT] = round(criticality, 3)

        print("[Builder] Edge weights assigned")
        return self

    # ------------------------------------------------------------------
    # Full pipeline shortcut
    # ------------------------------------------------------------------

    def run_pipeline(self, data: list[dict] | str | Path) -> nx.DiGraph:
        """
        Run all six steps in sequence and return the final graph.

        Parameters
        ----------
        data : list[dict] | str | Path
            Raw documents — same format as accepted by load_documents().

        Returns
        -------
        nx.DiGraph
            Fully built and weighted supply chain knowledge graph.
        """
        return (
            self
            .load_documents(data)
            .extract_entities()
            .extract_triples()
            .disambiguate()
            .build_graph()
            .assign_edge_weights()
            .graph
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def save_graph(self, path: str | Path) -> None:
        """Persist the graph to a .gpickle file for use in later modules."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # NetworkX 3 removed top-level write_gpickle; use stdlib pickle fallback.
        try:
            nx.write_gpickle(self.graph, str(path))  # type: ignore[attr-defined]
        except AttributeError:
            import pickle
            with path.open("wb") as f:
                pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[Builder] Graph saved → {path}")

    def save_triples(self, path: str | Path) -> None:
        """Save extracted triples as JSON for inspection and reuse."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "head":       t.head,
                "relation":   t.relation.value,
                "tail":       t.tail,
                "confidence": t.confidence,
                "source":     t.source_text,
            }
            for t in self.triples
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[Builder] Triples saved → {path}")

    def save_entities(self, path: str | Path) -> None:
        """Save canonical entity registry as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "name":        e.name,
                "entity_type": e.entity_type.value,
                "aliases":     e.aliases,
                "metadata":    e.metadata,
            }
            for e in self.canonical_entities
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[Builder] Entities saved → {path}")

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------

    def graph_stats(self) -> dict:
        """Return a dict of basic graph statistics for quick inspection."""
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
            "nodes":                       G.number_of_nodes(),
            "edges":                       G.number_of_edges(),
            "node_types":                  type_counts,
            "relation_types":              rel_counts,
            "avg_out_degree":              round(
                sum(d for _, d in G.out_degree()) / G.number_of_nodes(), 2
            ),
            "is_weakly_connected":         nx.is_weakly_connected(G),
            "weakly_connected_components": nx.number_weakly_connected_components(G),
        }
