"""
schema.py
---------
Supply chain ontology definition.

Defines the fixed vocabulary of node types, edge types, and the dataclasses
used throughout the pipeline to represent entities and relationships.
Everything in the project that creates or reads graph data imports from here,
so the ontology stays in one authoritative place.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Ontology — Entity Types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """
    Node types in the supply chain knowledge graph.
    Each maps to a real-world SC concept and drives NER label assignment.
    """
    SUPPLIER       = "Supplier"        # raw material or component vendor
    MANUFACTURER   = "Manufacturer"    # assembles finished goods
    PART           = "Part"            # component or raw material
    PORT           = "Port"            # logistics hub (sea, air, rail)
    REGION         = "Region"          # geographic area (country, province)
    LOGISTICS_ROUTE = "LogisticsRoute" # shipping lane or transport link
    DISRUPTION     = "DisruptionEvent" # an event that causes supply shock
    CUSTOMER       = "Customer"        # end buyer or downstream entity


# ---------------------------------------------------------------------------
# Ontology — Relation Types
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    """
    Directed edge types in the supply chain knowledge graph.
    Direction always follows the physical or logical flow:
      (source node) --[relation]--> (target node)
    """
    SUPPLIES         = "supplies"          # Supplier → Part / Manufacturer
    DEPENDS_ON       = "depends_on"        # Manufacturer → Part / Supplier
    SHIPS_THROUGH    = "ships_through"     # Supplier/Manufacturer → Port
    LOCATED_IN       = "located_in"        # any entity → Region
    AFFECTED_BY      = "affected_by"       # any entity → DisruptionEvent
    ALTERNATIVE_TO   = "alternative_to"    # Supplier/Part → Supplier/Part
    SELLS_TO         = "sells_to"          # Manufacturer → Customer
    CONNECTS         = "connects"          # Port/Route → Port/Region


# ---------------------------------------------------------------------------
# Dataclasses — used throughout the pipeline for type safety
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    A single supply chain entity before it becomes a graph node.

    Attributes
    ----------
    name        : canonical name after disambiguation
    entity_type : one of EntityType enum values
    aliases     : alternative surface forms seen in raw text
    metadata    : free-form dict for extra attributes (country, capacity, etc.)
    source_text : the sentence from which this entity was extracted
    """
    name:        str
    entity_type: EntityType
    aliases:     list[str]            = field(default_factory=list)
    metadata:    dict                 = field(default_factory=dict)
    source_text: Optional[str]        = None

    def __post_init__(self):
        # Normalise name to title-case for consistency
        self.name = self.name.strip()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Entity) and self.name == other.name


@dataclass
class Triple:
    """
    A single (head, relation, tail) fact extracted from text.

    Attributes
    ----------
    head         : canonical name of the source entity
    relation     : one of RelationType enum values
    tail         : canonical name of the target entity
    confidence   : extraction confidence score (0.0 – 1.0)
    source_text  : original sentence this was extracted from
    """
    head:        str
    relation:    RelationType
    tail:        str
    confidence:  float         = 1.0
    source_text: Optional[str] = None

    def __repr__(self):
        return f"Triple({self.head!r} --[{self.relation.value}]--> {self.tail!r})"


# ---------------------------------------------------------------------------
# Constants — used for graph node/edge attribute keys
# ---------------------------------------------------------------------------

NODE_ATTR_TYPE      = "entity_type"   # EntityType value stored on each node
NODE_ATTR_ALIASES   = "aliases"       # list[str] of known alternative names
NODE_ATTR_METADATA  = "metadata"      # dict of extra attributes
NODE_ATTR_EMBEDDING = "embedding"     # np.ndarray, populated in Module 1 Step 6

EDGE_ATTR_RELATION  = "relation"      # RelationType value stored on each edge
EDGE_ATTR_WEIGHT    = "weight"        # float, used in disruption propagation
EDGE_ATTR_CONF      = "confidence"    # extraction confidence from NLP step
EDGE_ATTR_SOURCE    = "source_text"   # provenance sentence


# ---------------------------------------------------------------------------
# Helper — human-readable lists for prompts and logging
# ---------------------------------------------------------------------------

ENTITY_TYPES  = [e.value for e in EntityType]
RELATION_TYPES = [r.value for r in RelationType]
