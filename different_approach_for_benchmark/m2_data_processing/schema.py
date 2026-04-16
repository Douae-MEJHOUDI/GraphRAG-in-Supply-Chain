from dataclasses import dataclass, field, asdict
from typing import Optional
import uuid


@dataclass
class EntityRecord:
    name: str
    type: str
    source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    aliases: list[str] = field(default_factory=list)
    properties: dict   = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RelationshipRecord:
    source_entity: str
    target_entity: str
    type: str
    evidence: str
    evidence_source: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    confidence: float = 1.0
    properties: dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


ENTITY_TYPES = {"company", "country", "mineral", "product", "risk_event"}

RELATIONSHIP_TYPES = {
    "SUPPLIES",
    "MANUFACTURES_FOR",
    "LOCATED_IN",
    "PRODUCES",
    "DEPENDS_ON",
    "AFFECTS",
    "TRADE_FLOW",
}
