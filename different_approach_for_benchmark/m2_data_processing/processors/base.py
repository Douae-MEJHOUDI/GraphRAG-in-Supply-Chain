from abc import ABC, abstractmethod
from pathlib import Path

from m2_data_processing.schema import EntityRecord, RelationshipRecord


class BaseProcessor(ABC):

    source_name: str = ""

    def __init__(self):
        self.entities: list[EntityRecord] = []
        self.relationships: list[RelationshipRecord] = []

    @abstractmethod
    def process(self, input_dir: Path) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        pass

    def _add_entity(self, **kwargs) -> EntityRecord:
        e = EntityRecord(**kwargs)
        self.entities.append(e)
        return e

    def _add_rel(self, **kwargs) -> RelationshipRecord:
        r = RelationshipRecord(**kwargs)
        self.relationships.append(r)
        return r
