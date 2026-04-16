import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("m3_resolution.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

CANONICAL_DIR = Path("data/canonical")


def main():
    from m3_entity_resolution.resolver import EntityResolver

    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

    resolver = EntityResolver()
    entities, relationships = resolver.resolve_all()

    with open(CANONICAL_DIR / "entities.json", "w", encoding="utf-8") as f:
        json.dump(entities, f, indent=2, ensure_ascii=False)

    with open(CANONICAL_DIR / "relationships.json", "w", encoding="utf-8") as f:
        json.dump(relationships, f, indent=2, ensure_ascii=False)

    with open(CANONICAL_DIR / "ambiguous.json", "w", encoding="utf-8") as f:
        json.dump(resolver.ambiguous, f, indent=2, ensure_ascii=False)

    type_counts = Counter(e["type"] for e in entities)
    rel_counts  = Counter(r["type"] for r in relationships)
    stats = {
        "canonical_entities": len(entities),
        "relationships":      len(relationships),
        "entities_by_type":   dict(type_counts),
        "relationships_by_type": dict(rel_counts),
        "ambiguous_flagged":  len(resolver.ambiguous),
    }
    with open(CANONICAL_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n--- M3 Resolution Summary ---")
    print(f"  Canonical entities : {len(entities)}")
    for t, n in type_counts.most_common():
        print(f"    {t:15s}: {n}")
    print(f"  Relationships      : {len(relationships)}")
    for t, n in rel_counts.most_common():
        print(f"    {t:20s}: {n}")
    print(f"  Ambiguous flagged  : {len(resolver.ambiguous)}")
    print(f"  Output             : data/canonical/")


if __name__ == "__main__":
    main()
