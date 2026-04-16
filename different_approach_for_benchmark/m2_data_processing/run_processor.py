import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("m2_processing.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ALL_SOURCES   = ["sec", "usgs", "gdelt", "ifixit"]


def save_results(source: str, entities, relationships):
    out_dir = PROCESSED_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "entities.json", "w", encoding="utf-8") as f:
        json.dump([e.to_dict() for e in entities], f, indent=2, ensure_ascii=False)

    with open(out_dir / "relationships.json", "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in relationships], f, indent=2, ensure_ascii=False)

    logger.info(f"[{source}] Saved {len(entities)} entities, {len(relationships)} relationships")


def run_source(source: str):
    input_dir = RAW_DIR / source

    if source == "usgs":
        from m2_data_processing.processors.usgs_processor import USGSProcessor
        processor = USGSProcessor()

    elif source == "sec":
        from m2_data_processing.processors.sec_processor import SECProcessor
        processor = SECProcessor()

    elif source == "gdelt":
        from m2_data_processing.processors.gdelt_processor import GDELTProcessor
        processor = GDELTProcessor()

    elif source == "ifixit":
        from m2_data_processing.processors.ifixit_processor import IFixitProcessor
        processor = IFixitProcessor()

    else:
        raise ValueError(f"Unknown source: {source}")

    entities, relationships = processor.process(input_dir)
    save_results(source, entities, relationships)
    return len(entities), len(relationships)


def main():
    parser = argparse.ArgumentParser(description="M2: Data Processing Pipeline")
    parser.add_argument(
        "--source", nargs="+", choices=ALL_SOURCES + ["all"],
        default=["all"]
    )
    args = parser.parse_args()

    sources = ALL_SOURCES if "all" in args.source else args.source

    print("\n--- Processing Summary ---")
    for source in sources:
        logger.info(f"=== Processing: {source} ===")
        start = time.time()
        try:
            n_ent, n_rel = run_source(source)
            elapsed = time.time() - start
            print(f"  {source:8s}: {n_ent:4d} entities  {n_rel:5d} relationships  ({elapsed:.1f}s)")
        except Exception as e:
            logger.error(f"=== {source}: FAILED — {e} ===", exc_info=True)
            print(f"  {source:8s}: FAILED — {e}")

    print(f"\n  Output: data/processed/")


if __name__ == "__main__":
    main()
