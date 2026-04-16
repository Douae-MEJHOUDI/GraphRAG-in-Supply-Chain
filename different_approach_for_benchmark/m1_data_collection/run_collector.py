import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("m1_collection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ALL_SOURCES = ["sec", "usgs", "gdelt", "ifixit"]


def run_sec(args) -> list[Path]:
    from m1_data_collection.connectors.sec_edgar import SECEdgarConnector
    return SECEdgarConnector().collect(filing_count=args.sec_filings)


def run_usgs(args) -> list[Path]:
    from m1_data_collection.connectors.usgs_minerals import USGSMineralsConnector
    return USGSMineralsConnector().collect()


def run_gdelt(args) -> list[Path]:
    from m1_data_collection.connectors.gdelt import GDELTConnector
    return GDELTConnector().collect(lookback_days=args.gdelt_days)


def run_ifixit(args) -> list[Path]:
    from m1_data_collection.connectors.ifixit import IFixitConnector
    return IFixitConnector().collect()


RUNNERS = {
    "sec":    run_sec,
    "usgs":   run_usgs,
    "gdelt":  run_gdelt,
    "ifixit": run_ifixit,
}


def main():
    parser = argparse.ArgumentParser(description="M1: Data Collection Pipeline")
    parser.add_argument(
        "--source", nargs="+", choices=ALL_SOURCES + ["all"],
        default=["all"], help="Which connector(s) to run"
    )
    parser.add_argument(
        "--sec-filings", type=int, default=1,
        help="Number of 10-K filings to fetch per company (default: 1)"
    )
    parser.add_argument(
        "--gdelt-days", type=int, default=7,
        help="Number of days to look back for GDELT events (default: 7)"
    )
    args = parser.parse_args()

    sources = ALL_SOURCES if "all" in args.source else args.source

    results = {}
    total_files = 0

    for source in sources:
        logger.info(f"=== Running connector: {source} ===")
        start = time.time()
        try:
            files = RUNNERS[source](args)
            results[source] = {"status": "ok", "files": len(files)}
            total_files += len(files)
            logger.info(f"=== {source}: {len(files)} file(s) in {time.time()-start:.1f}s ===")
        except Exception as e:
            logger.error(f"=== {source}: FAILED — {e} ===")
            results[source] = {"status": "failed", "error": str(e)}

    print("\n--- Collection Summary ---")
    for source, result in results.items():
        status = result["status"]
        if status == "ok":
            print(f"  {source:8s}: OK  ({result['files']} file(s))")
        else:
            print(f"  {source:8s}: FAILED — {result['error']}")
    print(f"\n  Total files saved: {total_files}")
    print(f"  Output directory:  data/raw/")


if __name__ == "__main__":
    main()
