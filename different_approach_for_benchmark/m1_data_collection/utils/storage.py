import json
import csv
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


def save_json(data: dict | list, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {path}")
    return path


def save_text(text: str, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info(f"Saved {path}")
    return path


def save_bytes(data: bytes, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "wb") as f:
        f.write(data)
    logger.info(f"Saved {path}")
    return path


def save_csv(rows: list[dict], directory: Path, filename: str) -> Path:
    if not rows:
        logger.warning(f"No rows to save for {filename}")
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} rows to {path}")
    return path


def timestamped_filename(prefix: str, ext: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{ext}"
