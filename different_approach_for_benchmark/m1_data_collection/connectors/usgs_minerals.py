import logging
from pathlib import Path

from m1_data_collection.connectors.base import BaseConnector
from m1_data_collection.config import USGS_MINERALS
from m1_data_collection.utils.storage import save_bytes, save_json

logger = logging.getLogger(__name__)

SCIENCEBASE_SEARCH = "https://www.sciencebase.gov/catalog/items"
SCIENCEBASE_ITEM   = "https://www.sciencebase.gov/catalog/item/{item_id}"


class USGSMineralsConnector(BaseConnector):
    source_name = "usgs"

    def collect(self, minerals: list = None) -> list[Path]:
        minerals = minerals or USGS_MINERALS
        written  = []

        for mineral in minerals:
            logger.info(f"[USGS] Fetching: {mineral}")
            try:
                paths = self._fetch_mineral(mineral)
                written.extend(paths)
            except Exception as e:
                logger.error(f"[USGS] {mineral}: {e}")

        logger.info(f"[USGS] {len(written)} file(s) saved")
        return written

    def _fetch_mineral(self, mineral: str) -> list[Path]:
        query = f"mineral commodity summaries 2024 {mineral}"
        resp  = self.get(
            SCIENCEBASE_SEARCH,
            params={"q": query, "format": "json", "max": 5}
        )
        results = resp.json().get("items", [])

        if not results:
            logger.warning(f"[USGS] No ScienceBase entry for '{mineral}'")
            return []

        item_id = results[0]["id"]
        title   = results[0].get("title", "")
        logger.info(f"[USGS] Found: {title} (id={item_id})")

        item_resp = self.get(SCIENCEBASE_ITEM.format(item_id=item_id), params={"format": "json"})
        item      = item_resp.json()

        mineral_slug = mineral.lower().replace(" ", "_")
        written = []

        save_json(
            {"mineral": mineral, "title": title, "item_id": item_id},
            self.output_dir / mineral_slug, "meta.json"
        )

        for f in item.get("files", []):
            if f.get("contentType") != "text/csv":
                continue
            name        = f.get("name", "file.csv")
            download_url = f.get("downloadUri") or f.get("url")
            if not download_url:
                continue

            logger.info(f"[USGS] Downloading {name}")
            try:
                data_resp = self.session.get(download_url, timeout=30)
                data_resp.raise_for_status()
                path = save_bytes(data_resp.content, self.output_dir / mineral_slug, name)
                written.append(path)
            except Exception as e:
                logger.warning(f"[USGS] Failed to download {name}: {e}")

        if not written:
            logger.warning(f"[USGS] No CSV files found for '{mineral}' (item may only have PDFs)")

        return written
