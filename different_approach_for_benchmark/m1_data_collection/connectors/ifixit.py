import logging
from pathlib import Path

from m1_data_collection.connectors.base import BaseConnector
from m1_data_collection.config import IFIXIT_CATEGORIES, IFIXIT_MAX_GUIDES
from m1_data_collection.utils.storage import save_json

logger = logging.getLogger(__name__)

API_BASE = "https://www.ifixit.com/api/2.0"


class IFixitConnector(BaseConnector):
    source_name = "ifixit"

    def collect(self, categories: list = None, max_guides: int = None) -> list[Path]:
        categories = categories or IFIXIT_CATEGORIES
        max_guides = max_guides or IFIXIT_MAX_GUIDES
        written    = []

        for category in categories:
            logger.info(f"[iFixit] Searching teardowns: {category}")
            try:
                guide_ids = self._search(category, max_guides)
                logger.info(f"[iFixit] Found {len(guide_ids)} teardowns for '{category}'")
                for guide_id in guide_ids:
                    path = self._fetch_guide(guide_id)
                    if path:
                        written.append(path)
            except Exception as e:
                logger.error(f"[iFixit] {category} failed: {e}")

        logger.info(f"[iFixit] {len(written)} guide(s) saved")
        return written

    def _search(self, query: str, limit: int) -> list[int]:
        resp = self.get(
            f"{API_BASE}/search/{query} teardown",
            params={"doctypes": "guide", "limit": limit}
        )
        results = resp.json().get("results", [])

        ids = []
        for r in results:
            if r.get("type") == "teardown" and r.get("guideid"):
                ids.append(r["guideid"])
        return ids

    def _fetch_guide(self, guide_id: int) -> Path | None:
        out_file = self.output_dir / f"guide_{guide_id}.json"
        if out_file.exists():
            logger.debug(f"[iFixit] Guide {guide_id} already cached")
            return out_file

        try:
            resp = self.get(f"{API_BASE}/guides/{guide_id}")
            data = resp.json()

            if "guideid" not in data:
                return None

            slim = {
                "guideid":  data.get("guideid"),
                "title":    data.get("title"),
                "type":     data.get("type"),
                "category": data.get("category"),
                "subject":  data.get("subject"),
                "url":      data.get("url"),
                "tools":    data.get("tools", []),
                "parts":    data.get("parts", []),
            }
            logger.info(f"[iFixit] Saved: {slim['title']} ({len(slim['parts'])} parts)")
            return save_json(slim, self.output_dir, f"guide_{guide_id}.json")

        except Exception as e:
            logger.error(f"[iFixit] Guide {guide_id} failed: {e}")
            return None
