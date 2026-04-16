import json
import logging
import time
from pathlib import Path

import requests
import spacy

from m2_data_processing.processors.base import BaseProcessor
from m2_data_processing.schema import EntityRecord, RelationshipRecord

logger = logging.getLogger(__name__)

API_BASE      = "https://www.ifixit.com/api/2.0"
REQUEST_DELAY = 1.5

_NOT_COMPANIES = {
    "haptic touch", "3d touch", "android", "ios", "ic identification",
    "face id", "touch id", "true tone", "liquid retina", "super retina",
    "usb-c", "lightning", "wi-fi", "bluetooth", "nfc", "lte", "5g",
    "display", "battery", "camera", "sensor", "board", "module", "lcd",
    "nano-sim", "sim card", "iphones", "iphone", "macbook", "macbooks",
    "pentalobe", "phillips", "torx", "screwdriver",
    "ifixit", "wikipedia", "youtube", "amazon",
}


def _valid_org(name: str) -> bool:
    if len(name) < 3:
        return False
    if name[0].isdigit():
        return False
    if name.lower() in _NOT_COMPANIES:
        return False
    alpha_ratio = sum(c.isalpha() for c in name) / len(name)
    if alpha_ratio < 0.5:
        return False
    return True

COMPONENT_KEYWORDS = [
    "chip", "processor", "soc", "nand", "dram", "flash",
    "manufactured by", "made by", "produced by", "fabricated by",
    "apple a", "snapdragon", "exynos", "mediatek",
    "tsmc", "samsung foundry", "intel", "qualcomm",
    "micron", "sk hynix", "kioxia", "western digital",
    "broadcom", "texas instruments", "murata", "alps",
]


class IFixitProcessor(BaseProcessor):
    source_name = "ifixit"

    def __init__(self):
        super().__init__()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SupplyChainResearch researcher@university.edu"})
        logger.info("[iFixit] Loading spaCy model ...")
        self.nlp = spacy.load("en_core_web_sm")

    def process(self, input_dir: Path) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        self.entities = []
        self.relationships = []

        guide_files = sorted(input_dir.glob("guide_*.json"))
        if not guide_files:
            logger.warning(f"[iFixit] No guide JSON files in {input_dir}")
            return [], []

        seen_products = set()
        seen_orgs     = set()

        for guide_file in guide_files:
            with open(guide_file, encoding="utf-8", errors="replace") as f:
                guide = json.load(f)

            guide_id = guide.get("guideid")
            title    = guide.get("title", "").strip()
            category = guide.get("category", "").strip()
            subject  = guide.get("subject", "").strip()

            product_name = subject or category or title
            if not product_name:
                continue

            logger.info(f"[iFixit] Fetching steps for: {title} (id={guide_id})")

            steps_text = self._fetch_steps_text(guide_id)
            if not steps_text:
                logger.warning(f"[iFixit] No step text retrieved for guide {guide_id}")
                continue

            if product_name not in seen_products:
                self._add_entity(
                    name=product_name,
                    type="product",
                    source=str(guide_file),
                    properties={"guide_id": guide_id, "title": title},
                )
                seen_products.add(product_name)

            orgs_found = self._extract_orgs(steps_text, str(guide_file))

            for org_name in orgs_found:
                if org_name not in seen_orgs:
                    self._add_entity(
                        name=org_name,
                        type="company",
                        source=str(guide_file),
                    )
                    seen_orgs.add(org_name)

                self._add_rel(
                    source_entity=product_name,
                    target_entity=org_name,
                    type="DEPENDS_ON",
                    evidence=f"Component manufacturer '{org_name}' mentioned in {title} teardown",
                    evidence_source=f"iFixit guide {guide_id}",
                    confidence=0.6,
                    properties={"guide_id": guide_id, "product": product_name},
                )

        logger.info(
            f"[iFixit] Done: {len(seen_products)} products, "
            f"{len(seen_orgs)} component orgs, {len(self.relationships)} DEPENDS_ON edges"
        )
        return self.entities, self.relationships

    def _fetch_steps_text(self, guide_id: int) -> str:
        time.sleep(REQUEST_DELAY)
        try:
            resp = self.session.get(f"{API_BASE}/guides/{guide_id}", timeout=30)
            resp.raise_for_status()
            data  = resp.json()
            steps = data.get("steps", [])

            parts = []
            for step in steps:
                if step.get("title"):
                    parts.append(step["title"])
                for line in step.get("lines", []):
                    if line.get("text_raw"):
                        parts.append(line["text_raw"])

            return " ".join(parts)
        except Exception as e:
            logger.error(f"[iFixit] Failed to fetch steps for guide {guide_id}: {e}")
            return ""

    def _extract_orgs(self, text: str, source: str) -> list[str]:
        found = set()

        doc = self.nlp(text[:self.nlp.max_length])
        for ent in doc.ents:
            if ent.label_ == "ORG" and _valid_org(ent.text.strip()):
                found.add(ent.text.strip())

        text_lower = text.lower()
        known_manufacturers = [
            "TSMC", "Samsung", "Qualcomm", "Broadcom", "Micron",
            "SK Hynix", "Kioxia", "Texas Instruments", "Murata",
            "Bosch", "STMicroelectronics", "NXP", "Infineon",
        ]
        for mfr in known_manufacturers:
            if mfr.lower() in text_lower:
                found.add(mfr)

        return list(found)
