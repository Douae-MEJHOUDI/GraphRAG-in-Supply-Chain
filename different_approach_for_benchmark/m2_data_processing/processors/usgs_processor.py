import csv
import logging
import re
from pathlib import Path

from m2_data_processing.processors.base import BaseProcessor
from m2_data_processing.schema import EntityRecord, RelationshipRecord

logger = logging.getLogger(__name__)

def _clean_country(name: str) -> str:
    name = name.replace("\xa0", " ").strip()
    name = re.sub(r"[\d,]+$", "", name).strip()
    if name.endswith("e") and len(name) > 2 and name[-2] not in "aeiouAEIOU ":
        name = name[:-1]
    return name.strip()


def _find_col(fieldnames: list, candidates: list) -> str | None:
    for c in candidates:
        if c in fieldnames:
            return c
    return None


class USGSProcessor(BaseProcessor):
    source_name = "usgs"

    def process(self, input_dir: Path) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        self.entities = []
        self.relationships = []

        mineral_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if not mineral_dirs:
            logger.warning(f"[USGS] No mineral subdirectories in {input_dir}")
            return [], []

        seen_countries = set()
        seen_minerals  = set()

        for mineral_dir in sorted(mineral_dirs):
            mineral_name = mineral_dir.name.replace("_", " ")
            world_csvs   = list(mineral_dir.glob("*_world.csv"))

            if not world_csvs:
                logger.warning(f"[USGS] No world CSV in {mineral_dir}")
                continue

            csv_path = world_csvs[0]
            logger.info(f"[USGS] Processing {csv_path.name}")

            if mineral_name not in seen_minerals:
                self._add_entity(
                    name=mineral_name,
                    type="mineral",
                    source=str(csv_path),
                )
                seen_minerals.add(mineral_name)

            with open(csv_path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                prod_col     = _find_col(reader.fieldnames or [], ["Prod_t_est_2023", "Prod_kg_est_2023", "Prod_t_2023"])
                reserves_col = _find_col(reader.fieldnames or [], ["Reserves_t", "Reserves_kg", "Cap_kg_2023"])

                for row in reader:
                    country = _clean_country(row.get("Country", ""))
                    prod    = row.get(prod_col, "").strip() if prod_col else ""
                    reserves= row.get(reserves_col, "").strip() if reserves_col else ""

                    skip = {"world total", "other countries", "other", "total",
                            "world total (rounded)", "undistributed"}
                    if not country or country.lower() in skip:
                        continue

                    if not prod or prod in ("W", "—", "-", "NA", ""):
                        prod = None

                    if country not in seen_countries:
                        self._add_entity(
                            name=country,
                            type="country",
                            source=str(csv_path),
                        )
                        seen_countries.add(country)

                    if prod is not None:
                        self._add_rel(
                            source_entity=country,
                            target_entity=mineral_name,
                            type="PRODUCES",
                            evidence=f"{country} produced {prod} metric tons of {mineral_name} in 2023",
                            evidence_source=f"USGS MCS 2024 — {csv_path.name}",
                            properties={
                                "production_mt_2023": prod,
                                "reserves_mt": reserves or None,
                                "mineral": mineral_name,
                            },
                        )

        logger.info(
            f"[USGS] Done: {len(seen_countries)} countries, "
            f"{len(seen_minerals)} minerals, {len(self.relationships)} PRODUCES edges"
        )
        return self.entities, self.relationships
