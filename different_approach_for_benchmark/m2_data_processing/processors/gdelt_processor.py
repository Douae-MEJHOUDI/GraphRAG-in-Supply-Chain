import csv
import logging
from pathlib import Path

from m2_data_processing.processors.base import BaseProcessor
from m2_data_processing.schema import EntityRecord, RelationshipRecord

logger = logging.getLogger(__name__)

SUPPLY_CHAIN_COUNTRIES = {
    "TW", "CH", "KS", "JA", "MY", "TH", "VM", "SN",
    "IN", "ID", "RP", "GM", "NL", "US",
}

DISRUPTION_ROOT_CODES = {"13", "14", "17", "18", "19", "20"}

GOLDSTEIN_THRESHOLD = -3.0

MIN_ARTICLES = 3

FIPS_TO_NAME = {
    "TW": "Taiwan",       "CH": "China",        "KS": "South Korea",
    "JA": "Japan",        "MY": "Malaysia",     "TH": "Thailand",
    "VM": "Vietnam",      "SN": "Singapore",    "IN": "India",
    "ID": "Indonesia",    "RP": "Philippines",  "GM": "Germany",
    "NL": "Netherlands",  "US": "United States",
}

CAMEO_ROOT_LABELS = {
    "13": "Threaten", "14": "Protest/Strike", "17": "Sanction/Embargo",
    "18": "Assault", "19": "Armed Conflict", "20": "Mass Violence",
}


class GDELTProcessor(BaseProcessor):
    source_name = "gdelt"

    def process(self, input_dir: Path) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        self.entities = []
        self.relationships = []

        csv_files = sorted(input_dir.glob("events_*.csv"))
        if not csv_files:
            logger.warning(f"[GDELT] No event CSVs in {input_dir}")
            return [], []

        total_rows = 0
        kept_rows  = 0
        seen_locs  = set()

        for csv_path in csv_files:
            logger.info(f"[GDELT] Processing {csv_path.name}")

            with open(csv_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1

                    country_code = row.get("ActionGeo_CountryCode", "").strip()
                    if country_code not in SUPPLY_CHAIN_COUNTRIES:
                        continue

                    event_root = row.get("EventRootCode", "").strip()
                    if event_root not in DISRUPTION_ROOT_CODES:
                        continue

                    try:
                        goldstein = float(row.get("GoldsteinScale", "0") or "0")
                        articles  = int(row.get("NumArticles", "0") or "0")
                    except ValueError:
                        continue

                    if goldstein >= GOLDSTEIN_THRESHOLD:
                        continue
                    if articles < MIN_ARTICLES:
                        continue

                    kept_rows += 1

                    location     = row.get("ActionGeo_FullName", FIPS_TO_NAME.get(country_code, country_code)).strip()
                    country_name = FIPS_TO_NAME.get(country_code, country_code)
                    date         = row.get("Day", "").strip()
                    event_code   = row.get("EventCode", "").strip()
                    url          = row.get("SOURCEURL", "").strip()
                    tone         = row.get("AvgTone", "0")
                    event_label  = CAMEO_ROOT_LABELS.get(event_root, event_root)

                    if country_name not in seen_locs:
                        self._add_entity(
                            name=country_name,
                            type="country",
                            source=str(csv_path),
                            properties={"country_code": country_code},
                        )
                        seen_locs.add(country_name)

                    event_name = f"{event_label} in {country_name} ({date})"
                    self._add_entity(
                        name=event_name,
                        type="risk_event",
                        source=str(csv_path),
                        properties={
                            "date": date,
                            "event_code": event_code,
                            "event_root": event_root,
                            "event_label": event_label,
                            "goldstein": goldstein,
                            "num_articles": articles,
                            "tone": tone,
                            "country": country_name,
                            "location": location,
                            "source_url": url,
                        },
                    )

                    self._add_rel(
                        source_entity=event_name,
                        target_entity=country_name,
                        type="AFFECTS",
                        evidence=f"{event_label} event (code {event_code}) in {location} on {date}. "
                                 f"Goldstein={goldstein}, articles={articles}. Source: {url}",
                        evidence_source=f"GDELT {csv_path.name}",
                        confidence=min(0.5 + articles * 0.02, 0.9),
                        properties={
                            "event_code": event_code,
                            "event_label": event_label,
                            "goldstein": goldstein,
                            "date": date,
                        },
                    )

        logger.info(
            f"[GDELT] {total_rows} rows scanned → {kept_rows} kept "
            f"({100*kept_rows/max(total_rows,1):.1f}%) — "
            f"supply chain disruptions only"
        )
        logger.info(f"[GDELT] {len(self.entities)} entities, {len(self.relationships)} AFFECTS edges")
        return self.entities, self.relationships
