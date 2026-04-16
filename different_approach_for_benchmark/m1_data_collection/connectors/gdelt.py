import csv
import gzip
import io
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

import requests

from m1_data_collection.connectors.base import BaseConnector
from m1_data_collection.config import GDELT_COUNTRIES, GDELT_LOOKBACK_DAYS
from m1_data_collection.utils.storage import save_csv, timestamped_filename

logger = logging.getLogger(__name__)

GDELT_BASE_URL = "http://data.gdeltproject.org/events/"

GDELT_COLUMNS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
    "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
    "Actor1Geo_ADM1Code", "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName", "ActionGeo_CountryCode",
    "ActionGeo_ADM1Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL",
]

COUNTRY_FIELD_INDICES = [7, 17, 37, 42, 47]


class GDELTConnector(BaseConnector):
    source_name = "gdelt"

    def collect(
        self,
        countries: list = None,
        lookback_days: int = None,
    ) -> list[Path]:
        countries   = set(countries or GDELT_COUNTRIES)
        lookback    = lookback_days or GDELT_LOOKBACK_DAYS
        written     = []

        for days_back in range(1, lookback + 1):
            date = datetime.utcnow() - timedelta(days=days_back)
            date_str = date.strftime("%Y%m%d")

            out_file = self.output_dir / f"events_{date_str}.csv"
            if out_file.exists():
                logger.info(f"[GDELT] {date_str} already exists, skipping")
                written.append(out_file)
                continue

            try:
                rows = self._fetch_day(date_str, countries)
                if rows:
                    path = save_csv(rows, self.output_dir, f"events_{date_str}.csv")
                    if path:
                        written.append(path)
                    logger.info(f"[GDELT] {date_str}: {len(rows)} matching events saved")
                else:
                    logger.info(f"[GDELT] {date_str}: no matching events")
            except Exception as e:
                logger.error(f"[GDELT] {date_str} failed: {e}")

        logger.info(f"[GDELT] {len(written)} day file(s) saved")
        return written

    def _fetch_day(self, date_str: str, countries: set) -> list[dict]:
        url = f"{GDELT_BASE_URL}{date_str}.export.CSV.zip"
        logger.info(f"[GDELT] Fetching {url}")

        try:
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"[GDELT] {date_str} not yet available (404)")
                return []
            raise

        raw = io.BytesIO(resp.content)

        matching = []
        try:
            import zipfile
            with zipfile.ZipFile(raw) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    reader = csv.reader(
                        io.TextIOWrapper(f, encoding="utf-8", errors="replace"),
                        delimiter="\t"
                    )
                    for row in reader:
                        if len(row) < len(GDELT_COLUMNS):
                            continue
                        if self._matches_country(row, countries):
                            matching.append(dict(zip(GDELT_COLUMNS, row)))
        except Exception as e:
            logger.error(f"[GDELT] Parse error for {date_str}: {e}")
            return []

        return matching

    def _matches_country(self, row: list, countries: set) -> bool:
        for idx in COUNTRY_FIELD_INDICES:
            if idx < len(row) and row[idx] in countries:
                return True
        return False
