import logging
import time
from pathlib import Path

from m1_data_collection.connectors.base import BaseConnector
from m1_data_collection.config import SEC_TARGET_COMPANIES
from m1_data_collection.utils.storage import save_text, save_json

logger = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL    = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}"


class SECEdgarConnector(BaseConnector):
    source_name = "sec"

    def collect(self, companies: dict = None, filing_count: int = 1) -> list[Path]:
        companies = companies or SEC_TARGET_COMPANIES
        written   = []

        for ticker, (cik, form_type) in companies.items():
            logger.info(f"[SEC] {ticker} (CIK {cik}, form {form_type})")
            try:
                written.extend(self._fetch_company(ticker, cik.lstrip("0"), form_type, filing_count))
            except Exception as e:
                logger.error(f"[SEC] {ticker} failed: {e}")

        return written

    def _fetch_company(self, ticker: str, cik: str, form_type: str, filing_count: int) -> list[Path]:
        url = SUBMISSIONS_URL.format(cik=cik.zfill(10))
        submissions = self.get(url).json()
        save_json(submissions, self.output_dir / ticker, "submissions.json")

        recent     = submissions.get("filings", {}).get("recent", {})
        forms      = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        dates      = recent.get("filingDate", [])
        docs       = recent.get("primaryDocument", [])

        written = []
        fetched = 0

        for i, form in enumerate(forms):
            if form != form_type:
                continue
            if fetched >= filing_count:
                break

            acc_no_dashes = accessions[i].replace("-", "")
            date = dates[i]
            doc  = docs[i]

            url = ARCHIVES_URL.format(cik=cik, accession=acc_no_dashes, doc=doc)
            logger.info(f"[SEC] Downloading {ticker} {form_type} {date}")

            time.sleep(0.15)
            try:
                resp = self.session.get(url, timeout=120)
                resp.raise_for_status()

                ext   = "htm" if doc.endswith(".htm") else doc.rsplit(".", 1)[-1]
                fname = f"{form_type.replace('/','_')}_{ticker}_{date}.{ext}"
                path  = save_text(resp.text, self.output_dir / ticker, fname)

                save_json(
                    {"ticker": ticker, "cik": cik, "form": form_type, "date": date,
                     "accession": accessions[i], "url": url, "local_file": str(path)},
                    self.output_dir / ticker, f"meta_{date}.json"
                )
                written.append(path)
                fetched += 1

            except Exception as e:
                logger.warning(f"[SEC] Download failed {ticker} {date}: {e}")
                save_json(
                    {"ticker": ticker, "date": date, "accession": accessions[i],
                     "url": url, "error": str(e)},
                    self.output_dir / ticker, f"failed_{date}.json"
                )

        logger.info(f"[SEC] {ticker}: {fetched}/{filing_count} filing(s) saved")
        return written
