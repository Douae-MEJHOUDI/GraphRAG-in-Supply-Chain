import logging
import re
from pathlib import Path

import spacy

from m2_data_processing.processors.base import BaseProcessor
from m2_data_processing.schema import EntityRecord, RelationshipRecord

logger = logging.getLogger(__name__)

SUPPLIER_KEYWORDS = [
    "sole source", "sole supplier", "single source",
    "contract manufacturer", "contract manufacturing",
    "outsource", "outsourced",
    "supplier", "suppliers", "supply",
    "manufacturer", "manufacturers", "manufactures",
    "fabricat", "foundry", "assembly",
    "depend on", "relies on", "rely on", "dependent on",
    "procure", "source from", "purchased from",
]

KEYWORD_TO_REL = {
    "contract manufacturer": "MANUFACTURES_FOR",
    "contract manufacturing": "MANUFACTURES_FOR",
    "foundry":               "MANUFACTURES_FOR",
    "fabricat":              "MANUFACTURES_FOR",
    "assembly":              "MANUFACTURES_FOR",
    "sole source":           "SUPPLIES",
    "sole supplier":         "SUPPLIES",
    "single source":         "SUPPLIES",
    "supplier":              "SUPPLIES",
    "supply":                "SUPPLIES",
    "outsource":             "SUPPLIES",
    "procure":               "SUPPLIES",
    "source from":           "SUPPLIES",
    "purchased from":        "SUPPLIES",
}

CONTEXT_WINDOW = 500

TICKER_TO_NAME = {
    "AAPL": "Apple Inc.",
    "INTC": "Intel Corporation",
    "NVDA": "NVIDIA Corporation",
    "QCOM": "Qualcomm",
    "AMD":  "Advanced Micro Devices",
    "TSM":  "Taiwan Semiconductor Manufacturing",
}


class SECProcessor(BaseProcessor):
    source_name = "sec"

    def __init__(self):
        super().__init__()
        logger.info("[SEC] Loading spaCy model en_core_web_sm ...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.max_length = 2_000_000

    def process(self, input_dir: Path) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        self.entities = []
        self.relationships = []

        ticker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        if not ticker_dirs:
            logger.warning(f"[SEC] No ticker subdirectories in {input_dir}")
            return [], []

        for ticker_dir in sorted(ticker_dirs):
            ticker = ticker_dir.name
            htm_files = list(ticker_dir.glob("*.htm"))
            if not htm_files:
                logger.warning(f"[SEC] No .htm file for {ticker}")
                continue

            filing = htm_files[0]
            logger.info(f"[SEC] Processing {filing.name}")
            self._process_filing(ticker, filing)

        logger.info(
            f"[SEC] Done: {len(self.entities)} entities, "
            f"{len(self.relationships)} relationships"
        )
        return self.entities, self.relationships

    def _process_filing(self, ticker: str, path: Path):
        company_name = TICKER_TO_NAME.get(ticker, ticker)

        self._add_entity(
            name=company_name,
            type="company",
            source=str(path),
            aliases=[ticker],
            properties={"ticker": ticker, "role": "filer"},
        )

        with open(path, encoding="utf-8", errors="replace") as f:
            html = f.read()
        text = _strip_html(html)

        seen_orgs = set()
        seen_rels = set()

        for keyword in SUPPLIER_KEYWORDS:
            for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                start = max(0, match.start() - CONTEXT_WINDOW // 2)
                end   = min(len(text), match.end() + CONTEXT_WINDOW // 2)
                snippet = text[start:end]

                doc = self.nlp(snippet[:self.nlp.max_length])

                rel_type = _keyword_to_rel(keyword)

                for ent in doc.ents:
                    if ent.label_ not in ("ORG", "GPE", "LOC"):
                        continue

                    ent_name = ent.text.strip()

                    if not _valid_entity(ent_name, company_name):
                        continue

                    if ent_name not in seen_orgs:
                        etype = "company" if ent.label_ == "ORG" else "country"
                        self._add_entity(
                            name=ent_name,
                            type=etype,
                            source=str(path),
                        )
                        seen_orgs.add(ent_name)

                    rel_key = (company_name, ent_name, rel_type)
                    if ent.label_ == "ORG" and rel_key not in seen_rels:
                        self._add_rel(
                            source_entity=ent_name,
                            target_entity=company_name,
                            type=rel_type,
                            evidence=snippet.strip()[:300],
                            evidence_source=f"{ticker} {path.stem}",
                            confidence=0.7,
                            properties={"keyword_matched": keyword, "ticker": ticker},
                        )
                        seen_rels.add(rel_key)

                    if ent.label_ in ("GPE", "LOC"):
                        for other in doc.ents:
                            if other.label_ == "ORG" and other.text.strip() != ent_name:
                                loc_key = (other.text.strip(), ent_name, "LOCATED_IN")
                                if loc_key not in seen_rels:
                                    self._add_rel(
                                        source_entity=other.text.strip(),
                                        target_entity=ent_name,
                                        type="LOCATED_IN",
                                        evidence=snippet.strip()[:300],
                                        evidence_source=f"{ticker} {path.stem}",
                                        confidence=0.5,
                                        properties={"ticker": ticker},
                                    )
                                    seen_rels.add(loc_key)

        logger.info(
            f"[SEC] {ticker}: {len(seen_orgs)} org/geo entities, "
            f"{len(seen_rels)} relationships extracted"
        )


_SKIP_WORDS = {
    "company", "companies", "government", "authority", "authorities",
    "exchange", "act", "agreement", "amendment", "section", "assembly",
    "the company", "our company", "such company", "board of directors",
    "board of", "committee", "commission", "department", "division",
    "subsidiary", "affiliates", "partners", "associates",
}

_LEGAL_FRAGMENTS = re.compile(
    r"(amended|restated|limited liability|corporation agreement|"
    r"table of contents|exhibit|form \d|item \d|schedule)",
    re.IGNORECASE,
)


def _valid_entity(name: str, filer_name: str) -> bool:
    if len(name) < 4 or len(name) > 80:
        return False
    if name[0].isdigit():
        return False
    if "/" in name or name.count(" ") > 7:
        return False
    if name.lower() in _SKIP_WORDS:
        return False
    if _LEGAL_FRAGMENTS.search(name):
        return False
    if filer_name.lower() in name.lower() or name.lower() in filer_name.lower():
        return False
    alpha_ratio = sum(c.isalpha() for c in name) / len(name)
    if alpha_ratio < 0.55:
        return False
    words = name.split()
    if len(words) != len(set(words)):
        return False
    return True


def _strip_html(html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#8217;", "'").replace("&#8220;", '"').replace("&#8221;", '"')
    text = re.sub(r"&#\d+;", " ", text)
    text = re.sub(r"&\w+;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _keyword_to_rel(keyword: str) -> str:
    kw_lower = keyword.lower()
    for k, rel in KEYWORD_TO_REL.items():
        if k in kw_lower:
            return rel
    return "SUPPLIES"
