import logging
from pathlib import Path
from typing import Any

import kuzu

logger = logging.getLogger(__name__)

GRAPH_DIR = Path("data/graph/supply_chain.db")


class GraphRetriever:

    def __init__(self, graph_dir: Path = GRAPH_DIR):
        self.db   = kuzu.Database(str(graph_dir))
        self.conn = kuzu.Connection(self.db)

    def resolve_name(self, raw: str, label: str) -> str:
        if not raw:
            return raw
        rows = self._q(f"MATCH (n:{label}) WHERE n.name = $name RETURN n.name LIMIT 1", {"name": raw})
        if rows:
            return rows[0][0]
        rows = self._q(
            f"MATCH (n:{label}) WHERE lower(n.name) CONTAINS lower($kw) RETURN n.name LIMIT 1",
            {"kw": raw}
        )
        if rows:
            return rows[0][0]
        try:
            from m6_rag_retrieval.vector_store import get_store
            result = get_store().find_entity(raw, label.lower(), threshold=0.72)
            if result:
                logger.debug(f"[M6] Vector resolved '{raw}' → '{result}' (label={label})")
                return result
        except Exception as ex:
            logger.debug(f"[M6] Vector store lookup failed: {ex}")
        return raw

    def retrieve(self, intent: dict) -> dict:
        itype      = intent.get("type", "general")
        raw_entity = intent.get("entity", "")

        label_map = {
            "company_disruption": "Company",
            "country_disruption": "Country",
            "mineral_disruption": "Mineral",
            "product_supply":     "Product",
        }
        label  = label_map.get(itype)
        entity = self.resolve_name(raw_entity, label) if label else raw_entity

        subgraph: dict[str, Any] = {
            "query_entity":   entity,
            "intent_type":    itype,
            "companies":      [],
            "countries":      [],
            "minerals":       [],
            "products":       [],
            "risk_events":    [],
            "supply_edges":   [],
            "produces_edges": [],
            "depends_edges":  [],
            "affects_edges":  [],
        }

        if itype == "company_disruption":
            self._company_disruption(entity, subgraph)
        elif itype == "country_disruption":
            self._country_disruption(entity, subgraph)
        elif itype == "mineral_disruption":
            self._mineral_disruption(entity, subgraph)
        elif itype == "product_supply":
            self._product_supply(entity, subgraph, raw_keyword=raw_entity)
        else:
            self._general_search(entity, subgraph)

        return subgraph

    def _company_disruption(self, company: str, sg: dict):
        rows = self._q(
            "MATCH (sup:Company)-[r:SUPPLIES]->(cust:Company) "
            "WHERE sup.name = $name "
            "RETURN cust.name, r.evidence, r.confidence",
            {"name": company}
        )
        for row in rows:
            sg["supply_edges"].append({
                "from": company, "to": row[0],
                "type": "SUPPLIES", "evidence": row[1], "confidence": row[2]
            })
            self._add_company(sg, row[0])

        rows = self._q(
            "MATCH (sup:Company)-[r:MANUFACTURES_FOR]->(cust:Company) "
            "WHERE sup.name = $name "
            "RETURN cust.name, r.evidence, r.confidence",
            {"name": company}
        )
        for row in rows:
            sg["supply_edges"].append({
                "from": company, "to": row[0],
                "type": "MANUFACTURES_FOR", "evidence": row[1], "confidence": row[2]
            })
            self._add_company(sg, row[0])

        rows = self._q(
            "MATCH (sup:Company)-[r:SUPPLIES]->(cust:Company) "
            "WHERE cust.name = $name "
            "RETURN sup.name, r.evidence, r.confidence",
            {"name": company}
        )
        for row in rows:
            sg["supply_edges"].append({
                "from": row[0], "to": company,
                "type": "SUPPLIES", "evidence": row[1], "confidence": row[2]
            })
            self._add_company(sg, row[0])

        rows = self._q(
            "MATCH (sup:Company)-[r:MANUFACTURES_FOR]->(cust:Company) "
            "WHERE cust.name = $name "
            "RETURN sup.name, r.evidence, r.confidence",
            {"name": company}
        )
        for row in rows:
            sg["supply_edges"].append({
                "from": row[0], "to": company,
                "type": "MANUFACTURES_FOR", "evidence": row[1], "confidence": row[2]
            })
            self._add_company(sg, row[0])

        rows = self._q(
            "MATCH (p:Product)-[r:DEPENDS_ON]->(c:Company) "
            "WHERE c.name = $name "
            "RETURN p.name, r.evidence",
            {"name": company}
        )
        for row in rows:
            sg["depends_edges"].append({"product": row[0], "company": company, "evidence": row[1]})
            self._add_product(sg, row[0])

        rows = self._q(
            "MATCH (c:Company)-[r:LOCATED_IN]->(co:Country) "
            "WHERE c.name = $name "
            "RETURN co.name",
            {"name": company}
        )
        for row in rows:
            self._add_country(sg, row[0])

        for country in [c["name"] for c in sg["countries"]]:
            self._fetch_risk_events_for_country(country, sg)

        self._add_company(sg, company)

    def _country_disruption(self, country: str, sg: dict):
        rows = self._q(
            "MATCH (c:Country)-[r:PRODUCES]->(m:Mineral) "
            "WHERE c.name = $name "
            "RETURN m.name, r.production_mt_2023, r.reserves_mt",
            {"name": country}
        )
        for row in rows:
            sg["produces_edges"].append({
                "country": country, "mineral": row[0],
                "production_mt_2023": row[1], "reserves_mt": row[2]
            })
            self._add_mineral(sg, row[0])

        for mineral in [m["name"] for m in sg["minerals"]]:
            rows = self._q(
                "MATCH (c:Country)-[r:PRODUCES]->(m:Mineral) "
                "WHERE m.name = $mineral AND c.name <> $country "
                "RETURN c.name, r.production_mt_2023 "
                "ORDER BY r.production_mt_2023 DESC LIMIT 5",
                {"mineral": mineral, "country": country}
            )
            for row in rows:
                sg["produces_edges"].append({
                    "country": row[0], "mineral": mineral,
                    "production_mt_2023": row[1], "reserves_mt": None
                })
                self._add_country(sg, row[0])

        rows = self._q(
            "MATCH (c:Company)-[r:LOCATED_IN]->(co:Country) "
            "WHERE co.name = $name "
            "RETURN c.name LIMIT 20",
            {"name": country}
        )
        for row in rows:
            self._add_company(sg, row[0])

        self._fetch_risk_events_for_country(country, sg)
        self._add_country(sg, country)

    def _mineral_disruption(self, mineral: str, sg: dict):
        rows = self._q(
            "MATCH (c:Country)-[r:PRODUCES]->(m:Mineral) "
            "WHERE m.name = $name "
            "RETURN c.name, r.production_mt_2023, r.reserves_mt "
            "ORDER BY r.production_mt_2023 DESC",
            {"name": mineral}
        )
        for row in rows:
            sg["produces_edges"].append({
                "country": row[0], "mineral": mineral,
                "production_mt_2023": row[1], "reserves_mt": row[2]
            })
            self._add_country(sg, row[0])

        for country in [c["name"] for c in sg["countries"][:3]]:
            self._fetch_risk_events_for_country(country, sg)

        self._add_mineral(sg, mineral)

    def _product_supply(self, product: str, sg: dict, raw_keyword: str = ""):
        rows = self._q(
            "MATCH (p:Product)-[r:DEPENDS_ON]->(c:Company) "
            "WHERE p.name = $name "
            "RETURN p.name, c.name, r.evidence",
            {"name": product}
        )
        for row in rows:
            sg["depends_edges"].append({"product": row[0], "company": row[1], "evidence": row[2]})
            self._add_company(sg, row[1])

        keyword = raw_keyword if raw_keyword else product
        contains_rows = self._q(
            "MATCH (p:Product)-[r:DEPENDS_ON]->(c:Company) "
            "WHERE lower(p.name) CONTAINS lower($kw) "
            "RETURN DISTINCT p.name, c.name, r.evidence LIMIT 500",
            {"kw": keyword}
        )
        for row in contains_rows:
            sg["depends_edges"].append({"product": row[0], "company": row[1], "evidence": row[2]})
            self._add_company(sg, row[1])

        for company in [c["name"] for c in sg["companies"]]:
            rows = self._q(
                "MATCH (c:Company)-[:LOCATED_IN]->(co:Country) "
                "WHERE c.name = $name RETURN co.name",
                {"name": company}
            )
            for row in rows:
                self._add_country(sg, row[0])

        self._add_product(sg, product)

    def _general_search(self, entity: str, sg: dict):
        try:
            from m6_rag_retrieval.vector_store import get_store
            vs = get_store()
            hits = vs.search(entity, top_k=10, threshold=0.35)
            type_to_key = {
                "company":  "companies",
                "country":  "countries",
                "mineral":  "minerals",
                "product":  "products",
            }
            for hit in hits:
                key = type_to_key.get(hit["type"])
                if key:
                    self._add_to(sg, key, {"name": hit["name"]})
        except Exception as ex:
            logger.debug(f"[M6] Vector store general search failed: {ex}")
            for label, key in [("Company", "companies"), ("Country", "countries"),
                               ("Mineral", "minerals"), ("Product", "products")]:
                rows = self._q(
                    f"MATCH (n:{label}) WHERE n.name CONTAINS $kw RETURN n.name LIMIT 10",
                    {"kw": entity}
                )
                for row in rows:
                    self._add_to(sg, key, {"name": row[0]})

        for company in [c["name"] for c in sg["companies"][:5]]:
            self._company_disruption(company, sg)

    def _fetch_risk_events_for_country(self, country: str, sg: dict):
        rows = self._q(
            "MATCH (e:RiskEvent)-[r:AFFECTS]->(c:Country) "
            "WHERE c.name = $name "
            "RETURN e.name, e.event_label, e.date, r.goldstein, r.event_code, e.source_url "
            "ORDER BY r.goldstein ASC LIMIT 10",
            {"name": country}
        )
        for row in rows:
            ev = {
                "name":        row[0],
                "event_label": row[1],
                "date":        row[2],
                "goldstein":   row[3],
                "event_code":  row[4],
                "source_url":  row[5],
                "affects":     country,
            }
            if ev not in sg["risk_events"]:
                sg["risk_events"].append(ev)

    def _q(self, cypher: str, params: dict) -> list:
        try:
            result = self.conn.execute(cypher, params)
            rows = []
            while result.has_next():
                rows.append(result.get_next())
            return rows
        except Exception as ex:
            logger.debug(f"[M6] Query failed: {ex}\n  {cypher}")
            return []

    def _add_company(self, sg: dict, name: str):
        self._add_to(sg, "companies", {"name": name})

    def _add_country(self, sg: dict, name: str):
        self._add_to(sg, "countries", {"name": name})

    def _add_mineral(self, sg: dict, name: str):
        self._add_to(sg, "minerals", {"name": name})

    def _add_product(self, sg: dict, name: str):
        self._add_to(sg, "products", {"name": name})

    def _add_to(self, sg: dict, key: str, item: dict):
        if item not in sg[key]:
            sg[key].append(item)
