import json
import logging
import re
from pathlib import Path

import kuzu

from m4_graph_construction.schema import NODE_TABLES, REL_TABLES

logger = logging.getLogger(__name__)

CANONICAL_DIR = Path("data/canonical")
GRAPH_DIR     = Path("data/graph/supply_chain.db")

TYPE_TO_TABLE = {
    "company":    "Company",
    "country":    "Country",
    "mineral":    "Mineral",
    "product":    "Product",
    "risk_event": "RiskEvent",
}

REL_TYPE_TO_TABLE = {
    "SUPPLIES":         "SUPPLIES",
    "MANUFACTURES_FOR": "MANUFACTURES_FOR",
    "LOCATED_IN":       "LOCATED_IN",
    "PRODUCES":         "PRODUCES",
    "DEPENDS_ON":       "DEPENDS_ON",
    "AFFECTS":          "AFFECTS",
}


class GraphLoader:

    def __init__(self, graph_dir: Path = GRAPH_DIR):
        graph_dir.parent.mkdir(parents=True, exist_ok=True)
        self.db   = kuzu.Database(str(graph_dir))
        self.conn = kuzu.Connection(self.db)
        logger.info(f"[M4] Kuzu database at {graph_dir}")

    def build(self):
        self._create_schema()
        entities, relationships = self._load_canonical()
        n_nodes = self._insert_nodes(entities)
        n_rels  = self._insert_relationships(relationships, entities)
        return n_nodes, n_rels

    def _create_schema(self):
        logger.info("[M4] Creating schema ...")

        for table, cols in NODE_TABLES.items():
            col_defs = ", ".join(f"{c} {t}" for c, t in cols)
            cypher   = f"CREATE NODE TABLE IF NOT EXISTS {table} ({col_defs}, PRIMARY KEY (id))"
            self.conn.execute(cypher)
            logger.info(f"[M4]   Node table: {table}")

        for from_t, to_t, rel, props in REL_TABLES:
            prop_defs = "".join(f", {c} {t}" for c, t in props)
            cypher    = (
                f"CREATE REL TABLE IF NOT EXISTS {rel} "
                f"(FROM {from_t} TO {to_t}{prop_defs})"
            )
            self.conn.execute(cypher)
            logger.info(f"[M4]   Rel table:  {from_t} -[{rel}]-> {to_t}")

    def _load_canonical(self):
        with open(CANONICAL_DIR / "entities.json", encoding="utf-8") as f:
            entities = json.load(f)
        with open(CANONICAL_DIR / "relationships.json", encoding="utf-8") as f:
            relationships = json.load(f)
        logger.info(f"[M4] Loaded {len(entities)} entities, {len(relationships)} relationships")
        return entities, relationships

    def _insert_nodes(self, entities: list[dict]) -> int:
        inserted = 0
        skipped  = 0

        for e in entities:
            etype  = e.get("type", "")
            table  = TYPE_TO_TABLE.get(etype)
            if not table:
                skipped += 1
                continue

            props  = e.get("properties", {})
            params = self._build_node_params(table, e, props)

            try:
                cypher = self._node_insert_cypher(table, params)
                self.conn.execute(cypher, params)
                inserted += 1
            except Exception as ex:
                logger.debug(f"[M4] Node insert failed ({e['name']}): {ex}")
                skipped += 1

        logger.info(f"[M4] Nodes: {inserted} inserted, {skipped} skipped")
        return inserted

    def _build_node_params(self, table: str, e: dict, props: dict) -> dict:
        base = {
            "id":   e["id"],
            "name": e["name"],
        }

        if table == "Company":
            base["aliases"] = json.dumps(e.get("aliases", []))
            base["sources"] = json.dumps(e.get("sources", [])[:3])
            base["ticker"]  = props.get("ticker", "")
            base["role"]    = props.get("role", "")

        elif table == "Country":
            base["country_code"] = props.get("country_code", "")
            base["aliases"]      = json.dumps(e.get("aliases", []))

        elif table == "Mineral":
            pass

        elif table == "Product":
            base["guide_id"] = str(props.get("guide_id", ""))

        elif table == "RiskEvent":
            base["date"]        = props.get("date", "")
            base["event_code"]  = props.get("event_code", "")
            base["event_label"] = props.get("event_label", "")
            base["country"]     = props.get("country", "")
            base["location"]    = props.get("location", "")
            base["goldstein"]   = float(props.get("goldstein", 0.0) or 0.0)
            base["num_articles"]= int(props.get("num_articles", 0) or 0)
            base["source_url"]  = props.get("source_url", "")

        return base

    def _node_insert_cypher(self, table: str, params: dict) -> str:
        props = ", ".join(f"{k}: ${k}" for k in params.keys())
        return f"CREATE (:{table} {{{props}}})"

    def _insert_relationships(self, relationships: list[dict], entities: list[dict]) -> int:
        name_to_entity: dict[str, dict] = {}
        for e in entities:
            name_to_entity[e["name"]] = e

        inserted = 0
        skipped  = 0

        for r in relationships:
            rel_table = REL_TYPE_TO_TABLE.get(r["type"])
            if not rel_table:
                skipped += 1
                continue

            src_e = name_to_entity.get(r["source_entity"])
            tgt_e = name_to_entity.get(r["target_entity"])

            if not src_e or not tgt_e:
                skipped += 1
                continue

            src_table = TYPE_TO_TABLE.get(src_e["type"])
            tgt_table = TYPE_TO_TABLE.get(tgt_e["type"])

            if not src_table or not tgt_table:
                skipped += 1
                continue

            valid_combos = {(f, t, rn) for f, t, rn, _ in REL_TABLES}
            if (src_table, tgt_table, rel_table) not in valid_combos:
                skipped += 1
                continue

            props   = r.get("properties", {})
            ep      = self._build_rel_params(rel_table, r, props)

            try:
                cypher = (
                    f"MATCH (a:{src_table} {{id: $src_id}}), "
                    f"      (b:{tgt_table} {{id: $tgt_id}}) "
                    f"CREATE (a)-[:{rel_table} {{{', '.join(f'{k}: ${k}' for k in ep)}}}]->(b)"
                )
                self.conn.execute(cypher, {"src_id": src_e["id"], "tgt_id": tgt_e["id"], **ep})
                inserted += 1
            except Exception as ex:
                logger.debug(f"[M4] Rel insert failed ({r['source_entity']} -> {r['target_entity']}): {ex}")
                skipped += 1

        logger.info(f"[M4] Relationships: {inserted} inserted, {skipped} skipped")
        return inserted

    def _build_rel_params(self, rel_table: str, r: dict, props: dict) -> dict:
        evidence   = r.get("evidence", "")[:500]
        confidence = float(r.get("confidence", 1.0))

        if rel_table in ("SUPPLIES", "MANUFACTURES_FOR"):
            return {
                "evidence":   evidence,
                "confidence": confidence,
                "keyword":    props.get("keyword_matched", ""),
                "ticker":     props.get("ticker", ""),
            }
        elif rel_table == "LOCATED_IN":
            return {
                "evidence":   evidence,
                "confidence": confidence,
                "ticker":     props.get("ticker", ""),
            }
        elif rel_table == "PRODUCES":
            return {
                "production_mt_2023": str(props.get("production_mt_2023", "")),
                "reserves_mt":        str(props.get("reserves_mt", "")),
            }
        elif rel_table == "DEPENDS_ON":
            return {
                "evidence":   evidence,
                "confidence": confidence,
                "guide_id":   str(props.get("guide_id", "")),
            }
        elif rel_table == "AFFECTS":
            return {
                "evidence":   evidence,
                "confidence": confidence,
                "event_code": props.get("event_code", ""),
                "goldstein":  float(props.get("goldstein", 0.0) or 0.0),
                "date":       props.get("date", ""),
            }
        return {}
