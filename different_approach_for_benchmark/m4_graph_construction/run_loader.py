import argparse
import json
import logging
import sys

import kuzu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("m4_loading.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def verify(conn: kuzu.Connection):
    print("\n--- Graph Verification ---")

    checks = [
        ("Total nodes",     "MATCH (n) RETURN count(n) AS c"),
        ("Companies",       "MATCH (n:Company) RETURN count(n) AS c"),
        ("Countries",       "MATCH (n:Country) RETURN count(n) AS c"),
        ("Minerals",        "MATCH (n:Mineral) RETURN count(n) AS c"),
        ("Products",        "MATCH (n:Product) RETURN count(n) AS c"),
        ("RiskEvents",      "MATCH (n:RiskEvent) RETURN count(n) AS c"),
        ("Total edges",     "MATCH ()-[r]->() RETURN count(r) AS c"),
        ("SUPPLIES",        "MATCH ()-[r:SUPPLIES]->() RETURN count(r) AS c"),
        ("MANUFACTURES_FOR","MATCH ()-[r:MANUFACTURES_FOR]->() RETURN count(r) AS c"),
        ("LOCATED_IN",      "MATCH ()-[r:LOCATED_IN]->() RETURN count(r) AS c"),
        ("PRODUCES",        "MATCH ()-[r:PRODUCES]->() RETURN count(r) AS c"),
        ("DEPENDS_ON",      "MATCH ()-[r:DEPENDS_ON]->() RETURN count(r) AS c"),
        ("AFFECTS",         "MATCH ()-[r:AFFECTS]->() RETURN count(r) AS c"),
    ]

    for label, cypher in checks:
        try:
            result = conn.execute(cypher)
            count  = result.get_next()[0]
            print(f"  {label:20s}: {count}")
        except Exception as e:
            print(f"  {label:20s}: ERROR — {e}")

    print("\n  Sample — TSMC relationships:")
    for rel in ("SUPPLIES", "MANUFACTURES_FOR", "LOCATED_IN"):
        try:
            result = conn.execute(
                f"MATCH (a)-[r:{rel}]->(b) "
                f"WHERE a.name = 'TSMC' OR b.name = 'TSMC' "
                f"RETURN a.name, b.name, r.evidence LIMIT 3"
            )
            while result.has_next():
                row = result.get_next()
                print(f"    {row[0]:25s} --{rel}--> {row[1]}")
                print(f"      {str(row[2])[:100]}")
        except Exception as e:
            print(f"    {rel}: {e}")

    print("\n  Sample — China disruption path:")
    try:
        result = conn.execute(
            "MATCH (e:RiskEvent)-[:AFFECTS]->(c:Country {name: 'China'}) "
            "RETURN e.name, e.event_label, e.goldstein "
            "ORDER BY e.goldstein ASC LIMIT 5"
        )
        while result.has_next():
            row = result.get_next()
            print(f"    {row[0]:40s}  label={row[1]}  goldstein={row[2]}")
    except Exception as e:
        print(f"    ERROR: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    from m4_graph_construction.loader import GraphLoader
    from pathlib import Path

    loader = GraphLoader()
    n_nodes, n_rels = loader.build()

    print(f"\n--- M4 Build Summary ---")
    print(f"  Nodes inserted    : {n_nodes}")
    print(f"  Edges inserted    : {n_rels}")
    print(f"  Graph location    : data/graph/")

    if args.verify:
        verify(loader.conn)


if __name__ == "__main__":
    main()
