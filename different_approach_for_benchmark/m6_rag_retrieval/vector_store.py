import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME   = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR   = Path("data/graph/chroma")
COLLECTION   = "graph_entities"
GRAPH_DIR    = Path("data/graph/supply_chain.db")

_LEGACY_NPZ  = Path("data/graph/entity_embeddings.npz")


class VectorStore:

    def __init__(self):
        self._model      = None
        self._client     = None
        self._collection = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"[VectorStore] Loading embedding model {MODEL_NAME} ...")
            self._model = SentenceTransformer(MODEL_NAME)
            logger.info("[VectorStore] Model ready.")
        return self._model

    def _get_client(self):
        if self._client is None:
            import chromadb
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            logger.info(f"[VectorStore] ChromaDB client opened at {CHROMA_DIR}")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build(self, conn=None):
        import kuzu

        own_db = False
        if conn is None:
            db     = kuzu.Database(str(GRAPH_DIR))
            conn   = kuzu.Connection(db)
            own_db = True

        logger.info("[VectorStore] Fetching entities from graph ...")
        records = self._fetch_entities(conn)
        logger.info(f"[VectorStore] {len(records)} entities to embed.")

        model = self._get_model()
        logger.info("[VectorStore] Embedding ...")
        descs = [r["desc"] for r in records]
        embeddings = model.encode(
            descs,
            batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,
        ).tolist()

        client = self._get_client()
        try:
            client.delete_collection(COLLECTION)
            logger.info("[VectorStore] Dropped existing collection.")
        except Exception:
            pass

        self._collection = client.create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

        ids        = [str(i) for i in range(len(records))]
        metadatas  = [{"name": r["name"], "type": r["type"]} for r in records]
        documents  = descs

        BATCH = 5000
        for start in range(0, len(ids), BATCH):
            self._collection.add(
                ids        = ids[start : start + BATCH],
                embeddings = embeddings[start : start + BATCH],
                metadatas  = metadatas[start : start + BATCH],
                documents  = documents[start : start + BATCH],
            )

        logger.info(
            f"[VectorStore] Saved {len(records)} vectors to ChromaDB "
            f"collection '{COLLECTION}' at {CHROMA_DIR}."
        )

        if _LEGACY_NPZ.exists():
            _LEGACY_NPZ.unlink()
            logger.info(f"[VectorStore] Removed legacy {_LEGACY_NPZ}")

    def _fetch_entities(self, conn) -> list[dict]:
        records = []

        r = conn.execute("MATCH (n:Company) RETURN n.name, n.aliases LIMIT 10000")
        while r.has_next():
            row = r.get_next()
            name    = row[0] or ""
            aliases = []
            try:
                aliases = json.loads(row[1] or "[]")
            except Exception:
                pass
            alias_str = ", ".join(aliases[:3]) if aliases else ""
            desc = f"Company: {name}" + (f" (also known as {alias_str})" if alias_str else "")
            records.append({"name": name, "type": "company", "desc": desc})

        r = conn.execute("MATCH (n:Country) RETURN n.name LIMIT 10000")
        while r.has_next():
            name = r.get_next()[0] or ""
            records.append({"name": name, "type": "country", "desc": f"Country: {name}"})

        r = conn.execute("MATCH (n:Mineral) RETURN n.name LIMIT 1000")
        while r.has_next():
            name = r.get_next()[0] or ""
            records.append({"name": name, "type": "mineral", "desc": f"Critical mineral for electronics: {name}"})

        r = conn.execute("MATCH (n:Product) RETURN n.name LIMIT 10000")
        while r.has_next():
            name = r.get_next()[0] or ""
            records.append({"name": name, "type": "product", "desc": f"Consumer electronics product: {name}"})

        r = conn.execute(
            "MATCH (n:RiskEvent) RETURN n.name, n.event_label, n.country LIMIT 10000"
        )
        while r.has_next():
            row   = r.get_next()
            name  = row[0] or ""
            label = row[1] or ""
            ctry  = row[2] or ""
            desc  = f"Geopolitical risk event: {label} in {ctry}. {name}"
            records.append({"name": name, "type": "risk_event", "desc": desc})

        return [r for r in records if r["name"].strip()]

    def load(self) -> bool:
        try:
            col = self._get_collection()
            count = col.count()
            if count == 0:
                return False
            logger.info(
                f"[VectorStore] Loaded collection '{COLLECTION}' "
                f"({count} vectors) from {CHROMA_DIR}."
            )
            return True
        except Exception as ex:
            logger.debug(f"[VectorStore] Load failed: {ex}")
            return False

    def load_or_build(self, conn=None):
        if not self.load():
            logger.info("[VectorStore] No collection found — building ...")
            self.build(conn)

    def search(
        self,
        query: str,
        top_k: int = 10,
        entity_type: str | None = None,
        threshold: float = 0.35,
    ) -> list[dict]:
        col = self._get_collection()
        if col.count() == 0:
            return []

        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True)[0].tolist()

        where = {"type": entity_type} if entity_type else None

        try:
            result = col.query(
                query_embeddings=[q_vec],
                n_results=min(top_k, col.count()),
                where=where,
                include=["metadatas", "distances"],
            )
        except Exception as ex:
            logger.debug(f"[VectorStore] Query failed: {ex}")
            return []

        hits = []
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        for meta, dist in zip(metadatas, distances):
            score = round(1.0 - dist, 4)
            if score < threshold:
                continue
            hits.append({
                "name":  meta["name"],
                "type":  meta["type"],
                "score": score,
            })

        return hits

    def find_entity(
        self,
        name: str,
        entity_type: str,
        threshold: float = 0.82,
    ) -> str | None:
        type_prefix = {
            "company":    "Company",
            "country":    "Country",
            "mineral":    "Critical mineral for electronics",
            "product":    "Consumer electronics product",
            "risk_event": "Geopolitical risk event",
        }
        prefix = type_prefix.get(entity_type, entity_type.capitalize())
        query  = f"{prefix}: {name}"

        results = self.search(query, top_k=1, entity_type=entity_type, threshold=threshold)
        if results:
            logger.debug(
                f"[VectorStore] '{name}' → '{results[0]['name']}' "
                f"(score={results[0]['score']:.3f})"
            )
            return results[0]["name"]
        return None


_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
        _store.load_or_build()
    return _store


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import kuzu
    db   = kuzu.Database(str(GRAPH_DIR))
    conn = kuzu.Connection(db)

    vs = VectorStore()
    vs.build(conn)

    print("\n--- Semantic search demos ---")
    tests = [
        ("Taiwan semiconductor foundry", None),
        ("chip manufacturing fab", "company"),
        ("rare earth elements", "mineral"),
        ("political instability conflict", "risk_event"),
        ("smartphone Apple iOS device", "product"),
        ("gallium export ban", None),
        ("TSMC Taiwan fab", "company"),
        ("Global Foundries chip maker", "company"),
    ]
    for query, etype in tests:
        results = vs.search(query, top_k=3, entity_type=etype)
        label = f"[{etype}]" if etype else "[any]"
        print(f"\n  Query {label}: '{query}'")
        for r in results:
            print(f"    {r['score']:.3f}  {r['type']:12s}  {r['name']}")
