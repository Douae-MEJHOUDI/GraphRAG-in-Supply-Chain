import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).parent.parent / "m9_ui"

_retriever = None
_generator = None


def get_retriever():
    global _retriever
    if _retriever is None:
        from m6_rag_retrieval.retriever import GraphRetriever
        _retriever = GraphRetriever()
    return _retriever


def get_generator():
    global _generator
    if _generator is None:
        from m7_report_generation.report_generator import ReportGenerator
        _generator = ReportGenerator()
    return _generator


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_retriever()
        logger.info("[M8] Graph connection ready.")
    except Exception as e:
        logger.error(f"[M8] Failed to connect to graph: {e}")
    yield


app = FastAPI(
    title="Supply Chain GraphRAG API",
    description="What-if impact analysis for electronics supply chains.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    context_only: bool = False


class QueryResponse(BaseModel):
    question:       str
    intent_type:    str
    query_entity:   str
    confidence:     str
    evidence_count: int
    report:         str
    context:        str


class NodeResponse(BaseModel):
    name:       str
    type:       str
    properties: dict[str, Any]
    neighbors:  list[dict[str, Any]]


class ImpactResponse(BaseModel):
    entity:        str
    supply_edges:  list[dict]
    depends_edges: list[dict]
    countries:     list[dict]
    risk_events:   list[dict]


@app.get("/health")
def health():
    try:
        r = get_retriever()
        rows = r._q("MATCH (n) RETURN count(n) AS c", {})
        node_count = rows[0][0] if rows else 0
        rows = r._q("MATCH ()-[e]->() RETURN count(e) AS c", {})
        edge_count = rows[0][0] if rows else 0
        return {"status": "ok", "nodes": node_count, "edges": edge_count}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    from m6_rag_retrieval.context_builder import parse_query, build_context

    intent    = parse_query(req.question)
    retriever = get_retriever()
    subgraph  = retriever.retrieve(intent)
    context   = build_context(subgraph)

    if req.context_only:
        return QueryResponse(
            question=req.question,
            intent_type=intent["type"],
            query_entity=intent["entity"],
            confidence="N/A",
            evidence_count=0,
            report="",
            context=context,
        )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not configured on server.",
        )

    generator = get_generator()
    report    = generator.generate(req.question, subgraph, context)

    return QueryResponse(
        question=req.question,
        intent_type=report.intent_type,
        query_entity=report.query_entity,
        confidence=report.confidence,
        evidence_count=report.evidence_count,
        report=report.to_markdown(),
        context=context,
    )


@app.get("/graph/node/{name}", response_model=NodeResponse)
def get_node(name: str):
    r = get_retriever()
    for label in ("Company", "Country", "Mineral", "Product", "RiskEvent"):
        rows = r._q(f"MATCH (n:{label}) WHERE n.name = $name RETURN n.*", {"name": name})
        if rows:
            neighbors = []
            for rel in ("SUPPLIES", "MANUFACTURES_FOR", "LOCATED_IN", "PRODUCES", "DEPENDS_ON", "AFFECTS"):
                nr = r._q(
                    f"MATCH (n:{label})-[:{rel}]->(m) WHERE n.name = $name RETURN m.name, '{rel}' LIMIT 10",
                    {"name": name}
                )
                for row in nr:
                    neighbors.append({"name": row[0], "relationship": row[1], "direction": "outgoing"})
                nr = r._q(
                    f"MATCH (m)-[:{rel}]->(n:{label}) WHERE n.name = $name RETURN m.name, '{rel}' LIMIT 10",
                    {"name": name}
                )
                for row in nr:
                    neighbors.append({"name": row[0], "relationship": row[1], "direction": "incoming"})
            return NodeResponse(name=name, type=label, properties={}, neighbors=neighbors)

    raise HTTPException(status_code=404, detail=f"Entity '{name}' not found in graph.")


@app.get("/graph/impact/{name}", response_model=ImpactResponse)
def get_impact(name: str):
    from m6_rag_retrieval.context_builder import parse_query

    intent   = parse_query(f"What if {name} stops production?")
    r        = get_retriever()
    subgraph = r.retrieve(intent)

    return ImpactResponse(
        entity=name,
        supply_edges=subgraph.get("supply_edges", []),
        depends_edges=subgraph.get("depends_edges", []),
        countries=subgraph.get("countries", []),
        risk_events=subgraph.get("risk_events", []),
    )


@app.get("/")
def serve_ui():
    return FileResponse(UI_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("m8_api.app:app", host="0.0.0.0", port=8000, reload=False)
