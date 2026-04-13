"""
app.py
------
FastAPI backend for the Supply Chain Risk Intelligence web interface.

Implements the UI:

  POST /analyze
    Body : { "query": "What if Taiwan has a conflict?" }
    Returns the structured risk report as JSON.

  GET  /health
    Returns API status and graph/cache stats.

  DELETE /cache
    Clears the semantic query cache.

Usage
-----
    # From the project root:
    uvicorn src.api.app:app --reload --port 8000

    # Or programmatically:
    from src.api.app import create_app
    app = create_app()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.run import build_pipeline
from src.cache.semantic_cache import CachedPipeline
from src.simulation.events import SCENARIO_LIBRARY


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    query:      str
    event_name: Optional[str] = None   # key in SCENARIO_LIBRARY; auto-detected if None


class AnalyzeResponse(BaseModel):
    event_name:     str
    query:          str
    confidence:     str            # HIGH / MEDIUM / LOW derived from cache hit + shock
    evidence_count: int
    critical_nodes: list[str]
    high_nodes:     list[str]
    top_exposed:    list[dict]     # [{name, score}, ...]
    risk_report:    str            # full Claude output text
    from_cache:     bool


class HealthResponse(BaseModel):
    status:       str
    graph_nodes:  int
    graph_edges:  int
    cache_size:   int
    scenarios:    list[str]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

_pipeline: Optional[CachedPipeline] = None


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(
        title="Supply Chain Risk Intelligence",
        description=(
            "Graph RAG-powered supply chain disruption analysis. "
            "Ask what-if questions about the S&P 500 supply network."
        ),
        version="1.0.0",
    )

    # Allow frontend (served from a different origin during development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve the static frontend files
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir / "static")),
            name="static",
        )

    # -----------------------------------------------------------------------
    # Startup: build the pipeline once (lazy — first request initialises it)
    # -----------------------------------------------------------------------

    def _get_pipeline() -> CachedPipeline:
        global _pipeline
        if _pipeline is None:
            print("[API] Initialising pipeline on first request...")
            _pipeline = build_pipeline()
        return _pipeline

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def root():
        index = Path(__file__).parent / "frontend" / "index.html"
        if index.exists():
            return FileResponse(str(index))
        return {"message": "Supply Chain Risk Intelligence API — see /docs"}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        pipe = _get_pipeline()
        G    = pipe.engine.G
        return HealthResponse(
            status="ok",
            graph_nodes=G.number_of_nodes(),
            graph_edges=G.number_of_edges(),
            cache_size=pipe.cache.size(),
            scenarios=list(SCENARIO_LIBRARY.keys()),
        )

    @app.post("/analyze", response_model=AnalyzeResponse)
    async def analyze(req: AnalyzeRequest):
        """
        Run the full pipeline (or return a cached result) for a what-if query.

        If `event_name` is provided it must be a key in SCENARIO_LIBRARY.
        Otherwise the API attempts to auto-detect the most relevant scenario
        by matching keywords in the query against scenario names/descriptions.
        """
        pipe = _get_pipeline()

        # Resolve event
        event_name = req.event_name
        if event_name:
            if event_name not in SCENARIO_LIBRARY:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown event '{event_name}'. "
                           f"Available: {list(SCENARIO_LIBRARY.keys())}",
                )
            event = SCENARIO_LIBRARY[event_name]
        else:
            event = _auto_detect_event(req.query)
            event_name = event.name

        # Check cache before calling full pipeline
        from_cache = pipe.cache.lookup(req.query) is not None

        # Run (cache hit returns instantly; miss runs the full pipeline)
        result = pipe.run(event=event, query=req.query)

        # Build confidence label from initial_shock
        shock = result.get("initial_shock", 0.5)
        if shock >= 0.80:
            confidence = "HIGH"
        elif shock >= 0.50:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        top_exposed = [
            {"name": name, "score": round(score, 3)}
            for name, score in result.get("top_10_exposed", [])
        ]

        # Count evidence records in the report text
        evidence_count = result.get("stats", {}).get("total_affected", 0)

        return AnalyzeResponse(
            event_name=event_name,
            query=req.query,
            confidence=confidence,
            evidence_count=evidence_count,
            critical_nodes=result.get("critical_nodes", []),
            high_nodes=result.get("high_nodes", []),
            top_exposed=top_exposed,
            risk_report=result.get("risk_report", ""),
            from_cache=from_cache,
        )

    @app.delete("/cache")
    async def clear_cache():
        """Wipe the semantic query cache."""
        _get_pipeline().cache.clear()
        return {"message": "Cache cleared."}

    return app


def _auto_detect_event(query: str):
    """
    Simple keyword-based event detection.
    Falls back to the first scenario in SCENARIO_LIBRARY if nothing matches.
    """
    q = query.lower()
    for key, event in SCENARIO_LIBRARY.items():
        keywords = [w.lower() for w in key.replace("_", " ").split()]
        if any(kw in q for kw in keywords):
            return event
    # Fallback to first scenario
    return next(iter(SCENARIO_LIBRARY.values()))


# ---------------------------------------------------------------------------
# Module-level app instance (for `uvicorn src.api.app:app`)
# ---------------------------------------------------------------------------

app = create_app()
