"""FastAPI application entry point.

Creates the app, registers routes, initializes all components on
startup, and provides clean shutdown.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from app.api.routes import all_routers
from app.api.schemas import ErrorResponse
from app.core.pipeline import RAGPipeline
from app.infrastructure.config import get_settings
from app.infrastructure.database import Database
from app.infrastructure.observability import get_logger, metrics
from app.ingestion.workers.ingestion_worker import IngestionWorker
from app.models.embedder import Embedder
from app.models.llm_client import LLMClient

logger = get_logger("main")

# API key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown."""
    settings = get_settings()

    # ----- STARTUP -----
    logger.info("app_starting", host=settings.server.host, port=settings.server.port)

    # Ensure directories exist
    Path(settings.database.db_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.observability.log_file).parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    db = Database(settings.database.db_path)
    await db.initialize()
    app.state.db = db

    # Initialize embedder
    try:
        embedder = Embedder(
            model_path=settings.embedding.model_path,
            dim=settings.embedding.dim,
            num_threads=settings.embedding.onnx_num_threads,
        )
        app.state.embedder = embedder
        logger.info("embedder_loaded")
    except FileNotFoundError as e:
        logger.error("embedder_not_found", error=str(e))
        app.state.embedder = None
        embedder = None  # type: ignore

    # Initialize LLM client
    llm = LLMClient(
        provider=settings.llm.provider,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        api_key=settings.llm.api_key,
        timeout=settings.llm.timeout_seconds,
    )
    app.state.llm = llm

    # Initialize pipeline (requires embedder)
    if embedder is not None:
        pipeline = RAGPipeline(db=db, embedder=embedder, llm=llm, settings=settings)
        app.state.pipeline = pipeline

        # Load procedural rules
        rules_path = Path(__file__).parent.parent / "configs" / "procedural_rules.yaml"
        if rules_path.exists():
            count = await pipeline.rules.load_rules_from_yaml(rules_path)
            logger.info("procedural_rules_loaded", count=count)

        # Start ingestion worker
        cpu_semaphore = threading.Semaphore(1)
        worker = IngestionWorker(
            db=db, embedder=embedder, settings=settings,
            cpu_semaphore=cpu_semaphore,
        )
        worker.start()
        app.state.ingestion_worker = worker
        logger.info("ingestion_worker_started")
    else:
        app.state.pipeline = None
        app.state.ingestion_worker = None
        logger.warning("pipeline_not_initialized", reason="Embedder not loaded")

    logger.info("app_started")
    yield

    # ----- SHUTDOWN -----
    logger.info("app_shutting_down")

    if hasattr(app.state, "ingestion_worker") and app.state.ingestion_worker:
        app.state.ingestion_worker.stop()
        app.state.ingestion_worker.join(timeout=5)

    if hasattr(app.state, "db"):
        app.state.db.close()

    logger.info("app_stopped")


# Create FastAPI application
app = FastAPI(
    title="RAG System",
    description="Production-grade Retrieval-Augmented Generation system",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all routers
for router in all_routers:
    app.include_router(router)


# API key middleware
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """Verify API key for all /v1/ endpoints."""
    if request.url.path.startswith("/v1/") and request.url.path != "/v1/admin/health":
        settings = get_settings()
        api_key = request.headers.get("X-API-Key", "")
        if settings.server.api_key != "change-me-in-production" and api_key != settings.server.api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": "Invalid or missing API key"},
            )

    # Check pipeline is initialized for pipeline-dependent endpoints
    if request.url.path.startswith("/v1/chat") or request.url.path.startswith("/v1/ingest"):
        if not hasattr(request.app.state, "pipeline") or request.app.state.pipeline is None:
            return JSONResponse(
                status_code=503,
                content={"error": "service_unavailable", "detail": "Pipeline not initialized. Embedder may not be loaded."},
            )

    response = await call_next(request)
    return response


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info."""
    return {
        "name": "RAG System",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/v1/admin/health",
    }


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    uvicorn.run(
        "app.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        workers=1,
    )
