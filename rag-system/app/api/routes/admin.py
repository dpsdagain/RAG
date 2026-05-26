"""Admin API routes: health, stats, cache management."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.schemas import StatsResponse, HealthResponse, FailedIngest, ErrorResponse
from app.infrastructure.observability import metrics

router = APIRouter(prefix="/v1/admin", tags=["Admin"])


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(request: Request):
    """Check system health status."""
    db_connected = False
    try:
        await request.app.state.db.fetch_one("SELECT 1")
        db_connected = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version="0.1.0",
        db_connected=db_connected,
        embedder_loaded=request.app.state.embedder is not None,
    )


@router.get("/stats", response_model=StatsResponse, summary="System statistics")
async def get_stats(request: Request):
    """Get system performance statistics."""
    db = request.app.state.db

    doc_count = await db.fetch_one("SELECT COUNT(*) as cnt FROM documents WHERE status='active'")
    chunk_count = await db.fetch_one("SELECT COUNT(*) as cnt FROM chunks")
    stats = metrics.get_stats()
    cache_stats = request.app.state.pipeline.cache.get_stats()

    latency_data = stats.get("latencies", {}).get("pipeline_total", {})
    avg_latency = latency_data.get("avg_ms", 0.0) if latency_data else 0.0

    return StatsResponse(
        total_documents=doc_count["cnt"] if doc_count else 0,
        total_chunks=chunk_count["cnt"] if chunk_count else 0,
        rss_mb=metrics.get_rss_mb(),
        cache_hit_rate=cache_stats.get("hit_rate", 0.0),
        avg_latency_ms=avg_latency,
        uptime_seconds=stats.get("uptime_seconds", 0.0),
        latency_stats=stats.get("latencies", {}),
        counters=stats.get("counters", {}),
    )


@router.get("/failed-ingests", response_model=list[FailedIngest], summary="Failed ingestions")
async def list_failed_ingests(request: Request, limit: int = 50):
    """List failed ingestion attempts."""
    rows = await request.app.state.db.fetch_all(
        "SELECT * FROM failed_ingests WHERE resolved = 0 ORDER BY failed_at DESC LIMIT ?",
        (limit,),
    )
    return [
        FailedIngest(
            failure_id=r["failure_id"],
            source_uri=r["source_uri"],
            error_message=r["error_message"],
            parser_attempted=r["parser_attempted"],
            failed_at=r["failed_at"],
            retry_count=r["retry_count"],
            resolved=bool(r["resolved"]),
        )
        for r in rows
    ]


@router.post("/cache/clear", summary="Clear semantic cache")
async def clear_cache(request: Request):
    """Clear the semantic cache."""
    request.app.state.pipeline.cache.invalidate_all()
    return {"status": "cache_cleared"}
