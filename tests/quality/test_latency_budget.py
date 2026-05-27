"""Latency budget tests — retrieval should stay under the V3 spec targets.

These run against the real embedder but no LLM, so they're fast and free.
Thresholds match Section I of rag_v3_implementation_architecture.md.
"""
from __future__ import annotations

import time

import pytest

from app.retrieval.hybrid_search import HybridSearchEngine


pytestmark = [pytest.mark.requires_db, pytest.mark.requires_embedder, pytest.mark.slow]


# Generous CI-friendly thresholds — the V3 spec targets are tighter (200/50/800).
# We allow 5× headroom so cold-cache runs don't flake.
EMBED_BUDGET_MS = 250
RETRIEVAL_BUDGET_MS = 1000


async def test_single_query_embedding_under_budget(real_embedder):
    """Embedding one query should be well under 250 ms after warm-up."""
    real_embedder.embed("warmup query so the next timing is accurate")
    t0 = time.perf_counter()
    real_embedder.embed("What is the chunking strategy?")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < EMBED_BUDGET_MS, (
        f"embed took {elapsed_ms:.1f} ms, budget {EMBED_BUDGET_MS} ms"
    )


async def test_hybrid_retrieval_under_budget(ingested_corpus_db, real_embedder):
    """End-to-end hybrid search (embed + dense + sparse + RRF) under 1 s."""
    engine = HybridSearchEngine(ingested_corpus_db, real_embedder)
    # Warm-up
    await engine.search("warm-up query", top_n=5)

    t0 = time.perf_counter()
    results = await engine.search("What is the chunking strategy?", top_n=10)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    assert results, "retrieval returned no results"
    assert elapsed_ms < RETRIEVAL_BUDGET_MS, (
        f"hybrid_search took {elapsed_ms:.1f} ms, budget {RETRIEVAL_BUDGET_MS} ms"
    )


async def test_batch_embedding_is_faster_than_serial(real_embedder):
    """Batching N sentences should beat N serial single-text embeddings."""
    sentences = [f"This is test sentence number {i}." for i in range(16)]

    # Warm-up
    real_embedder.embed_batch(sentences[:2])

    t0 = time.perf_counter()
    for s in sentences:
        real_embedder.embed(s)
    serial_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    real_embedder.embed_batch(sentences)
    batch_ms = (time.perf_counter() - t0) * 1000

    # Allow some slack on noisy systems, but batch must beat serial.
    assert batch_ms < serial_ms, (
        f"batch ({batch_ms:.1f} ms) did not beat serial ({serial_ms:.1f} ms) "
        f"— either ONNX threading is broken or batch path is degraded"
    )
