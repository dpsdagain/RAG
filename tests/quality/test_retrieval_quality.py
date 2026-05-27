"""Semantic retrieval quality tests against the real bge-small embedder.

These tests verify that for a known query, the right chunk content is
actually returned in the top-K results. The fake embedder in the rest of
the suite cannot test this — its vectors have no semantic structure.
"""
from __future__ import annotations

import pytest

from app.retrieval.hybrid_search import HybridSearchEngine

from .golden_set import GOLDEN_SET


pytestmark = [pytest.mark.requires_db, pytest.mark.requires_embedder, pytest.mark.slow]


def _top_k_contains(chunks, substrings: list[str], k: int = 5) -> bool:
    """Return True if any of `substrings` appears in any of the top-K chunks."""
    top = [c.content.lower() for c in chunks[:k]]
    return any(any(sub in content for sub in substrings) for content in top)


@pytest.mark.parametrize("case", GOLDEN_SET, ids=[c["id"] for c in GOLDEN_SET])
async def test_hybrid_search_finds_expected_chunk(case, ingested_corpus_db, real_embedder):
    """Hybrid search must surface the right section of the doc in top-5."""
    engine = HybridSearchEngine(ingested_corpus_db, real_embedder)
    results = await engine.search(case["query"], top_n=10)

    assert len(results) > 0, f"retrieval returned 0 hits for {case['id']!r}"
    assert _top_k_contains(results, case["must_retrieve"], k=5), (
        f"none of {case['must_retrieve']!r} found in top-5 results for "
        f"{case['query']!r}. Top results were:\n"
        + "\n".join(f"  - {r.content[:120]}" for r in results[:5])
    )


async def test_dense_search_returns_relevant_for_specific_query(
    ingested_corpus_db, real_embedder
):
    """Dense search alone should put a chunking-related chunk in the top-3."""
    embedding = real_embedder.embed("sentence-level semantic chunking algorithm")
    results = await ingested_corpus_db.vector_search(embedding, top_k=5)

    assert len(results) > 0
    top3 = " ".join(r.content.lower() for r in results[:3])
    assert "chunk" in top3 or "sentence" in top3


async def test_bm25_search_finds_literal_terms(ingested_corpus_db, real_embedder):
    """BM25 should match exact lexical terms when present in the corpus."""
    results = await ingested_corpus_db.bm25_search("FlashRank reranker", top_k=5)
    assert len(results) > 0
    joined = " ".join(r.content.lower() for r in results[:3])
    assert "flashrank" in joined or "rerank" in joined


async def test_unrelated_query_does_not_dominate_top_results(
    ingested_corpus_db, real_embedder
):
    """A clearly off-topic query should NOT surface architecture chunks high.

    This guards against the embedder collapsing all vectors close together,
    which would make retrieval useless.
    """
    engine = HybridSearchEngine(ingested_corpus_db, real_embedder)
    on_topic = await engine.search("hybrid retrieval and RRF fusion", top_n=3)
    off_topic = await engine.search("favorite recipe for chocolate cake", top_n=3)

    # On-topic top hit should have a higher score than off-topic top hit.
    if on_topic and off_topic:
        assert on_topic[0].score >= off_topic[0].score, (
            "off-topic query scored higher than on-topic — embeddings may be "
            "degenerate or RRF is broken"
        )
