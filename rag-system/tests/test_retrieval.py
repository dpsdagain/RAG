"""Tests for retrieval: hybrid search, RRF, reranking, cache."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from app.infrastructure.database import ChunkResult
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.semantic_cache import SemanticCache


class TestRRF:
    """Test Reciprocal Rank Fusion algorithm."""

    def test_rrf_single_list(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        sparse: list[tuple[str, float]] = []
        result = HybridSearchEngine.reciprocal_rank_fusion(dense, sparse, k=60)
        assert len(result) == 3
        assert result[0][0] == "a"

    def test_rrf_both_lists(self):
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 0.95), ("c", 0.85)]
        result = HybridSearchEngine.reciprocal_rank_fusion(dense, sparse, k=60)
        # "b" appears in both, should have highest score
        assert result[0][0] == "b"

    def test_rrf_top_n(self):
        dense = [(f"d{i}", 0.9 - i * 0.01) for i in range(50)]
        sparse = [(f"s{i}", 0.9 - i * 0.01) for i in range(50)]
        result = HybridSearchEngine.reciprocal_rank_fusion(
            dense, sparse, k=60, top_n=10
        )
        assert len(result) == 10

    def test_rrf_empty_lists(self):
        result = HybridSearchEngine.reciprocal_rank_fusion([], [], k=60)
        assert result == []


class TestSemanticCache:
    """Test semantic cache operations."""

    def test_store_and_lookup(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.99)
        vec = [1.0] * 384
        cache.store(vec, "test response", [{"chunk_id": "c1"}])

        # Exact match should hit
        result = cache.lookup(vec)
        assert result is not None
        assert result.response == "test response"

    def test_miss_below_threshold(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.99)
        vec1 = [1.0] + [0.0] * 383
        vec2 = [0.0] + [1.0] + [0.0] * 382

        cache.store(vec1, "response1", [])
        result = cache.lookup(vec2)
        assert result is None

    def test_lru_eviction(self):
        cache = SemanticCache(max_entries=2)
        cache.store([1.0] * 384, "r1", [])
        cache.store([0.5] * 384, "r2", [])
        cache.store([0.1] * 384, "r3", [])

        # Should have evicted oldest
        assert len(cache._entries) <= 2

    def test_invalidate_all(self):
        cache = SemanticCache()
        cache.store([1.0] * 384, "r1", [])
        cache.store([0.5] * 384, "r2", [])
        cache.invalidate_all()
        assert len(cache._entries) == 0

    def test_stats(self):
        cache = SemanticCache()
        stats = cache.get_stats()
        assert "hit_rate" in stats
        assert "entry_count" in stats

    def test_cosine_similarity(self):
        sim = SemanticCache.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 0.001

        sim = SemanticCache.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 0.001
