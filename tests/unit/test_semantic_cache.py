"""Tests for the in-process SemanticCache."""
from __future__ import annotations

import time

import pytest

from app.retrieval.semantic_cache import SemanticCache


class TestLookup:
    def test_empty_cache_returns_none(self):
        cache = SemanticCache(max_entries=10)
        assert cache.lookup([1.0] * 384) is None

    def test_exact_match_hits(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.99)
        vec = [1.0] + [0.0] * 383
        cache.store(vec, "the response", [{"chunk_id": "c1"}])
        hit = cache.lookup(vec)
        assert hit is not None
        assert hit.response == "the response"

    def test_orthogonal_query_misses(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.95)
        cache.store([1.0] + [0.0] * 383, "r1", [])
        # Orthogonal vector — similarity 0
        assert cache.lookup([0.0, 1.0] + [0.0] * 382) is None

    def test_threshold_is_strict(self):
        # Two vectors whose cosine similarity is ~0.7 should miss at 0.95.
        cache = SemanticCache(max_entries=10, similarity_threshold=0.95)
        a = [1.0, 0.0] + [0.0] * 382
        b = [0.7, 0.7] + [0.0] * 382  # cos_sim ≈ 0.7
        cache.store(a, "r1", [])
        assert cache.lookup(b) is None


class TestEviction:
    def test_lru_eviction_caps_size(self):
        cache = SemanticCache(max_entries=2)
        cache.store([1.0] * 384, "r1", [])
        cache.store([0.5] * 384, "r2", [])
        cache.store([0.1] * 384, "r3", [])
        assert len(cache._entries) <= 2

    def test_ttl_expiry(self):
        # ttl=0 → every entry is immediately expired
        cache = SemanticCache(max_entries=10, similarity_threshold=0.99, ttl_seconds=0)
        cache.store([1.0] * 384, "r1", [])
        time.sleep(0.01)
        assert cache.lookup([1.0] * 384) is None


class TestStats:
    def test_stats_keys_present(self):
        stats = SemanticCache().get_stats()
        for key in ("hit_rate", "entry_count", "total_lookups", "total_hits"):
            assert key in stats

    def test_hit_rate_tracks_hits(self):
        cache = SemanticCache(max_entries=10, similarity_threshold=0.99)
        vec = [1.0] + [0.0] * 383
        cache.store(vec, "r", [])
        cache.lookup(vec)
        cache.lookup(vec)
        cache.lookup([0.0, 1.0] + [0.0] * 382)  # miss
        stats = cache.get_stats()
        assert stats["total_lookups"] == 3
        assert stats["total_hits"] == 2
        assert 0.6 < stats["hit_rate"] < 0.7


class TestInvalidate:
    def test_invalidate_clears_everything(self):
        cache = SemanticCache(max_entries=10)
        for i in range(3):
            cache.store([float(i)] * 384, f"r{i}", [])
        cache.invalidate_all()
        assert cache._entries == []


class TestCosineHelper:
    def test_identical_vectors(self):
        sim = SemanticCache.cosine_similarity([1, 0, 0], [1, 0, 0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        sim = SemanticCache.cosine_similarity([1, 0, 0], [0, 1, 0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors(self):
        sim = SemanticCache.cosine_similarity([1, 0, 0], [-1, 0, 0])
        assert abs(sim + 1.0) < 1e-6

    def test_zero_vector_does_not_crash(self):
        # Avoid divide-by-zero — the implementation must guard this.
        sim = SemanticCache.cosine_similarity([0, 0, 0], [1, 0, 0])
        assert sim == 0.0
