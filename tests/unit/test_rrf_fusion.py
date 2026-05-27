"""Tests for Reciprocal Rank Fusion — the heart of hybrid retrieval."""
from __future__ import annotations

import pytest

from app.retrieval.hybrid_search import HybridSearchEngine


rrf = HybridSearchEngine.reciprocal_rank_fusion


class TestRRF:
    def test_empty_lists_returns_empty(self):
        assert rrf([], [], k=60) == []

    def test_single_list_preserves_order(self):
        dense = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        result = rrf(dense, [], k=60)
        assert [cid for cid, _ in result] == ["a", "b", "c"]

    def test_item_in_both_lists_outranks_solo(self):
        # "b" appears in both → its RRF score sums two contributions
        dense = [("a", 0.9), ("b", 0.8)]
        sparse = [("b", 0.95), ("c", 0.85)]
        result = rrf(dense, sparse, k=60)
        assert result[0][0] == "b"

    def test_rrf_score_uses_rank_not_raw_score(self):
        # Raw scores don't enter the formula — only positional rank.
        # Item ranked #1 in both lists must score 2 * 1/(k+1).
        result = rrf(
            [("x", 0.001)],  # tiny raw score but rank 1
            [("x", 999.0)],  # huge raw score, also rank 1
            k=60,
        )
        expected = 2.0 / 61.0
        assert abs(result[0][1] - expected) < 1e-9

    def test_top_n_caps_results(self):
        dense = [(f"d{i}", 1.0 - i * 0.01) for i in range(50)]
        sparse = [(f"s{i}", 1.0 - i * 0.01) for i in range(50)]
        result = rrf(dense, sparse, k=60, top_n=10)
        assert len(result) == 10

    def test_k_parameter_dampens_contribution(self):
        # Larger k makes lower-ranked items relatively more competitive.
        list_a = [(f"a{i}", 0) for i in range(10)]
        list_b = [(f"a{i}", 0) for i in range(10)]
        small_k = rrf(list_a, list_b, k=10)
        large_k = rrf(list_a, list_b, k=1000)
        # Top item's score under small k must exceed that under large k.
        assert small_k[0][1] > large_k[0][1]

    def test_all_returned_items_have_positive_score(self):
        result = rrf([("a", 0)], [("b", 0)], k=60)
        for _, score in result:
            assert score > 0

    def test_deterministic(self):
        # Same input → identical output (no ordering instability).
        for _ in range(3):
            r = rrf([("a", 0), ("b", 0)], [("b", 0), ("c", 0)], k=60)
            assert [cid for cid, _ in r] == ["b", "a", "c"]
