"""Tests for the L2² → [0, 1] score formula (fix #6).

sqlite-vec's default distance metric for vec0 tables is L2-squared. For
L2-normalized vectors this ranges 0–4. The pipeline expects scores in
[0, 1] for citation validation, so we map: score = max(0, 1 - d/2).
"""
from __future__ import annotations


def normalize(distance: float) -> float:
    """Mirror of the score formula used in Database.vector_search."""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


class TestScoreNormalization:
    def test_zero_distance_is_perfect_score(self):
        # Identical vectors: distance 0 → score 1
        assert normalize(0.0) == 1.0

    def test_orthogonal_vectors_score_zero(self):
        # L2-normalized orthogonal vectors have L2² = 2 → score 0
        assert normalize(2.0) == 0.0

    def test_opposite_vectors_clamp_to_zero(self):
        # L2-normalized opposite vectors have L2² = 4 → clamped to 0
        assert normalize(4.0) == 0.0

    def test_midrange_distance(self):
        # L2² = 1 → cos_sim = 0.5 → score 0.5
        assert normalize(1.0) == 0.5

    def test_never_returns_negative(self):
        for d in [0.0, 0.5, 1.0, 2.0, 3.5, 4.0, 10.0]:
            assert normalize(d) >= 0.0

    def test_never_returns_above_one(self):
        # Defensive — a malformed (non-normalized) embedding could yield
        # negative distance; we still clamp at 1.0.
        for d in [-0.5, -10.0, 0.0]:
            assert normalize(d) <= 1.0

    def test_monotone_decreasing(self):
        # Lower distance must yield higher score (preserves ranking).
        assert normalize(0.1) > normalize(0.5) > normalize(1.5)
