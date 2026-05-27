"""Tests for the heuristic QueryRouter.

The router classifies queries WITHOUT calling the LLM — so it must be
deterministic and well-behaved on common inputs.
"""
from __future__ import annotations

import pytest

from app.core.query_router import QueryRouter


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


class TestConversational:
    def test_short_greeting(self, router):
        assert router.classify("hello") == "conversational"
        assert router.classify("hey there") == "conversational"
        assert router.classify("thanks") == "conversational"

    def test_short_acknowledgement(self, router):
        # 4 words, no question word → conversational
        assert router.classify("got it thanks bye") == "conversational"

    def test_short_with_question_word_escapes_conversational(self, router):
        # Even short queries route to retrieval if they're a question.
        assert router.classify("what is RAG") != "conversational"


class TestSimpleRetrieval:
    def test_standard_question(self, router):
        assert router.classify("What is the chunking strategy?") == "simple_retrieval"

    def test_how_question(self, router):
        assert router.classify("How does sqlite-vec store embeddings?") == "simple_retrieval"

    def test_factual_query(self, router):
        # Long enough that it can't be conversational; no complex markers.
        assert router.classify("Explain how the reranker works in detail.") == "simple_retrieval"


class TestComplexRetrieval:
    def test_compare_keyword(self, router):
        assert router.classify("Compare dense and sparse retrieval.") == "complex_retrieval"

    def test_vs_keyword(self, router):
        # Terse comparative question — the "vs" marker plus the trailing `?`
        # are both strong retrieval signals and must override the short-query
        # conversational fallback.
        assert router.classify("BM25 vs vector search?") == "complex_retrieval"

    def test_difference_keyword(self, router):
        assert router.classify("What is the difference between FTS5 and sqlite-vec?") == "complex_retrieval"

    def test_semicolon_implies_multi_topic(self, router):
        assert router.classify("Explain CRAG; also describe faithfulness.") == "complex_retrieval"

    def test_long_multi_clause_with_and(self, router):
        # Two substantial clauses joined by " and " → complex
        query = "Explain the chunking algorithm and describe how reranking works"
        assert router.classify(query) == "complex_retrieval"


class TestHelpers:
    def test_needs_retrieval_true_for_question(self, router):
        assert router.needs_retrieval("What is RAG?") is True

    def test_needs_retrieval_false_for_greeting(self, router):
        assert router.needs_retrieval("hi") is False

    def test_needs_decomposition_only_for_complex(self, router):
        assert router.needs_decomposition("Compare X and Y") is True
        assert router.needs_decomposition("What is X?") is False
        assert router.needs_decomposition("hello") is False


class TestRobustness:
    def test_handles_empty_string(self, router):
        # Should not crash on empty input.
        result = router.classify("")
        assert result in {"conversational", "simple_retrieval", "complex_retrieval"}

    def test_handles_only_whitespace(self, router):
        result = router.classify("   ")
        assert result in {"conversational", "simple_retrieval", "complex_retrieval"}

    def test_case_insensitive(self, router):
        a = router.classify("COMPARE X AND Y")
        b = router.classify("compare x and y")
        assert a == b
