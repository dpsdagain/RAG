"""Tests for the core pipeline: routing, CRAG, generation."""
from __future__ import annotations

import pytest
from app.core.query_router import QueryRouter


class TestQueryRouter:
    """Test heuristic query classification."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_conversational_greeting(self):
        assert self.router.classify("hello") == "conversational"
        assert self.router.classify("thanks") == "conversational"
        assert self.router.classify("hi there") == "conversational"

    def test_simple_retrieval(self):
        assert self.router.classify("What is the capital of France?") == "simple_retrieval"
        assert self.router.classify("How does photosynthesis work?") == "simple_retrieval"
        assert self.router.classify("Explain the concept of recursion") == "simple_retrieval"

    def test_complex_retrieval_comparison(self):
        result = self.router.classify("Compare Python vs JavaScript for web development")
        assert result == "complex_retrieval"

    def test_complex_retrieval_multi_clause(self):
        result = self.router.classify(
            "What are the advantages of React, and how does it compare to Vue for large applications"
        )
        assert result == "complex_retrieval"

    def test_needs_retrieval(self):
        assert not self.router.needs_retrieval("hello")
        assert self.router.needs_retrieval("What is AI?")

    def test_needs_decomposition(self):
        assert not self.router.needs_decomposition("What is AI?")
        assert self.router.needs_decomposition("Compare TensorFlow vs PyTorch for research and production")


class TestPromptTemplates:
    """Test prompt template construction."""

    def test_generation_messages(self):
        from app.core.prompt_templates.generation import build_messages
        messages = build_messages(
            query="What is X?",
            retrieved_chunks="[Source 1]: X is a thing.",
            working_memory="User: Hello\nAssistant: Hi",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is X?" in messages[1]["content"]
        assert "[Source 1]" in messages[1]["content"]

    def test_crag_messages(self):
        from app.core.prompt_templates.crag_evaluation import build_messages
        messages = build_messages("What is X?", "Chunk text")
        assert len(messages) == 1
        assert "SUFFICIENT" in messages[0]["content"]

    def test_decomposition_messages(self):
        from app.core.prompt_templates.decomposition import build_messages
        messages = build_messages("Complex query about A and B")
        assert len(messages) == 1
        assert "Complex query" in messages[0]["content"]
