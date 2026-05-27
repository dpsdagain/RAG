"""Heuristic query router: classifies queries without LLM calls."""
from __future__ import annotations

import re


QUESTION_WORDS = {"what", "how", "why", "when", "where", "which", "who", "can", "does", "is", "are", "do", "explain", "describe"}
COMPLEX_MARKERS = {"compare", "vs", "versus", "difference", "differences", "contrast", "pros and cons", "advantages"}


class QueryRouter:
    """Simple heuristic query classifier — no LLM call needed.

    Classifies queries into:
    - conversational: greetings, short responses, no retrieval needed
    - simple_retrieval: standard factual queries
    - complex_retrieval: multi-hop, comparative, or multi-clause queries
    """

    def classify(self, query: str) -> str:
        """Classify a query into a routing category.

        Strong retrieval signals (comparison markers, `?`, question words)
        override the short-query conversational fallback. This way terse
        but clearly-comparative inputs like "BM25 vs vector search?" still
        route to retrieval instead of being mistaken for a greeting.

        Args:
            query: The user's query text.

        Returns:
            One of: 'conversational', 'simple_retrieval', 'complex_retrieval'.
        """
        query_lower = query.lower().strip()
        words = query_lower.split()
        num_words = len(words)

        has_question_word = any(w in QUESTION_WORDS for w in words)
        has_question_mark = "?" in query

        # 1. Complex retrieval — comparative or multi-topic signals
        #    override everything else, even on short queries.
        if any(marker in query_lower for marker in COMPLEX_MARKERS):
            return "complex_retrieval"

        if ";" in query:
            return "complex_retrieval"

        if "," in query and num_words > 8:
            clauses = [c.strip() for c in query.split(",") if len(c.strip()) > 10]
            if len(clauses) >= 2:
                return "complex_retrieval"

        # Two substantial clauses joined by " and "
        if " and " in query_lower and num_words >= 8:
            parts = query_lower.split(" and ")
            if len(parts) >= 2 and all(len(p.split()) > 3 for p in parts):
                return "complex_retrieval"

        # 2. Conversational fallback — only for truly signal-free short input.
        if num_words < 5 and not has_question_word and not has_question_mark:
            return "conversational"

        # 3. Default — anything with a question signal or substantive length.
        return "simple_retrieval"

    def needs_retrieval(self, query: str) -> bool:
        """Check if a query requires document retrieval."""
        return self.classify(query) != "conversational"

    def needs_decomposition(self, query: str) -> bool:
        """Check if a query should be decomposed into sub-queries."""
        return self.classify(query) == "complex_retrieval"
