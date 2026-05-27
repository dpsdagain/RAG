"""Tests for the FTS5 query sanitizer that fixes Bug A from the diagnostic.

Without sanitization, FTS5 treats characters like `?`, `*`, `:`, `^` as
query operators and crashes with `fts5: syntax error near "?"`.
"""
from __future__ import annotations

import pytest

from app.infrastructure.database import _sanitize_fts5_query


class TestFTS5Sanitizer:
    def test_empty_string_returns_empty_phrase(self):
        assert _sanitize_fts5_query("") == '""'

    def test_only_punctuation_returns_empty_phrase(self):
        assert _sanitize_fts5_query("???!!!") == '""'
        assert _sanitize_fts5_query("   ") == '""'

    def test_plain_words_get_quoted(self):
        assert _sanitize_fts5_query("hello world") == '"hello" "world"'

    def test_question_mark_does_not_crash(self):
        # The original symptom from the production logs.
        result = _sanitize_fts5_query("What is the chunking strategy?")
        assert result == '"What" "is" "the" "chunking" "strategy"'

    def test_fts5_operators_are_stripped(self):
        # `*`, `:`, `^`, `-` are FTS5 operators; they should not leak through.
        for raw in ["foo*", "tag:foo", "^anchor", "minus-word"]:
            cleaned = _sanitize_fts5_query(raw)
            for op in ["*", ":", "^"]:
                assert op not in cleaned, f"{op!r} leaked through for {raw!r}"

    def test_unicode_word_chars_preserved(self):
        # \w in Python's re matches unicode letters by default.
        out = _sanitize_fts5_query("café naïve")
        assert "café" in out
        assert "naïve" in out

    def test_quotes_in_input_dont_break_output(self):
        # Even if the user types a stray quote, the output must still parse.
        out = _sanitize_fts5_query('say "hi"')
        # Tokens are quoted, no unmatched bare quotes.
        # Should produce something like '"say" "hi"' (the literal quote chars
        # are not word characters so \w doesn't capture them).
        assert out.count('"') % 2 == 0

    def test_mixed_case_preserved(self):
        out = _sanitize_fts5_query("ChunkingStrategy")
        assert "ChunkingStrategy" in out

    def test_numbers_preserved(self):
        # \w matches digits, so "version2" should survive.
        out = _sanitize_fts5_query("version 2 of bge")
        assert '"version"' in out
        assert '"2"' in out
        assert '"bge"' in out
