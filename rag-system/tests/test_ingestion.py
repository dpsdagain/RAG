"""Tests for the ingestion pipeline: parsers, chunking, dedup."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from app.ingestion.router import IngestionRouter, EXTENSION_MAP
from app.ingestion.parsers.base import ParsedDocument, PageContent


class TestIngestionRouter:
    """Test file type detection and routing."""

    def setup_method(self):
        self.router = IngestionRouter()

    def test_detect_pdf(self):
        assert self.router.detect_file_type(Path("doc.pdf")) == "pdf"

    def test_detect_python(self):
        assert self.router.detect_file_type(Path("script.py")) == "code"

    def test_detect_javascript(self):
        assert self.router.detect_file_type(Path("app.js")) == "code"

    def test_detect_markdown(self):
        assert self.router.detect_file_type(Path("README.md")) == "markdown"

    def test_detect_image(self):
        assert self.router.detect_file_type(Path("scan.png")) == "image"

    def test_detect_unknown(self):
        assert self.router.detect_file_type(Path("data.xyz")) == "unknown"

    def test_all_extensions_covered(self):
        """Ensure all mapped extensions resolve to a valid type."""
        valid_types = {"pdf", "code", "markdown", "office", "image", "web"}
        for ext, ftype in EXTENSION_MAP.items():
            assert ftype in valid_types, f"Extension {ext} maps to invalid type: {ftype}"


class TestSemanticChunker:
    """Test semantic chunking logic."""

    def test_empty_document(self):
        from app.ingestion.chunking.semantic_chunker import SemanticChunker
        embedder = MagicMock()
        chunker = SemanticChunker(embedder)

        empty_doc = ParsedDocument(content="", source_type="test")
        result = chunker.chunk(empty_doc, "doc-1")
        assert result == []

    def test_special_block_extraction(self):
        from app.ingestion.chunking.semantic_chunker import SemanticChunker
        embedder = MagicMock()
        chunker = SemanticChunker(embedder)

        text = "Hello world.\n\n```python\ndef foo():\n    pass\n```\n\nMore text."
        blocks, clean = chunker._extract_special_blocks(text)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "code"
        assert "```python" in blocks[0]["text"]


class TestAdaptiveMetrics:
    """Test chunk quality metrics."""

    def test_icc_single_embedding(self):
        from app.ingestion.chunking.adaptive_metrics import compute_icc
        assert compute_icc([[1.0, 0.0, 0.0]]) == 1.0

    def test_icc_identical_embeddings(self):
        from app.ingestion.chunking.adaptive_metrics import compute_icc
        embs = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
        icc = compute_icc(embs)
        assert abs(icc - 1.0) < 0.01

    def test_dcc_identical(self):
        from app.ingestion.chunking.adaptive_metrics import compute_dcc
        vec = [1.0, 0.0, 0.0]
        assert abs(compute_dcc(vec, vec) - 1.0) < 0.01

    def test_quality_evaluation(self):
        from app.ingestion.chunking.adaptive_metrics import evaluate_chunk_quality
        assert evaluate_chunk_quality(0.1, 0.5) == "poor"
        assert evaluate_chunk_quality(0.3, 0.2) == "fair"
        assert evaluate_chunk_quality(0.5, 0.5) == "good"
