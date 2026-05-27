"""Integration tests for the HybridSearchEngine end-to-end."""
from __future__ import annotations

import hashlib

import pytest

from app.infrastructure.database import ChunkRecord, DocumentRecord
from app.retrieval.hybrid_search import HybridSearchEngine


pytestmark = pytest.mark.requires_db


async def _seed_minimal(db, embedder, n: int = 6):
    await db.insert_document(DocumentRecord(
        doc_id="doc1",
        source_uri="test://doc1",
        source_type="test",
        file_hash=hashlib.sha256(b"doc1").hexdigest(),
        status="active",
    ))
    contents = [
        "Sentence-level semantic chunking splits documents into coherent pieces.",
        "Hybrid retrieval combines dense vectors with BM25 sparse search.",
        "Reciprocal Rank Fusion fuses two ranked lists by reciprocal rank.",
        "FlashRank reranks candidates using a CPU cross-encoder model.",
        "The system stores embeddings in sqlite-vec virtual tables.",
        "Cats make wonderful pets according to many veterinarians.",
    ][:n]
    chunks = [
        ChunkRecord(
            chunk_id=f"c{i}",
            doc_id="doc1",
            content=content,
            embedding=embedder.embed(content),
            chunk_index=i,
            section_title=None,
            token_count=embedder.count_tokens(content),
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )
        for i, content in enumerate(contents)
    ]
    await db.insert_chunks_batch(chunks)


class TestHybridSearch:
    async def test_returns_results(self, temp_db, fake_embedder):
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search("chunking strategy?", top_n=10)
        assert isinstance(results, list)
        assert len(results) > 0

    async def test_top_n_respected(self, temp_db, fake_embedder):
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search("retrieval", top_n=3)
        assert len(results) <= 3

    async def test_no_duplicate_chunk_ids(self, temp_db, fake_embedder):
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search("retrieval reranking", top_n=10)
        ids = [r.chunk_id for r in results]
        assert len(ids) == len(set(ids))

    async def test_scores_descending(self, temp_db, fake_embedder):
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search("chunking", top_n=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_handles_empty_corpus(self, temp_db, fake_embedder):
        # No documents at all — must not crash.
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search("anything", top_n=10)
        assert results == []

    async def test_multi_query_search(self, temp_db, fake_embedder):
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        results = await engine.search_multi_query(
            ["chunking", "reranking"],
            top_n=5,
        )
        assert isinstance(results, list)
        assert len(results) <= 5

    async def test_punctuation_query_does_not_crash(self, temp_db, fake_embedder):
        # Production regression — `?` used to bring down BM25.
        await _seed_minimal(temp_db, fake_embedder)
        engine = HybridSearchEngine(temp_db, fake_embedder)
        for q in ["What?", "Why?!", "tag:foo", "**emphasis**", "a + b"]:
            results = await engine.search(q, top_n=3)
            assert isinstance(results, list)
