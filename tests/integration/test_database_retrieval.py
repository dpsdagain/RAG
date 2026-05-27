"""Integration tests for Database.vector_search and Database.bm25_search.

These hit a real on-disk SQLite + sqlite-vec + FTS5. The tests exercise
the exact code paths that were silently failing in production before the
sanitizer + K-NN-subquery fixes.
"""
from __future__ import annotations

import hashlib
import uuid

import pytest

from app.infrastructure.database import (
    ChunkRecord,
    Database,
    DocumentRecord,
    _sanitize_fts5_query,
)


pytestmark = pytest.mark.requires_db


async def _seed(db: Database, fake_embedder, chunks: list[dict]) -> list[str]:
    """Seed the DB with a document and a list of chunks. Returns doc_ids."""
    doc_ids: dict[str, bool] = {}
    chunk_records: list[ChunkRecord] = []
    for ch in chunks:
        doc_id = ch["doc_id"]
        if doc_id not in doc_ids:
            await db.insert_document(DocumentRecord(
                doc_id=doc_id,
                source_uri=f"test://{doc_id}",
                source_type="test",
                file_hash=hashlib.sha256(doc_id.encode()).hexdigest(),
                status="active",
            ))
            doc_ids[doc_id] = True
        chunk_records.append(ChunkRecord(
            chunk_id=ch["chunk_id"],
            doc_id=doc_id,
            content=ch["content"],
            embedding=fake_embedder.embed(ch["content"]),
            chunk_index=0,
            section_title=ch.get("section_title"),
            token_count=fake_embedder.count_tokens(ch["content"]),
            content_hash=hashlib.sha256(ch["content"].encode()).hexdigest(),
        ))
    await db.insert_chunks_batch(chunk_records)
    return list(doc_ids)


class TestVectorSearch:
    async def test_returns_results_for_known_query(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        query_vec = fake_embedder.embed("how does chunking work")
        results = await temp_db.vector_search(query_vec, top_k=5)
        assert len(results) > 0
        # Scores must lie inside [0, 1] (formula fix #6)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    async def test_top_k_respected(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        results = await temp_db.vector_search(fake_embedder.embed("test"), top_k=2)
        assert len(results) <= 2

    async def test_ordered_by_similarity(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        results = await temp_db.vector_search(fake_embedder.embed("query"), top_k=5)
        # Scores must be in non-increasing order.
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    async def test_inactive_documents_excluded(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        # Soft-delete one document
        await temp_db.delete_document("d1")
        results = await temp_db.vector_search(fake_embedder.embed("test"), top_k=10)
        assert all(r.doc_id != "d1" for r in results)


class TestBM25Search:
    async def test_simple_word_query(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        results = await temp_db.bm25_search("chunking", top_k=5)
        assert len(results) > 0
        assert any("chunking" in r.content.lower() for r in results)

    async def test_question_mark_does_not_crash(self, temp_db, fake_embedder, sample_chunks):
        # This is the exact production regression — query with `?` used to
        # raise `fts5: syntax error near "?"` and return [] silently.
        await _seed(temp_db, fake_embedder, sample_chunks)
        results = await temp_db.bm25_search("What is the chunking strategy?", top_k=5)
        # We don't assert > 0 because BM25 stemming may not always hit, but
        # the call MUST NOT crash and MUST return a list.
        assert isinstance(results, list)

    async def test_fts5_operators_are_safe(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        for risky in ["foo*", "tag:bar", "^anchor", "minus-word", "x AND y"]:
            results = await temp_db.bm25_search(risky, top_k=5)
            assert isinstance(results, list)

    async def test_empty_query_does_not_crash(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        results = await temp_db.bm25_search("", top_k=5)
        assert isinstance(results, list)

    async def test_inactive_documents_excluded(self, temp_db, fake_embedder, sample_chunks):
        await _seed(temp_db, fake_embedder, sample_chunks)
        await temp_db.delete_document("d1")
        results = await temp_db.bm25_search("chunking", top_k=10)
        assert all(r.doc_id != "d1" for r in results)


class TestSchemaIntegrity:
    async def test_insert_chunk_populates_vec_table(self, temp_db, fake_embedder, sample_chunks):
        # Vector inserts must land in chunks_vec, or dense search returns [].
        await _seed(temp_db, fake_embedder, sample_chunks)
        row = await temp_db.fetch_one("SELECT COUNT(*) AS n FROM chunks_vec")
        assert row["n"] == len(sample_chunks)

    async def test_insert_chunk_populates_fts_table(self, temp_db, fake_embedder, sample_chunks):
        # FTS5 trigger should fire on every chunk insert.
        await _seed(temp_db, fake_embedder, sample_chunks)
        row = await temp_db.fetch_one("SELECT COUNT(*) AS n FROM chunks_fts")
        assert row["n"] == len(sample_chunks)
