"""End-to-end smoke tests for the full RAGPipeline.

The LLM is mocked with a FakeLLMClient that returns canned text and a
canned async-iterator for streaming. The rest of the pipeline (DB,
embedder, reranker, cache) is exercised for real.
"""
from __future__ import annotations

import hashlib
from typing import AsyncIterator

import pytest

from app.core.pipeline import RAGPipeline
from app.infrastructure.config import get_settings
from app.infrastructure.database import ChunkRecord, DocumentRecord


pytestmark = pytest.mark.requires_db


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Records the messages sent to it and returns canned responses.

    Implements the subset of LLMClient that the pipeline actually calls:
    `generate` and `generate_stream`.
    """

    def __init__(
        self,
        canned_response: str = "Mock LLM answer with citation [Source 1].",
        canned_crag: str = "SUFFICIENT",
        canned_faith: str = "ALL CLAIMS VERIFIED",
    ) -> None:
        self._canned_response = canned_response
        self._canned_crag = canned_crag
        self._canned_faith = canned_faith
        self.calls: list[list[dict]] = []
        self.stream_calls: list[list[dict]] = []

    async def generate(self, messages, temperature=0.1, max_tokens=2048):
        self.calls.append(messages)
        # The pipeline calls generate() for CRAG, decomposition, faithfulness,
        # query-rewrite, episodic summarisation, and conversational replies.
        # Dispatch on max_tokens — small budgets are sub-evaluations.
        if max_tokens <= 20:
            return self._canned_crag
        joined = " ".join(m.get("content", "") for m in messages).lower()
        if "rate the retrieval" in joined:
            return self._canned_crag
        if "supported by the source" in joined or "all claims verified" in joined:
            return self._canned_faith
        if "break the following" in joined or "sub-question" in joined:
            return '["sub-query one", "sub-query two"]'
        return self._canned_response

    async def generate_stream(self, messages, temperature=0.1, max_tokens=2048) -> AsyncIterator[str]:
        self.stream_calls.append(messages)
        # Emit the canned response in word-sized chunks so the test sees
        # multiple token events.
        for word in self._canned_response.split():
            yield word + " "


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _seed_corpus(db, embedder):
    """Populate the DB with a small corpus the pipeline can retrieve against."""
    await db.insert_document(DocumentRecord(
        doc_id="d1",
        source_uri="test://corpus.md",
        source_type="markdown",
        file_hash=hashlib.sha256(b"d1").hexdigest(),
        status="active",
    ))
    contents = [
        "Sentence-level semantic chunking splits documents into coherent pieces using cosine similarity.",
        "The chunking algorithm uses a default threshold of 0.3 and a max chunk size of 512 tokens.",
        "Hybrid retrieval combines dense vector search with BM25 keyword search via Reciprocal Rank Fusion.",
        "FlashRank reranks the merged candidates using a CPU cross-encoder model.",
    ]
    records = [
        ChunkRecord(
            chunk_id=f"c{i}",
            doc_id="d1",
            content=text,
            embedding=embedder.embed(text),
            chunk_index=i,
            section_title="Chunking" if "chunking" in text.lower() else "Retrieval",
            token_count=embedder.count_tokens(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )
        for i, text in enumerate(contents)
    ]
    await db.insert_chunks_batch(records)


async def _make_pipeline(db, embedder, llm):
    """Construct a RAGPipeline with default settings."""
    settings = get_settings()
    # Disable optional features that would call the LLM unnecessarily, keep
    # the test focused on the retrieval + generation contract.
    settings.pipeline.decomposition_enabled = False
    settings.pipeline.crag_enabled = False
    settings.pipeline.faithfulness_check_enabled = False
    return RAGPipeline(db=db, embedder=embedder, llm=llm, settings=settings)


# ---------------------------------------------------------------------------
# Non-streaming pipeline
# ---------------------------------------------------------------------------

class TestPipelineExecute:
    async def test_returns_chat_response_for_retrieval_query(self, temp_db, fake_embedder):
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient(canned_response="Chunking is semantic [Source 1].")
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        result = await pipeline.execute("What is the chunking strategy?")
        assert result.response.startswith("Chunking")
        assert isinstance(result.sources, list)
        # All citation scores fall in the validation range
        for src in result.sources:
            assert 0.0 <= src.score <= 1.0

    async def test_citation_snippet_under_300_chars(self, temp_db, fake_embedder):
        # Regression: SourceCitation.content_snippet must be ≤ 300 chars.
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient()
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        result = await pipeline.execute("retrieval question")
        for src in result.sources:
            assert len(src.content_snippet) <= 300

    async def test_conversational_skips_retrieval(self, temp_db, fake_embedder):
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient(canned_response="Hi there!")
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        result = await pipeline.execute("hello")
        assert result.query_type == "conversational"
        # Conversational replies carry no source citations.
        assert result.sources == []

    async def test_abstains_when_corpus_empty(self, temp_db, fake_embedder):
        # No documents seeded — retrieval returns []; pipeline must abstain.
        llm = FakeLLMClient()
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        result = await pipeline.execute("What is the chunking strategy?")
        assert result.crag_verdict == "INSUFFICIENT"
        assert "don't have enough" in result.response.lower()

    async def test_cache_hit_on_repeat_query(self, temp_db, fake_embedder):
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient(canned_response="First answer.")
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        first = await pipeline.execute("What is the chunking strategy?")
        # Change the canned response — if the cache hits, we still see the first.
        llm._canned_response = "DIFFERENT"
        second = await pipeline.execute("What is the chunking strategy?")

        assert second.crag_verdict == "CACHED"
        assert second.response == first.response


# ---------------------------------------------------------------------------
# Streaming pipeline (execute_stream)
# ---------------------------------------------------------------------------

class TestPipelineStream:
    async def test_emits_sources_then_tokens_then_done(self, temp_db, fake_embedder):
        """Sources must arrive BEFORE tokens so UI can render inline citations."""
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient(canned_response="alpha beta gamma")
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        events: list[dict] = []
        async for evt in pipeline.execute_stream("What is the chunking strategy?"):
            events.append(evt)

        event_names = [e["event"] for e in events]
        assert "sources" in event_names
        assert "token" in event_names
        assert "done" in event_names
        # Sources first, then tokens, then done.
        assert event_names.index("sources") < event_names.index("token")
        assert event_names.index("token") < event_names.index("done")

    async def test_token_events_concatenate_to_response(self, temp_db, fake_embedder):
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient(canned_response="one two three four")
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        tokens: list[str] = []
        async for evt in pipeline.execute_stream("retrieval question"):
            if evt["event"] == "token":
                tokens.append(evt["data"])
        assert "".join(tokens).strip() == "one two three four"

    async def test_abstention_streams_as_tokens(self, temp_db, fake_embedder):
        # Empty corpus → abstention path must still yield tokens + done.
        llm = FakeLLMClient()
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        names: list[str] = []
        async for evt in pipeline.execute_stream("question?"):
            names.append(evt["event"])
        assert "token" in names
        assert names[-1] == "done"

    async def test_done_event_carries_metadata(self, temp_db, fake_embedder):
        await _seed_corpus(temp_db, fake_embedder)
        llm = FakeLLMClient()
        pipeline = await _make_pipeline(temp_db, fake_embedder, llm)

        done_event: dict | None = None
        async for evt in pipeline.execute_stream("chunking question"):
            if evt["event"] == "done":
                done_event = evt
        assert done_event is not None
        for key in ("conversation_id", "crag_verdict", "latency_ms"):
            assert key in done_event["data"]
