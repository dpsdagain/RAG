"""Shared pytest fixtures and import path setup.

The RAG codebase lives in ../rag-system/. We prepend that to sys.path so
tests can `from app.X import Y` without the project being pip-installed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

# Windows PowerShell defaults to cp1252 — pytest output containing
# Unicode (✓/✗, →, smart quotes from LLM responses) crashes with
# UnicodeEncodeError before the test result even prints. Force UTF-8
# stdio so the user can actually see the test run.
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Locate rag-system/ alongside this tests/ folder and put it on the path.
TESTS_ROOT = Path(__file__).resolve().parent
RAG_SYSTEM_ROOT = TESTS_ROOT.parent / "rag-system"
if str(RAG_SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_SYSTEM_ROOT))


# ---------------------------------------------------------------------------
# Lightweight stand-in for the heavy ONNX embedder.
# ---------------------------------------------------------------------------

class FakeEmbedder:
    """Deterministic, dependency-free embedder for fast tests.

    Hashes each input into a 384-d L2-normalized vector. Same text always
    produces the same vector; different texts produce different vectors.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str, input_type: str = "document") -> list[float]:
        import hashlib
        import math

        # Seed-based PRNG keyed by the text's hash → deterministic vector.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:8], "big")
        state = seed or 1
        vec: list[float] = []
        for _ in range(self._dim):
            # Linear congruential generator → values in roughly [-1, 1].
            state = (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            vec.append(((state >> 32) / (1 << 32)) * 2.0 - 1.0)

        # L2-normalize (matches bge-small's output)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_batch(
        self, texts: list[str], batch_size: int = 64, input_type: str = "document"
    ) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def count_tokens(self, text: str) -> int:
        # Rough heuristic — good enough for budgeting in tests.
        return max(1, len(text.split()))


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    """Provides a deterministic FakeEmbedder for tests that need an Embedder."""
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Temporary database fixture (real SQLite + sqlite-vec + FTS5)
# ---------------------------------------------------------------------------

@pytest.fixture
async def temp_db(tmp_path: Path) -> Iterator:
    """Create a fresh Database in a tmp dir and tear it down after the test."""
    from app.infrastructure.database import Database

    db_path = tmp_path / "test_rag.db"
    db = Database(db_path)
    await db.initialize()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def sample_chunks():
    """A small set of ChunkRecord-shaped dicts for seeding the DB."""
    return [
        {
            "chunk_id": "c1",
            "doc_id": "d1",
            "content": "Sentence-level semantic chunking splits text into coherent segments.",
            "section_title": "Chunking Strategy",
        },
        {
            "chunk_id": "c2",
            "doc_id": "d1",
            "content": "The system uses sqlite-vec for dense vector storage and FTS5 for BM25.",
            "section_title": "Database",
        },
        {
            "chunk_id": "c3",
            "doc_id": "d1",
            "content": "Reciprocal Rank Fusion merges dense and sparse results into one ranking.",
            "section_title": "Retrieval",
        },
        {
            "chunk_id": "c4",
            "doc_id": "d2",
            "content": "FlashRank runs a CPU cross-encoder to rerank candidates.",
            "section_title": "Reranker",
        },
        {
            "chunk_id": "c5",
            "doc_id": "d2",
            "content": "Cats and dogs make excellent companions for many households.",
            "section_title": "Pets",
        },
    ]


# ---------------------------------------------------------------------------
# Quality-tier fixtures: REAL embedder + REAL LLM + ingested corpus.
# These are session-scoped to amortize the cost of model loading and
# ingestion. Tests opt in via the requires_embedder / requires_llm markers.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def real_embedder():
    """Load the actual bge-small-en-v1.5 ONNX embedder."""
    try:
        from app.models.embedder import Embedder
        return Embedder(
            model_path="BAAI/bge-small-en-v1.5",
            dim=384,
            num_threads=4,
        )
    except (FileNotFoundError, ImportError) as e:
        pytest.skip(f"real bge-small not available: {e}")


@pytest.fixture(scope="session")
def real_llm_client():
    """Build an LLMClient from .env + config.yaml. Skips when no API key."""
    import os
    from pathlib import Path

    # Best-effort .env load — the production config.py does the same on startup.
    env_path = TESTS_ROOT.parent / "rag-system" / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except ImportError:
            pass

    from app.infrastructure.config import get_settings
    settings = get_settings()
    api_key = os.getenv("RAG_LLM__API_KEY", "") or settings.llm.api_key

    if not api_key:
        pytest.skip("RAG_LLM__API_KEY not set; quality LLM tests are opt-in")

    from app.models.llm_client import LLMClient
    return LLMClient(
        provider=settings.llm.provider,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        api_key=api_key,
        timeout=settings.llm.timeout_seconds,
    )


@pytest.fixture(scope="session")
async def ingested_corpus_db(tmp_path_factory, real_embedder):
    """Ingest the V3 architecture markdown into a session-scoped temp DB.

    Yields a Database with the doc already chunked, embedded, and stored —
    exactly the state a real production DB would be in after `/v1/ingest/file`.
    Used by quality tests to assert that retrieval finds the right sections.
    """
    import hashlib

    from app.infrastructure.database import Database, DocumentRecord
    from app.ingestion.chunking.semantic_chunker import SemanticChunker
    from app.ingestion.router import IngestionRouter

    doc_path = TESTS_ROOT.parent / "rag_v3_implementation_architecture.md"
    if not doc_path.exists():
        pytest.skip(f"test corpus missing: {doc_path}")

    db_path = tmp_path_factory.mktemp("quality_db") / "rag.db"
    db = Database(db_path)
    await db.initialize()

    router = IngestionRouter()
    chunker = SemanticChunker(real_embedder)
    parsed = await router.parse_file(doc_path)

    doc_id = "v3_arch_doc"
    file_hash = hashlib.sha256(doc_path.read_bytes()).hexdigest()
    await db.insert_document(DocumentRecord(
        doc_id=doc_id,
        source_uri=str(doc_path),
        source_type="markdown",
        file_hash=file_hash,
        title=parsed.title,
        parser_used=parsed.metadata.get("parser", "raw"),
        status="active",
    ))

    chunks = chunker.chunk(parsed, doc_id)
    await db.insert_chunks_batch(chunks)

    try:
        yield db
    finally:
        db.close()
