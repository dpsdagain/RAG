# RAG System Test Suite

External test suite for the RAG system in `../rag-system/`. Kept outside
the project folder so it doesn't ship with the application code and so
you can swap implementations without touching tests.

## Layout

```
tests/
├── unit/           Pure-function tests — no DB, no embedder, no LLM
├── integration/    Real SQLite + sqlite-vec + FTS5 (uses a FakeEmbedder)
├── e2e/            Full RAGPipeline with mocked LLM
└── quality/        Real bge-small embedder + real LLM (slow, opt-in)
```

## Setup

The suite reuses the virtual environment in `../rag-system/.venv/`.
Install the test deps once:

```powershell
..\rag-system\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
```

## Running

From this directory:

```powershell
# Everything (≈2:15 wall-clock — quality tier dominates)
..\rag-system\.venv\Scripts\python.exe -m pytest -v

# Cheap tiers only (~10 s, no model load, no API calls)
..\rag-system\.venv\Scripts\python.exe -m pytest unit integration e2e -v

# Quality tier only (real model + real LLM, ~2 min, costs API tokens)
..\rag-system\.venv\Scripts\python.exe -m pytest quality -v

# Skip the LLM-dependent tests but keep real-embedder tests
..\rag-system\.venv\Scripts\python.exe -m pytest -v -m "not requires_llm"
```

## Markers

| Marker | What it requires |
|---|---|
| `requires_db` | sqlite-vec installed in the venv (already true) |
| `requires_embedder` | The real ONNX bge-small model on disk (currently unused — integration tests use a deterministic `FakeEmbedder`) |
| `requires_llm` | `RAG_LLM__API_KEY` env var and a real LLM endpoint |
| `slow` | Long-running test (currently unused) |

## What each tier covers

### Unit (`unit/`)
- `test_fts5_sanitizer.py` — the BM25 query escape (fix #4)
- `test_score_normalization.py` — L2² → [0, 1] formula (fix #6)
- `test_rrf_fusion.py` — Reciprocal Rank Fusion math
- `test_query_router.py` — heuristic conversational / simple / complex classifier
- `test_semantic_cache.py` — lookup, LRU, TTL, threshold, cosine helper

### Integration (`integration/`)
- `test_database_retrieval.py` — `vector_search` and `bm25_search` end-to-end against a temp SQLite. Explicitly exercises the production-regression queries (`?` in BM25, joined-WHERE in vec0 K-NN — fixes #4 and #5).
- `test_hybrid_search.py` — `HybridSearchEngine` full path: dense + sparse + RRF + multi-query.

### E2E (`e2e/`)
- `test_pipeline_smoke.py` — `RAGPipeline.execute()` and `execute_stream()` with a `FakeLLMClient`. Covers cache hit, abstention, conversational, and the streaming event sequence (`token` → `sources` → `done`).

### Quality (`quality/`) — real model + real LLM
The first three tiers verify *plumbing* with deterministic fakes. The quality tier verifies *semantics* — that retrieval actually finds the right chunks and the LLM produces grounded answers.

- `golden_set.py` — 5 `(query, must_retrieve, must_generate)` triples covering chunking, database, hybrid search, embedder, CRAG.
- `test_retrieval_quality.py` — parametrized over the golden set. Asserts that real-bge-small + hybrid search puts the right chunk in the top-5. Plus dense-only / BM25-only spot checks and an off-topic-doesn't-dominate guard.
- `test_pipeline_quality.py` — parametrized over the golden set. Runs the full `RAGPipeline` with the real LLM (CRAG / decomposition / faithfulness disabled to keep cost down to one LLM call per test), asserts the answer contains at least one expected fact and carries source citations.
- `test_latency_budget.py` — embed-one-query under 250 ms, full hybrid search under 1 s, batch embedding faster than serial.

Skipped automatically when bge-small isn't on disk (`requires_embedder`) or `RAG_LLM__API_KEY` isn't set (`requires_llm`).

## Notes

- All integration tests use `pytest-asyncio` in auto mode and a temp
  SQLite DB per test (no shared state).
- The `FakeEmbedder` in `conftest.py` produces deterministic 384-d
  L2-normalized vectors from a SHA-256 of the input text. Same text →
  same vector. Avoids downloading the real bge-small model just to run
  the suite.
