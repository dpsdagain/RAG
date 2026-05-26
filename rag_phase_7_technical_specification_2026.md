# Phase 7 — Technical Specification: V2 Elite Thin-Client Architecture

This document serves as the implementation-ready, production-grade engineering specification for the 2026 "Thin-Client" RAG System. It is strictly designed for a hardware constraint of **16GB RAM / 0 VRAM**, prioritizing local logic execution and CPU-based vector operations, while offloading heavy LLM generation to the cloud.

---

## A. PROJECT STRUCTURE

The project follows a modular, domain-driven design running entirely within a **single native Python process**. There are no heavy container stacks or Redis brokers.

```text
rag-system/
├── app/                        # Root application module
│   ├── api/                    # API Gateway (FastAPI)
│   │   ├── routes/             # Endpoints (ingest, retrieve, admin)
│   │   └── schemas.py          # Pydantic validation models
│   ├── core/                   # Core business logic & orchestration
│   │   ├── orchestration/      # LangGraph state machines & agent graphs
│   │   ├── agents/             # Individual agent definitions (Planner, Critic)
│   │   └── memory/             # Local preference engine
│   ├── ingestion/              # Data ingestion & parsing pipeline
│   │   ├── workers/            # In-process threading/asyncio workers
│   │   ├── parsers/            # PyMuPDF local, LlamaParse API wrappers
│   │   └── chunking/           # Semantic Markdown splitting
│   ├── retrieval/              # The Retrieval Engine
│   │   ├── engines/            # Vector (sqlite-vec), BM25
│   │   ├── rerankers/          # FlashRank (Local CPU)
│   │   └── routers/            # Confidence-based routing heuristics
│   ├── models/                 # Model wrappers
│   │   ├── llm/                # Ollama client
│   │   └── embeddings/         # MiniLM-L6 (ONNX, CPU)
│   └── infrastructure/         # DB connection management
│       ├── database.py         # SQLite & sqlite-vec setup (WAL mode)
│       └── observability.py    # Local metrics (RSS, CPU latency)
├── deployment/                 # Deployment scripts
├── tests/                      # Pytest suite
├── configs/                    # YAML configuration files
└── scripts/                    # CLI tools for DB migrations, evaluation
```

---

## B. DATABASE SCHEMAS

The entire storage layer relies on SQLite extended with `sqlite-vec`. 

**CRITICAL PRAGMAS:** Upon initialization, the database must execute `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;` to prevent `database is locked` concurrency errors between the active API thread and background ingestion tasks.

### 1. Vector DB (sqlite-vec) & Relational Store
**Table: `documents`** (Metadata Store)
*   `doc_id` (UUID, Primary Key)
*   `source_uri` (String, Indexed)
*   `file_hash` (String, deduplication)
*   `modality` (Enum: PDF, CODE, WEB, etc.)

**Table: `chunks`** (Vector + Payload)
*   `chunk_id` (UUID, Primary Key)
*   `doc_id` (UUID, Foreign Key -> documents)
*   `content` (Text, the raw chunk)
*   `embedding` (F32_BLOB, sqlite-vec dimension=384 for confirmed embedder: bge-small-en-v1.5)
*   `bm25_tokens` (Text, tokenized representation for FTS5 keyword search)
*   `chunk_index` (Integer, sequential order in document)
*   `parent_chunk_id` (UUID, Optional, Foreign Key -> chunks)

### 2. State & Memory Store
**Table: `working_memory`** (LangGraph State)
*   `thread_id` (UUID)
*   `state_blob` (JSON)
*   `written_at` (Timestamp)

**Table: `user_preferences`** (Versioned preference store)
*   `pref_id` (UUID)
*   `preference` (String)
*   `last_accessed` (Timestamp, for optional decay)

---

## C. INGESTION WORKERS

Ingestion runs via a **single in-process background worker** (using Python `threading` or `asyncio`). Idempotency is enforced by a file-hash check against the `documents` table. Failed parses are logged to a local `failed_ingests` SQLite table.

**CPU Contention Rule:** Indexing is strictly throttled to N CPU threads and paused/niced while a query is in flight, so embedding a large PDF doesn't starve active query latency.

### Modality Specifications
*   **PDF Ingestion:**
    *   *Parser:* Attempt **pymupdf4llm** local text extraction first. Route to the **LlamaParse API** *only* when local text-extraction coverage is below threshold (e.g., heavily scanned/complex).
    *   *Chunking:* Semantic Markdown Splitter.
    *   *Embedding:* bge-small-en-v1.5, CPU, ONNX-quantized (30–400 chunks/s batched).
*   **Codebase Ingestion:**
    *   *Parser:* Local Tree-sitter AST extraction.

---

## D. RETRIEVAL ENGINE SPECIFICATION

The pipeline is optimized for maximum semantic precision running on local CPU resources.

### Pipeline Flow
1.  **Routing (Confidence-Based):** 
    *   If query is simple conversational -> Skip retrieval.
2.  **Hybrid Retrieval (Local Stage 1):**
    *   Dense Search (`sqlite-vec`) -> Top 50.
    *   Sparse Search (SQLite FTS5 BM25) -> Top 50.
    *   Merge via Reciprocal Rank Fusion (RRF) -> Top 100.
3.  **Local Reranking (Local Stage 2):**
    *   Pass Top 100 to `FlashRank` (CPU bound, <200MB RAM).
    *   Prune to Top 15.
4.  **Parent-Context Injection (Small-to-Big):**
    *   For the Top 15 chunks, retrieve their associated `parent_chunk_id` content to inject broader surrounding context before final truncation.
5.  **Final Reranking (Local/Cloud Stage 3):**
    *   If Cohere is disabled: Take the Top 5 of the FlashRank 15.
    *   If Cohere is enabled: Pass Top 15 + Query to Cohere API (Cross-Encoder) for maximum precision, pruning to Top 5.

### Thresholds
*   **Relevance Threshold:** If maximum relevance < 0.8, trigger an "Anti-Hallucination" halt. Ask the user for clarification.

---

## E. MEMORY ENGINE SPECIFICATION

*   **Working Memory:** Local `working_memory` SQLite checkpointer. Stores the last N turns.
*   **Versioned Preference Store (Long-Term):** User-editable SQLite key-value pairs representing persistent rules (e.g., "Always output Python code"). Optional chronological decay.

---

## F. AGENT ORCHESTRATION

Powered by **LangGraph** using an asynchronous cyclical graph, but explicitly optimized to avoid infinite loops and gracefully handle API failures.

### Orchestration Logic & Safeguards
*   **Max Recursion:** Graph loops are hard-capped at 2 iterations.
*   **Fallback Handling:** The generation LLM is the sole cloud dependency. On a generation timeout: trigger one retry with exponential backoff. If it still fails, LangGraph **gracefully degrades by returning the reranked Top-5 chunks as raw evidence** with a "generation unavailable" notice. 
*   *Note: Running a 1.5B model locally on CPU yields ~5-15 tok/s and severe hallucination risks, thus honest degradation is preferred over a local micro-model.*

---

## G. API DESIGN

FastAPI-based async REST architecture.
*   `POST /v1/ingest/file` (Triggers in-process background worker)
*   `POST /v1/chat/completions` (OpenAI Compatible, SSE streaming)

---

## H. QUEUE AND EVENT SYSTEM

Because this is a single-user system designed for 0 VRAM and constrained RAM, external brokers (Redis/Celery/ARQ) are eliminated. 
*   **Mechanics:** Uses a watched-folder `scripts/ingest.py` or a single `asyncio.Queue` worker running concurrently with the FastAPI app.
*   **Failure Recovery:** `failed_ingests` table tracks files requiring manual review.

---

## I. OBSERVABILITY

VRAM metrics are entirely eliminated. The relevant ceiling is Resident Set Size (RSS) against the 16 GB RAM budget.
*   **Logs:** Structural JSON logging via `structlog`.
*   **Metrics Tracked:**
    *   `ttft_ms`: Time to first token.
    *   `cpu_embed_ms`: CPU latency for bge-small-en-v1.5 embedding.
    *   `rerank_ms`: CPU latency for FlashRank.
    *   `rss_mb`: Resident process memory gauge.
    *   `hallucination_score`: Offline Ragas evaluation.

---

## J. SECURITY + PRIVACY

*   **Local-First Isolation:** The `sqlite-vec` database and preference stores never leave the local SSD. 
*   **API Security:** Secured via static API Key.

---

## K. DEPLOYMENT

Optimized for a native deployment on a 16GB RAM Windows/Linux machine. **Docker Desktop is not used**, as its WSL2 VM alone consumes 2–4 GB of RAM overhead.

### Architecture
*   Deployment consists of a Python Virtual Environment (`venv`).
*   Execution is managed by a single boot script (`start.ps1` or `start.sh`).

### Recomputed RAM Budget (Warm State)

| Item | RAM Usage |
| :--- | :--- |
| OS + browser (Windows floor) | ~4–5 GB |
| Python + onnxruntime/torch-cpu loaded | ~1–2 GB |
| Embedder resident (bge-small-en-v1.5) | ~0.3–0.5 GB |
| FlashRank reranker resident | ~0.2–0.5 GB |
| sqlite-vec mmap working set | ~0.2–1.0 GB |
| App + API buffers | ~0.5–1.0 GB |
| **Total Warm System Allocation** | **~7–9 GB** |
| **Available Headroom on 16 GB** | **~7 GB** |

### Boot Script (`start.ps1`)
1.  Verifies available RSS headroom is sufficient for launch.
2.  Connects to SQLite and initializes `PRAGMA journal_mode=WAL;`.
3.  Boots the FastAPI application and background ingestion threads natively on `localhost:8000`.
