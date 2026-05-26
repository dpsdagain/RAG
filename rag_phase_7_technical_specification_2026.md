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
│   │   ├── orchestration/      # Async pipeline state and routing logic
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
*   `embedding` (F32_BLOB, sqlite-vec dimension=384 for confirmed embedder: BAAI/bge-small-en-v1.5)
*   `chunk_index` (Integer, sequential order in document)
*   `parent_chunk_id` (UUID, Optional, Foreign Key -> chunks)

**Table: `chunks_fts`** (Virtual FTS5 Table for BM25)
*   `CREATE VIRTUAL TABLE chunks_fts USING fts5(content, content='chunks', content_rowid='rowid')`

### 2. State & Memory Store
**Table: `working_memory`** (Linear Pipeline State)
*   `thread_id` (UUID)
*   `state_blob` (JSON)
*   `written_at` (Timestamp)

**Table: `user_preferences`** (Episodic Mem0-style store)
*   `pref_id` (UUID)
*   `preference` (String)
*   `created_at` (Timestamp)
*   `supersedes_id` (UUID, Optional, points to outdated preference)
*   `is_active` (Boolean)

---

## C. INGESTION WORKERS

Ingestion runs via a **multi-process background worker** (using Python `multiprocessing`). Idempotency is enforced by a file-hash check against the `documents` table. Failed parses are logged to a local `failed_ingests` SQLite table.

**GIL & CPU Contention Rule:** Because heavy PDF parsing and chunking block the Global Interpreter Lock (GIL) and freeze the UI, ingestion is strictly isolated to a separate OS-level process. Indexing is throttled to N-2 cores. ONNX thread limits are explicitly set (`OMP_NUM_THREADS`, `sess_options.intra_op_num_threads`) to prevent runaway thread spawning during embeddings.

### Modality Specifications
*   **PDF Ingestion:**
    *   *Parser:* Attempt **pymupdf4llm** local text extraction first. Route to the **LlamaParse API** *only* when local text-extraction coverage is below threshold (e.g., heavily scanned/complex).
    *   *Chunking:* Strict Semantic Markdown Splitter. Boundaries are triggered on `#` or `##` headers. Maximum token size is 512 tokens before a hard split. Every chunk must inherit and track its `parent_chunk_id`.
    *   *Embedding:* Local `bge-small-en-v1.5` via ONNX runtime.
*   **Codebase Ingestion:**
    *   *Parser:* Local Tree-sitter AST extraction.

---

## D. RETRIEVAL ENGINE SPECIFICATION

The pipeline is optimized for maximum semantic precision running on local CPU resources.

### Pipeline Flow
1.  **Routing (Confidence-Based):** 
    *   If query is simple conversational -> Skip retrieval.
2.  **Hybrid Retrieval (Stage 1):**
    *   Dense Search (`sqlite-vec` via local bge-small) -> Top 50.
    *   Sparse Search (SQLite FTS5 BM25) -> Top 50.
    *   Merge via Reciprocal Rank Fusion (RRF) -> Top 100.
3.  **Local Reranking (Stage 2):**
    *   Pass Top 100 to `FlashRank` (CPU bound, <200MB RAM).
    *   Prune to Top 15.
4.  **Parent-Context Injection (Small-to-Big):**
    *   For the Top 15 chunks, retrieve their associated `parent_chunk_id` content to inject broader surrounding context before final generation.

### Thresholds
*   **Relevance Threshold:** Calibrated dynamically against a local Ragas evaluation Golden Set (no hardcoded cosine thresholds).

---

## E. MEMORY ENGINE SPECIFICATION

*   **Working Memory:** Local `working_memory` SQLite checkpointer. Stores the last N turns.
*   **Versioned Preference Store (Long-Term):** User-editable SQLite key-value pairs representing persistent rules (e.g., "Always output Python code"). Optional chronological decay.

---

## F. ORCHESTRATION: LINEAR PIPELINE

Replacing the complex multi-agent cyclical graph with a predictable **Linear Pipeline** (`Retrieve -> Rerank -> Generate`).

### Orchestration Logic & Safeguards
*   **Self-Correction:** On API failure or generation timeout, it triggers exactly **one** targeted query rewrite and retry.
*   **Graceful Degradation:** If the retry fails, the pipeline gracefully degrades without infinite looping, returning the raw reranked chunks directly to the user as evidence.

---

## G. API DESIGN

FastAPI-based async REST architecture.
*   `POST /v1/ingest/file` (Triggers in-process background worker)
*   `POST /v1/chat/completions` (OpenAI Compatible, SSE streaming)

---

## H. QUEUE AND EVENT SYSTEM

Because this is a single-user system designed for 0 VRAM and constrained RAM, external brokers (Redis/Celery/ARQ) are eliminated. 
*   **Mechanics:** Uses a watched-folder `scripts/ingest.py` running as a dedicated `multiprocessing` background service isolated from the FastAPI app.
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
| FlashRank reranker resident | ~0.2–0.5 GB |
| sqlite-vec mmap working set | ~0.2–1.0 GB |
| App + API buffers | ~0.5–1.0 GB |
| **Total Warm System Allocation** | **~6.5–8.5 GB** |
| **Available Headroom on 16 GB** | **~7.5 GB** |

### Boot Script (`start.ps1`)
1.  Verifies available RSS headroom is sufficient for launch.
2.  Connects to SQLite and initializes `PRAGMA journal_mode=WAL;`.
3.  Boots the FastAPI application and background `multiprocessing` ingestion processes natively on `localhost:8000`.
