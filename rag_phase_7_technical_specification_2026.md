# Phase 7 — Technical Specification: V2 Elite Thin-Client Architecture

This document serves as the implementation-ready, production-grade engineering specification for the 2026 "Thin-Client" RAG System. It is strictly designed for the 16GB RAM / 2GB VRAM hardware constraint, prioritizing local logic execution and cloud-augmented heavy compute.

---

## A. PROJECT STRUCTURE

The project follows a modular, domain-driven design to ensure clear ownership boundaries, easy swapping of underlying models, and maintainable asynchronous workflows.

```text
rag-system/
├── app/                        # Root application module
│   ├── api/                    # API Gateway (FastAPI)
│   │   ├── routes/             # Endpoints (ingest, retrieve, memory, admin)
│   │   ├── dependencies.py     # Auth, DB connections, Rate Limiting
│   │   └── schemas.py          # Pydantic validation models
│   ├── core/                   # Core business logic & orchestration
│   │   ├── orchestration/      # LangGraph state machines & agent graphs
│   │   ├── agents/             # Individual agent definitions (Planner, Critic)
│   │   └── memory/             # CoALA Memory Engine (Working, Semantic, Episodic)
│   ├── ingestion/              # Data ingestion & parsing pipeline
│   │   ├── workers/            # Async Celery/ARQ workers per modality
│   │   ├── parsers/            # LlamaParse API wrappers, OCR wrappers
│   │   └── chunking/           # Jina Late Chunking & hierarchical splitters
│   ├── retrieval/              # The Retrieval Engine
│   │   ├── engines/            # Vector (sqlite-vec), BM25, Graph (Graphiti)
│   │   ├── rerankers/          # FlashRank (Local), Cohere API (Cloud)
│   │   └── routers/            # Confidence-based routing heuristics
│   ├── models/                 # Cloud & Local model wrappers
│   │   ├── llm/                # Ollama client, llama.cpp fallback wrappers
│   │   └── embeddings/         # Jina v5-text-small local runner
│   └── infrastructure/         # DB & Queue connection management
│       ├── database.py         # SQLite & sqlite-vec setup
│       ├── redis.py            # Redis connection (queues & caching)
│       └── observability.py    # OpenTelemetry / LangSmith hooks
├── deployment/                 # Docker, Compose, and deployment scripts
├── tests/                      # Pytest suite (unit, integration, load)
├── configs/                    # YAML configuration files & ENV templates
└── scripts/                    # CLI tools for DB migrations, evaluation runs
```

### Module Boundaries & Communication
*   **API Gateway:** Handles HTTP/WebSocket requests, validates schemas, pushes tasks to Redis queues or awaits immediate core responses. Never performs heavy compute.
*   **Ingestion Workers:** Completely decoupled. Triggered via Redis queues. They pull files, call external APIs (LlamaParse), generate embeddings locally, and write directly to the DB.
*   **Core Orchestration (LangGraph):** Owns the query lifecycle. Communicates with `Retrieval` and `Models`. Does not directly parse files.
*   **Infrastructure:** Provides the interface to `sqlite-vec` and Redis. Abstracted so the engine doesn't care if it's SQLite or Postgres in the future.

---

## B. DATABASE SCHEMAS

The storage layer relies on SQLite extended with `sqlite-vec` for local, low-memory persistence. Redis is used for ephemeral state and queues.

### 1. Vector DB (sqlite-vec) & Relational Store
**Table: `documents`** (Metadata Store)
*   `doc_id` (UUID, Primary Key)
*   `source_uri` (String, Indexed)
*   `file_hash` (String, deduplication)
*   `modality` (Enum: PDF, CODE, WEB, etc.)
*   `ingested_at` (Timestamp)

**Table: `chunks`** (Vector + Payload)
*   `chunk_id` (UUID, Primary Key)
*   `doc_id` (UUID, Foreign Key -> documents)
*   `parent_chunk_id` (UUID, self-referencing for hierarchical retrieval)
*   `content` (Text, the raw chunk)
*   `embedding` (F32_BLOB, sqlite-vec dimension=384 for Jina)
*   `bm25_tokens` (Text, tokenized representation for FTS5 keyword search)
*   `chunk_index` (Integer, sequential order in document)
*   `salience_score` (Float, base importance score updated over time)

### 2. Graph DB (Graphiti / SQLite Relational)
**Table: `entities`** (Nodes)
*   `entity_id` (UUID)
*   `name` (String)
*   `type` (Enum: PERSON, CONCEPT, PREFERENCE)
*   `summary` (Text)

**Table: `edges`** (Relationships)
*   `edge_id` (UUID)
*   `source_id` (UUID)
*   `target_id` (UUID)
*   `relation` (String, e.g., "PREFERS", "RELATES_TO")
*   `weight` (Float)
*   `created_at` (Timestamp)
*   `last_accessed` (Timestamp, for decay)

### 3. Memory & State Store
**Table: `working_memory_checkpoints`** (LangGraph State)
*   `thread_id` (UUID)
*   `checkpoint_id` (UUID)
*   `state_blob` (JSON)
*   `written_at` (Timestamp)

---

## C. INGESTION WORKERS

Workers run asynchronously via **ARQ (Async Redis Queues)** to ensure the 16GB RAM is not blocked by background ingestion.

### General Worker Flow
1. **Input:** File URI or payload on Redis Queue.
2. **Deduplication:** Hash file content. If hash exists in `documents`, skip or update.
3. **Parsing:** Route to modality-specific parser.
4. **Chunking:** Apply Hierarchical or Late Chunking.
5. **Embedding:** Batch chunks (size 32) -> Jina v5 local.
6. **Indexing:** Transactional insert into `documents` and `chunks`.

### Modality Specifications
*   **PDF Ingestion:**
    *   *Parser:* Call LlamaParse API (extracts markdown + tables).
    *   *Failure:* 3 retries with exponential backoff.
    *   *Chunking:* Semantic Markdown Splitter (respects headers `##`).
*   **Codebase Ingestion:**
    *   *Parser:* Local Tree-sitter. Extracts AST (Functions, Classes).
    *   *Metadata:* File path, language, function signatures.
*   **Website Ingestion:**
    *   *Parser:* Local Firecrawl/Playwright headless scrape. 
    *   *Sanitization:* Readability.js to strip HTML noise.

---

## D. RETRIEVAL ENGINE SPECIFICATION

The pipeline is optimized to minimize network payload to the cloud while maximizing semantic precision.

### Pipeline Flow
1.  **Query Preprocessing:** LLM (Ollama) or local regex rewrites the query for expansion (e.g., resolving pronouns using working memory).
2.  **Routing (Confidence-Based):** 
    *   If query is simple conversational -> Skip retrieval.
    *   Else -> Route to Hybrid Retrieval.
3.  **Hybrid Retrieval (Local Stage 1):**
    *   Dense Search (`sqlite-vec`) -> Top 50.
    *   Sparse Search (SQLite FTS5 BM25) -> Top 50.
    *   Merge via Reciprocal Rank Fusion (RRF) -> Top 100.
4.  **Local Reranking (Local Stage 2):**
    *   Pass Top 100 to `FlashRank` (CPU bound, <200MB RAM).
    *   Prune to Top 15.
5.  **Cloud Reranking (Cloud Stage 3):**
    *   Pass Top 15 + Query to Cohere API (Cross-Encoder).
    *   Prune to Top 5.
6.  **Contextual Compression:** If the Top 5 chunks are larger than the allocated context window constraint (e.g., 4000 tokens), extract exactly the relevant sentences using a lightweight local summarizer (optional fallback).

### Thresholds
*   **Reranking Threshold:** If Cohere returns max relevance < 0.60, trigger an "Anti-Hallucination" halt. Ask the user for clarification or trigger Web Search.

---

## E. MEMORY ENGINE SPECIFICATION

Follows the CoALA (Cognitive Architectures for Language Agents) spec.

### 1. Working Memory (Short-Term)
*   **Storage:** Redis ephemeral keys & LangGraph thread state.
*   **Flow:** Stores the last N turns (conversation history) and current retrieved context.
*   **Clearance:** Wiped or summarized after session ends (timeout > 30 mins).

### 2. Episodic/Procedural Memory (Long-Term Rules)
*   **Storage:** Graph DB (Entities/Edges in SQLite).
*   **Flow:** Async background worker scans Working Memory summaries. If it detects a user preference (e.g., "Always write code in Rust"), it writes a `USER -> PREFERS -> RUST` edge.
*   **Memory Decay:** Edges have a `last_accessed` timestamp. A daily Cron job applies a decay factor (0.95/day) to the `weight`. If `weight < 0.1`, the edge is pruned to keep the graph fast and relevant.

---

## F. AGENT ORCHESTRATION

Powered by **LangGraph** using an asynchronous cyclical graph, but explicitly optimized to avoid infinite loops.

### Agent Definitions
1.  **Router Agent:** Evaluates query confidence. Routes to standard generation OR research.
2.  **Research Agent:** Interfaces with the Retrieval Engine.
3.  **Critic/Reflection Agent:** Evaluates the retrieved context against the user query.

### Orchestration Logic & Safeguards
*   **Max Recursion:** Graph loops (`Research -> Critic -> Research`) are hard-capped at exactly **2 iterations**.
*   **Verification Loop:** 
    *   Critic output schema: `{"is_sufficient": boolean, "missing_info_query": string}`.
    *   If `is_sufficient == true`, route to Final Generation.
    *   If `is_sufficient == false` AND iteration < 2, rewrite query and retrieve again.
    *   If iteration == 2, gracefully degrade: "I found partial information: [X], but I am missing [Y]."
*   **Fallback Handling:** If Ollama Cloud times out (timeout=10s), LangGraph catches the `TimeoutError` and swaps the LLM node to invoke the local `llama.cpp` (Qwen2.5-1.5B) instance for degraded synthesis.

---

## G. API DESIGN

FastAPI-based, async REST architecture.

### Endpoints
*   `POST /v1/ingest/file`
    *   *Payload:* `multipart/form-data` (file), `tags` (list[str])
    *   *Response:* `{"task_id": "uuid", "status": "queued"}` (Async)
*   `GET /v1/ingest/status/{task_id}`
    *   *Response:* `{"status": "completed|failed|processing", "error": null}`
*   `POST /v1/chat/completions` (OpenAI Compatible)
    *   *Payload:* `{"messages": [...], "stream": true, "use_rag": true}`
    *   *Response:* Server-Sent Events (SSE) streaming the LLM tokens.
*   `GET /v1/memory/graph`
    *   *Response:* JSON dump of the current user preferences/entities for UI visualization.

---

## H. QUEUE AND EVENT SYSTEM

*   **Technology:** Redis + ARQ (Python Async Redis Queues). We avoid Celery/RabbitMQ to save RAM.
*   **Queues:**
    1.  `ingestion_queue`: Handles PDF parsing and chunking. (Concurrency: 2 workers).
    2.  `memory_consolidation_queue`: Background tasks to update the Graph DB. (Concurrency: 1 worker).
*   **Idempotency & Retries:** Tasks use the file hash as the task ID. ARQ handles exponential backoff (e.g., 5s, 25s, 125s) if LlamaParse throws a 429 Too Many Requests. Dead-letter queue retains failed tasks for 7 days.

---

## I. OBSERVABILITY

Critical for tuning a constrained system.

*   **Logging:** Structural JSON logging via `structlog`.
*   **Tracing:** LangSmith (if cloud allowed) OR local OpenTelemetry Jaeger container (if 16GB RAM permits, though borderline). We default to writing traces to a local SQLite `traces.db`.
*   **Metrics Tracked:**
    *   `ttft_ms`: Time to first token (API layer).
    *   `retrieval_latency_ms`: SQLite vs Cohere time split.
    *   `vram_usage_mb`: Monitored via `pynvml` before executing Jina embeddings to prevent OOM crashes.
    *   `hallucination_score`: Calculated offline during the nightly Cron evaluation using Ragas.

---

## J. SECURITY + PRIVACY

*   **Local-First Isolation:** The `sqlite-vec` database and `Graphiti` stores never leave the local SSD. 
*   **Cloud API Sanitization:** When sending chunks to Cohere or Ollama Cloud, PII stripping (optional local Presidio NER pass) can be enabled, though it costs CPU time.
*   **API Security:** API Gateway secured via static API Key passed in headers (since it's a single-user system, OAuth is overengineering).

---

## K. DEPLOYMENT

Optimized for a local 16GB RAM Windows/Linux machine.

### Docker Compose Architecture
We use a single `docker-compose.yml` to orchestrate the backend, avoiding heavy virtualization where possible.

**Services:**
1.  **`redis`:** Lightweight Alpine image, limits set to `mem_limit: 512m`.
2.  **`api`:** The FastAPI application. `mem_limit: 2g`. CPU bound.
3.  **`worker_ingest`:** ARQ worker. `mem_limit: 4g`. Needs RAM for chunking large PDFs.
4.  **`ollama` (Optional Local Fallback):** Only booted if explicitly requested, otherwise system relies on external Ollama Cloud URI.

### Resource Limits (The 16GB Budget)
*   OS/Background: ~4 GB
*   Redis: 0.5 GB
*   API (FastAPI + LangGraph + sqlite-vec in memory): 2.0 GB
*   Ingestion Workers: 4.0 GB
*   Jina Embeddings (VRAM): 1.5 GB
*   *Buffer / Headroom:* 4.0 GB (Crucial for preventing OS swapping).

### Boot Script (`start.ps1`)
1.  Checks if port 6379 (Redis) is free.
2.  Checks VRAM availability using `nvidia-smi`.
3.  Boots docker-compose.
4.  Runs DB migrations (Alembic for SQLite schemas).
5.  Readies the API on `localhost:8000`.
