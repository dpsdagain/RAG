# V3 Implementation Architecture — Fully Implementable Specification
## 16 GB RAM / 0 VRAM / 500 GB SSD / CPU-First

> [!NOTE]
> This document supersedes Phase 7. It incorporates **every component from Phases 1-2 that is fully implementable** on the target hardware, as validated by the Principal Architect's Audit. No component in this spec requires GPU compute. Every component has been verified to fit within the ~10.5 GB usable RAM budget.

> [!IMPORTANT]
> **What changed from Phase 7:**
> - BM25/FTS5 moved into core (was wrongly deferred)
> - CRAG quality gate added (was wrongly dropped)
> - Citation/RADIO prompting added (was wrongly dropped)
> - Episodic memory added (was wrongly dropped)
> - Chunking upgraded from header-only to sentence-level semantic
> - Chunk metadata expanded (timestamps, page numbers, section titles)
> - Docling added as primary parser (alongside pymupdf4llm)
> - PaddleOCR added for clean scanned documents
> - Crawl4AI added for website ingestion
> - Tree-sitter code parsing specified in detail
> - In-process semantic cache specified
> - Query decomposition added
> - Post-generation faithfulness check added
> - MCP tool interface specified

---

## A. PROJECT STRUCTURE

```text
rag-system/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── ingest.py            # POST /v1/ingest/file, /v1/ingest/url
│   │   │   ├── chat.py              # POST /v1/chat/completions (SSE)
│   │   │   ├── memory.py            # GET/PUT /v1/memory/preferences
│   │   │   └── admin.py             # GET /v1/admin/stats, /v1/admin/health
│   │   ├── schemas.py               # Pydantic request/response models
│   │   └── mcp_server.py            # MCP tool server (RAG as MCP tool)
│   ├── core/
│   │   ├── pipeline.py              # Linear async pipeline orchestrator
│   │   ├── crag_gate.py             # CRAG quality evaluation (cloud LLM)
│   │   ├── query_decomposer.py      # Query decomposition (cloud LLM)
│   │   ├── faithfulness_checker.py  # Post-generation verification (cloud LLM)
│   │   ├── query_router.py          # Simple heuristic query classifier
│   │   └── prompt_templates/
│   │       ├── generation.py        # RAG generation prompt with RADIO citations
│   │       ├── crag_evaluation.py   # CRAG quality gate prompt
│   │       ├── decomposition.py     # Query decomposition prompt
│   │       ├── faithfulness.py      # Post-gen verification prompt
│   │       └── query_rewrite.py     # Failed-query rewrite prompt
│   ├── ingestion/
│   │   ├── router.py                # Routes files to correct parser by type
│   │   ├── parsers/
│   │   │   ├── pdf_local.py         # pymupdf4llm (text-extractable PDFs)
│   │   │   ├── pdf_docling.py       # Docling CPU (complex layouts)
│   │   │   ├── pdf_cloud.py         # LlamaParse API (scanned/visual PDFs)
│   │   │   ├── ocr_paddle.py        # PaddleOCR (clean scanned docs)
│   │   │   ├── ocr_cloud.py         # Cloud VLM API (messy/handwriting OCR)
│   │   │   ├── code_treesitter.py   # Tree-sitter AST parsing
│   │   │   ├── web_crawl4ai.py      # Crawl4AI website extraction
│   │   │   └── base.py              # Abstract parser interface
│   │   ├── chunking/
│   │   │   ├── semantic_chunker.py  # Sentence-level semantic chunking
│   │   │   ├── adaptive_metrics.py  # ICC/DCC chunk quality metrics
│   │   │   └── code_chunker.py      # AST-boundary code chunking
│   │   └── workers/
│   │       └── ingestion_worker.py  # Background threading worker
│   ├── retrieval/
│   │   ├── hybrid_search.py         # Dense + BM25 + RRF fusion
│   │   ├── reranker.py              # FlashRank CPU reranking
│   │   ├── parent_context.py        # Small-to-Big parent injection
│   │   └── semantic_cache.py        # In-process query-response cache
│   ├── memory/
│   │   ├── working_memory.py        # Session state (last N turns)
│   │   ├── episodic_memory.py       # Past conversation storage + retrieval
│   │   ├── preference_store.py      # Versioned user preferences
│   │   └── procedural_rules.py      # Manual procedure/rule store
│   ├── models/
│   │   ├── embedder.py              # bge-small-en-v1.5 ONNX wrapper
│   │   └── llm_client.py            # Cloud LLM client (Ollama/OpenAI/Anthropic)
│   └── infrastructure/
│       ├── database.py              # SQLite + sqlite-vec + FTS5 setup
│       ├── observability.py         # structlog + metrics collection
│       └── config.py                # TOML/YAML config loader
├── configs/
│   ├── config.yaml                  # System configuration
│   └── procedural_rules.yaml       # Manual procedure rules
├── tests/
│   ├── golden_set/                  # Evaluation dataset
│   │   ├── queries.json             # 30 test queries with expected answers
│   │   └── relevant_chunks.json     # Labeled relevant chunk IDs per query
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_pipeline.py
│   └── test_memory.py
├── scripts/
│   ├── start.ps1                    # Windows boot script
│   ├── start.sh                     # Linux boot script
│   ├── evaluate.py                  # Ragas evaluation runner
│   └── backup.py                    # SQLite backup script
└── requirements.txt
```

---

## B. DATABASE SCHEMAS

**CRITICAL PRAGMAS (on every connection):**
```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

### 1. Document Store

**Table: `documents`**
```sql
CREATE TABLE documents (
    doc_id          TEXT PRIMARY KEY,        -- UUID
    source_uri      TEXT NOT NULL,           -- File path, URL, or identifier
    source_type     TEXT NOT NULL,           -- 'pdf', 'code', 'web', 'audio', 'image'
    file_hash       TEXT NOT NULL,           -- SHA-256 of raw file (dedup)
    content_hash    TEXT,                    -- SHA-256 of extracted text (change detection)
    title           TEXT,                    -- Extracted or inferred document title
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    parser_used     TEXT,                    -- 'pymupdf4llm', 'docling', 'llamaparse', etc.
    total_chunks    INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'active'    -- 'active', 'reindexing', 'deleted'
);
CREATE INDEX idx_documents_hash ON documents(file_hash);
CREATE INDEX idx_documents_source ON documents(source_uri);
CREATE INDEX idx_documents_status ON documents(status);
```

### 2. Chunk Store (Vector + Payload)

**Table: `chunks`**
```sql
CREATE TABLE chunks (
    chunk_id        TEXT PRIMARY KEY,        -- UUID
    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    content         TEXT NOT NULL,           -- Raw chunk text
    embedding       F32_BLOB(384),           -- sqlite-vec, bge-small-en-v1.5
    chunk_index     INTEGER NOT NULL,        -- Sequential order in document
    parent_chunk_id TEXT REFERENCES chunks(chunk_id),  -- For small-to-big
    section_title   TEXT,                    -- Header/section this chunk belongs to
    source_page     INTEGER,                -- Page number in original document
    token_count     INTEGER NOT NULL,        -- Token count for context budgeting
    content_hash    TEXT NOT NULL,           -- SHA-256 of chunk content (dedup)
    event_time      TIMESTAMP,              -- When the content was originally written/published
    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When we learned this fact
    metadata_json   TEXT                    -- Extensible JSON metadata blob
);
CREATE INDEX idx_chunks_doc ON chunks(doc_id);
CREATE INDEX idx_chunks_parent ON chunks(parent_chunk_id);
CREATE INDEX idx_chunks_ingested ON chunks(ingested_at);
```

### 3. BM25 Full-Text Search Index

```sql
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    section_title,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

-- Triggers to keep FTS5 in sync with chunks table
CREATE TRIGGER chunks_fts_insert AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, section_title)
    VALUES (NEW.rowid, NEW.content, NEW.section_title);
END;

CREATE TRIGGER chunks_fts_delete AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.section_title);
END;

CREATE TRIGGER chunks_fts_update AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
    VALUES ('delete', OLD.rowid, OLD.content, OLD.section_title);
    INSERT INTO chunks_fts(rowid, content, section_title)
    VALUES (NEW.rowid, NEW.content, NEW.section_title);
END;
```

### 4. Episodic Memory (Conversations)

```sql
CREATE TABLE conversations (
    conversation_id TEXT PRIMARY KEY,        -- UUID
    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    summary         TEXT,                    -- LLM-generated conversation summary
    summary_embedding F32_BLOB(384),         -- For semantic search over past conversations
    turn_count      INTEGER DEFAULT 0
);

CREATE TABLE conversation_turns (
    turn_id         TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
    role            TEXT NOT NULL,           -- 'user' or 'assistant'
    content         TEXT NOT NULL,
    retrieved_chunk_ids TEXT,                -- JSON array of chunk IDs used
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    turn_index      INTEGER NOT NULL
);
CREATE INDEX idx_turns_conv ON conversation_turns(conversation_id);
```

### 5. Working Memory (Session State)

```sql
CREATE TABLE working_memory (
    thread_id       TEXT PRIMARY KEY,
    state_blob      TEXT NOT NULL,           -- JSON: current turns, active context
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 6. User Preferences (Versioned)

```sql
CREATE TABLE user_preferences (
    pref_id         TEXT PRIMARY KEY,
    category        TEXT NOT NULL,           -- 'output_format', 'language', 'domain', etc.
    preference      TEXT NOT NULL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    supersedes_id   TEXT REFERENCES user_preferences(pref_id),
    is_active       BOOLEAN DEFAULT 1
);
CREATE INDEX idx_prefs_active ON user_preferences(is_active);
```

### 7. Procedural Rules

```sql
CREATE TABLE procedural_rules (
    rule_id         TEXT PRIMARY KEY,
    trigger_pattern TEXT NOT NULL,           -- When to apply (e.g., 'query_about_code')
    rule_text       TEXT NOT NULL,           -- The instruction to inject
    priority        INTEGER DEFAULT 0,
    is_active       BOOLEAN DEFAULT 1,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 8. Ingestion Tracking

```sql
CREATE TABLE failed_ingests (
    failure_id      TEXT PRIMARY KEY,
    source_uri      TEXT NOT NULL,
    error_message   TEXT NOT NULL,
    parser_attempted TEXT,
    failed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count     INTEGER DEFAULT 0,
    resolved        BOOLEAN DEFAULT 0
);
```

---

## C. INGESTION PIPELINE

### Routing Logic

```
Input File → Detect Type
  ├── .pdf → Is text-extractable?
  │     ├── YES, simple layout  → pymupdf4llm (local, fastest)
  │     ├── YES, complex layout → Docling CPU (local, layout-aware)
  │     └── NO, scanned/visual  → PaddleOCR (clean) or LlamaParse API (complex)
  ├── .py/.js/.ts/.go/.rs/etc → Tree-sitter AST parser
  ├── .docx/.pptx/.xlsx       → Docling CPU
  ├── .md/.txt/.csv            → Direct text loading
  ├── URL                      → Crawl4AI (headless Chromium → Markdown)
  └── .png/.jpg (standalone)   → Cloud VLM API (Gemini Flash) → description text
```

### Parser Specifications

#### PDF Parsing (3-Tier)
1. **pymupdf4llm** (Tier 1 — default):
   - Markdown-aware text extraction preserving headers, bold, lists
   - RAM: ~50 MB. Speed: ~50-100 pages/second
   - Use when: `len(extracted_text) / num_pages > 100 chars` (has real text)

2. **Docling CPU** (Tier 2 — complex layouts):
   - Loads Granite-Docling-258M VLM via ONNX INT8
   - RAM: ~300 MB during parsing
   - Speed: ~5-15 seconds per page on CPU
   - Use when: Multi-column, tables, mixed layouts that pymupdf4llm mangles
   - **MUST unload after batch completes** to free RAM

3. **LlamaParse API** (Tier 3 — scanned/visual):
   - Cloud API call. Zero local RAM.
   - Use when: Local text extraction yields <100 chars/page (scanned doc)
   - Fallback for OCR failures

#### Code Parsing (Tree-sitter)
- **Supported languages:** Python, JavaScript, TypeScript, Go, Rust, Java, C, C++
- **Chunk boundaries:** Functions, classes, methods, top-level declarations
- **Metadata extracted per chunk:**
  - `function_name`, `class_name`, `language`, `file_path`
  - `start_line`, `end_line` (for source linking)
  - `imports` (dependencies referenced)
- **RAM:** ~30 MB for all grammars loaded

#### Website Parsing (Crawl4AI)
- Launches headless Chromium, renders JS, extracts clean Markdown
- **RAM:** ~300-400 MB during crawl (headless browser)
- **Rule:** Run ONLY during batch ingestion. Kill browser process after crawl completes.
- **Metadata:** `url`, `crawled_at`, `page_title`, `domain`

#### OCR (PaddleOCR — clean documents)
- Detection + Recognition + Angle Classification models
- RAM: ~400 MB during OCR
- Speed: 1-3 seconds per page for clean forms/invoices
- Use for: Scanned but clean-layout documents (forms, invoices, printed text)

### Chunking Strategy: Sentence-Level Semantic Chunking

**This replaces the Phase 7 "header-only Markdown splitting."**

#### Algorithm

```
INPUT:  Extracted document text + metadata
OUTPUT: List of semantically coherent chunks with metadata

1. SPLIT text into sentences using spaCy sentence tokenizer
   (or regex fallback: split on '. ', '? ', '! ', newline)

2. EMBED every sentence using bge-small-en-v1.5
   (batch embed for speed: all sentences in one ONNX call)

3. COMPUTE inter-sentence cosine similarity:
   sim[i] = cosine(embedding[i], embedding[i+1])

4. IDENTIFY SPLIT POINTS where:
   - sim[i] < SEMANTIC_THRESHOLD (default: 0.3)
   - OR a Markdown header (# or ##) is encountered
   - OR accumulated tokens exceed MAX_CHUNK_TOKENS (default: 512)

5. GROUP consecutive sentences between split points into chunks

6. POST-PROCESS:
   - If chunk < MIN_CHUNK_TOKENS (default: 50), merge with previous chunk
   - If chunk > MAX_CHUNK_TOKENS, hard-split at sentence boundary nearest to midpoint
   - Preserve tables as single chunks (detect by '|' pipe characters)
   - Preserve code blocks as single chunks (detect by ``` markers)

7. ASSIGN PARENT CHUNKS:
   - Every chunk gets a parent_chunk_id pointing to a "section chunk"
   - Section chunks = full content under each ## header (or full document if no headers)

8. COMPUTE QUALITY METRICS per chunk:
   - ICC (Intrachunk Cohesion) = mean cosine similarity between all sentence pairs within chunk
   - DCC (Document Contextual Coherence) = cosine similarity between chunk embedding and full-doc embedding
   - Flag chunks with ICC < 0.2 for manual review
```

#### Special Cases

| Content Type | Chunking Rule |
|---|---|
| **Tables** | Keep entire table as one chunk, even if > 512 tokens. Store table header as `section_title`. |
| **Code blocks** | Keep entire code block as one chunk. Store language and function name in metadata. |
| **Lists** | Keep entire list under one header as one chunk if < 1024 tokens. |
| **Short documents** | If total document < 512 tokens, store as single chunk with no parent. |

### Embedding Specification

- **Model:** BAAI/bge-small-en-v1.5
- **Runtime:** ONNX Runtime (CPU)
- **Dimensions:** 384
- **Batch size:** 64 sentences per ONNX inference call
- **Thread limits:** `OMP_NUM_THREADS=4`, `sess_options.intra_op_num_threads=4`
- **Normalization:** L2-normalize all embeddings before storage
- **Estimated speed:** ~5 ms per single query, ~500 ms for 64-sentence batch

### Ingestion Worker

**Threading model:** `threading.Thread` (NOT `multiprocessing`)

**Rationale:** ONNX Runtime releases the GIL during C++ inference. Using threads avoids the Windows `multiprocessing.spawn` problem of duplicating the ONNX model in RAM. The ingestion thread shares the same ONNX session as the query thread.

**CPU Contention Rule:**
- Ingestion thread runs at `threading.NORM_PRIORITY - 2` (lower priority)
- A `threading.Semaphore` gates embedding calls: when a user query is active, the ingestion thread blocks on the semaphore until the query completes
- Maximum ingestion batch size: 32 chunks between semaphore checks

**Idempotency:**
- Before ingestion, compute `file_hash` (SHA-256 of raw file)
- If `file_hash` exists in `documents` table → skip
- After text extraction, compute `content_hash` (SHA-256 of extracted text)
- If `content_hash` matches existing doc with same `source_uri` → skip (content unchanged despite file metadata change)

---

## D. RETRIEVAL ENGINE

### Pipeline Flow

```
User Query
    │
    ▼
┌─────────────────────────────┐
│ 1. SEMANTIC CACHE CHECK     │  In-process dict. If cosine(query, cached) > 0.95
│    Hit? → Return cached     │  → return cached response immediately (<5ms)
└─────────────┬───────────────┘
              │ Cache miss
              ▼
┌─────────────────────────────┐
│ 2. QUERY ROUTING            │  Simple heuristic classifier:
│    conversational? → skip   │  - Length < 5 words AND no question word → conversational
│    retrieval-needed? → go   │  - Contains '?', 'what', 'how', 'why', 'find', 'show' → retrieval
└─────────────┬───────────────┘
              │ Needs retrieval
              ▼
┌─────────────────────────────┐
│ 3. QUERY DECOMPOSITION      │  Cloud LLM call (1 call, ~100 tokens):
│    (if query is complex)    │  "Break this into 1-3 focused sub-queries"
│                             │  Trigger: query contains 'compare', 'vs', 'and',
│                             │  or has multiple clauses (detected by comma/semicolon count)
└─────────────┬───────────────┘
              │ 1-3 sub-queries
              ▼
┌─────────────────────────────┐
│ 4. HYBRID RETRIEVAL         │  For EACH sub-query:
│    Dense + BM25 + RRF       │
│                             │  a) Dense: sqlite-vec cosine search → Top 50
│                             │  b) Sparse: FTS5 BM25 search → Top 50
│                             │  c) Merge via RRF (k=60) → Top 100
│                             │
│                             │  If multiple sub-queries: union all results,
│                             │  deduplicate by chunk_id, re-apply RRF
└─────────────┬───────────────┘
              │ Top 100 unique chunks
              ▼
┌─────────────────────────────┐
│ 5. FLASHRANK RERANKING      │  CPU cross-encoder reranking
│    100 → 15                 │  Model: FlashRank (~200 MB resident)
│                             │  Latency: <800ms for 100 chunks on CPU
└─────────────┬───────────────┘
              │ Top 15 chunks
              ▼
┌─────────────────────────────┐
│ 6. PARENT CONTEXT INJECTION │  For each of Top 15 chunks:
│    (Small-to-Big)           │  - Fetch parent_chunk_id content from SQLite
│                             │  - Inject as surrounding context
│                             │  BUDGET CAP: Total context ≤ 12,000 tokens
│                             │  If exceeded: drop lowest-ranked parents first
└─────────────┬───────────────┘
              │ Context payload (≤ 12K tokens)
              ▼
┌─────────────────────────────┐
│ 7. CRAG QUALITY GATE        │  Cloud LLM call (1 call, ~600 tokens):
│                             │  "Rate retrieval quality: SUFFICIENT / PARTIAL / INSUFFICIENT"
│                             │
│                             │  SUFFICIENT → proceed to generation
│                             │  PARTIAL → proceed but flag low confidence
│                             │  INSUFFICIENT → trigger query rewrite (1 retry)
│                             │                 if retry also INSUFFICIENT → abstain
└─────────────┬───────────────┘
              │ Quality-verified context
              ▼
┌─────────────────────────────┐
│ 8. GENERATION               │  Cloud LLM call with RADIO citation prompt:
│    (with RADIO citations)   │  "Answer using ONLY the provided context.
│                             │   For every claim, cite [chunk_id, page_number].
│                             │   If context is insufficient, say 'I don't have
│                             │   enough information to answer this.'"
│                             │  SSE streaming response
└─────────────┬───────────────┘
              │ Generated response
              ▼
┌─────────────────────────────┐
│ 9. FAITHFULNESS CHECK       │  Cloud LLM call (1 call, ~800 tokens):
│    (Post-generation)        │  "Is every claim in this response supported by
│                             │   the provided context? Flag unsupported claims."
│                             │
│                             │  If unsupported claims found:
│                             │  → Append warning to response
│                             │  → Log for offline review
└─────────────┬───────────────┘
              │ Verified response
              ▼
┌─────────────────────────────┐
│ 10. CACHE + LOG + RETURN    │  - Cache query embedding + response in semantic cache
│                             │  - Log to conversation_turns table (episodic memory)
│                             │  - Return to user via SSE stream
└─────────────────────────────┘
```

### Reciprocal Rank Fusion (RRF) Implementation

```python
def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],   # [(chunk_id, score), ...]
    sparse_results: list[tuple[str, float]],
    k: int = 60,
    top_n: int = 100
) -> list[tuple[str, float]]:
    """Merge dense + sparse results using RRF."""
    rrf_scores: dict[str, float] = {}

    for rank, (chunk_id, _) in enumerate(dense_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

    for rank, (chunk_id, _) in enumerate(sparse_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]
```

### Semantic Cache Specification

```python
# In-process cache — no Redis needed
class SemanticCache:
    max_entries: int = 500
    similarity_threshold: float = 0.95
    ttl_seconds: int = 3600  # 1 hour

    # Storage: dict[str, CacheEntry]
    # CacheEntry: (query_embedding, response_text, timestamp, hit_count)

    # On query:
    #   1. Embed query with bge-small
    #   2. Compute cosine similarity against all cached query embeddings
    #   3. If max similarity > 0.95 AND age < TTL → return cached response
    #   4. Else → cache miss, proceed with retrieval
    #   5. LRU eviction when cache exceeds max_entries
```

---

## E. MEMORY ENGINE

### L2: Working Memory

- Stores the last **10 conversation turns** (5 user + 5 assistant) in `working_memory` table
- Injected into every generation prompt as conversation history
- When turn count exceeds 10, the oldest pair is evicted
- Total working memory budget: ≤ 2,000 tokens

### L3: Episodic Memory (Cross-Session Recall)

**On every conversation end (or every 5 turns):**
1. Send the conversation to cloud LLM: "Summarize this conversation in 2-3 sentences focusing on key facts discussed and decisions made."
2. Embed the summary with bge-small (local, ~5ms)
3. Store in `conversations` table with `summary_embedding`

**On every new query:**
1. Embed the query
2. Search `conversations.summary_embedding` via sqlite-vec → Top 3 relevant past conversations
3. Inject summaries into the generation prompt as "Relevant context from past interactions"

**RAM cost:** Negligible. Conversation embeddings share the existing sqlite-vec infrastructure.

### L3: Semantic Memory

This IS the RAG knowledge base (chunks table). Already fully specified above.

### L4: Procedural Rules

- Stored in `procedural_rules` table and/or `configs/procedural_rules.yaml`
- Manually authored for v1 (not LLM-generated)
- Matched against queries by simple keyword pattern matching
- Injected into generation prompt when triggered

**Example rules:**
```yaml
rules:
  - trigger: "code|programming|function|class|debug"
    instruction: "Always include code examples. Use Python unless the user specifies another language."
    priority: 10

  - trigger: "explain|teach|learn"
    instruction: "Explain step by step. Start with a high-level overview, then dive into details."
    priority: 5

  - trigger: "compare|vs|difference|versus"
    instruction: "Use a comparison table to highlight key differences."
    priority: 8
```

### User Preferences

- Versioned append-only store in `user_preferences` table
- Queried at generation time: `SELECT preference FROM user_preferences WHERE is_active = 1`
- Injected into system prompt as persistent user rules
- User can view/edit/deactivate via `GET/PUT /v1/memory/preferences`

---

## F. PROMPT TEMPLATES

### RAG Generation Prompt (with RADIO Citations)

```
SYSTEM:
You are a knowledgeable assistant. Answer the user's question using ONLY the provided context documents.

RULES:
1. Every factual claim MUST cite its source using [chunk_id | page X] format.
2. If the context is insufficient to answer the question fully, explicitly state:
   "I don't have enough information in my knowledge base to fully answer this."
3. Do NOT use information from your training data — only the provided context.
4. If multiple sources disagree, note the disagreement and cite both.

{procedural_rules}

USER PREFERENCES:
{active_preferences}

RELEVANT PAST INTERACTIONS:
{episodic_memory_summaries}

CONTEXT DOCUMENTS:
{retrieved_chunks_with_metadata}

CONVERSATION HISTORY:
{working_memory_turns}

USER QUESTION:
{query}
```

### CRAG Quality Gate Prompt

```
You are a retrieval quality evaluator. Given a user question and retrieved documents,
rate the retrieval quality.

USER QUESTION: {query}

RETRIEVED DOCUMENTS:
{top_15_chunks}

Rate the retrieval as one of:
- SUFFICIENT: The documents contain enough information to fully answer the question.
- PARTIAL: The documents contain some relevant information but gaps exist.
- INSUFFICIENT: The documents are not relevant to the question.

Respond with ONLY one word: SUFFICIENT, PARTIAL, or INSUFFICIENT.
```

### Query Decomposition Prompt

```
Break the following complex question into 1-3 simple, focused sub-questions
that can each be answered by searching a document database independently.

Question: {query}

Respond as a JSON array of strings. Example: ["sub-question 1", "sub-question 2"]
```

### Faithfulness Check Prompt

```
Compare the RESPONSE against the SOURCE DOCUMENTS.
For each claim in the response, determine if it is supported by the source documents.

SOURCE DOCUMENTS:
{retrieved_chunks}

RESPONSE:
{generated_response}

List any claims that are NOT supported by the source documents.
If all claims are supported, respond with: "ALL CLAIMS VERIFIED"
```

---

## G. ORCHESTRATION: LINEAR PIPELINE WITH SAFEGUARDS

### Core Flow

```
Query → Cache Check → Route → [Decompose] → Hybrid Retrieve → Rerank →
Parent Inject → CRAG Gate → Generate (with citations) → Faithfulness Check →
Cache + Log → Return
```

### Failure Handling

| Failure Point | Action |
|---|---|
| Cloud LLM unreachable (generation) | 1 retry with 5s backoff → surface Top 5 raw chunks as evidence |
| Cloud LLM unreachable (CRAG) | Skip CRAG gate, proceed to generation with warning flag |
| Cloud LLM unreachable (decomposition) | Skip decomposition, use original query |
| Cloud LLM unreachable (faithfulness) | Skip check, return response with "unverified" flag |
| sqlite-vec returns 0 results | Attempt BM25-only search → if still 0, respond "No relevant documents found" |
| FlashRank throws exception | Return top 15 from hybrid search without reranking |
| Ingestion parser fails | Log to `failed_ingests`, continue with next file |

### Abstention Logic

The system MUST abstain (refuse to generate) when:
1. CRAG gate returns INSUFFICIENT on both original and rewritten query
2. All retrieved chunks have FlashRank scores below a calibrated minimum threshold
3. The query explicitly asks about a document that doesn't exist in the knowledge base

**Abstention response:** "I don't have enough information in my knowledge base to answer this question. Here are the closest documents I found: [list top 3 chunks with source info]."

---

## H. MCP TOOL SERVER

The RAG system exposes itself as an MCP (Model Context Protocol) tool, making it interoperable with any MCP-compatible agent or application.

### Tool Definition

```json
{
  "name": "rag_search",
  "description": "Search the local knowledge base for information relevant to a query. Returns ranked, cited evidence from ingested documents.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": { "type": "string", "description": "The search query" },
      "max_results": { "type": "integer", "default": 5 },
      "source_filter": { "type": "string", "description": "Optional: filter by source type (pdf, code, web)" }
    },
    "required": ["query"]
  }
}
```

### Implementation
- Uses the same retrieval pipeline (hybrid search → rerank → parent context)
- Returns structured JSON with chunk content, source metadata, relevance scores
- Runs as a local MCP server alongside the FastAPI app on a separate port

---

## I. OBSERVABILITY

### Structured Logging (structlog)

Every pipeline step logs a JSON event:
```json
{
  "timestamp": "2026-05-26T18:30:00Z",
  "level": "info",
  "event": "retrieval_complete",
  "query_id": "uuid",
  "dense_results": 50,
  "sparse_results": 50,
  "rrf_merged": 87,
  "reranked_top15_scores": [0.92, 0.87, ...],
  "retrieval_ms": 145,
  "rerank_ms": 620
}
```

### Metrics Tracked

| Metric | Target | Alert Threshold |
|---|---|---|
| `ttft_ms` | < 2000 ms | > 5000 ms |
| `retrieval_ms` | < 200 ms | > 1000 ms |
| `rerank_ms` | < 800 ms | > 2000 ms |
| `embed_ms` | < 50 ms | > 200 ms |
| `crag_verdict` | SUFFICIENT rate > 80% | < 50% (bad retrieval quality) |
| `faithfulness_pass_rate` | > 95% | < 80% |
| `rss_mb` | < 8500 MB | > 12000 MB |
| `cache_hit_rate` | > 20% | N/A |
| `total_chunks` | tracked | N/A |

### Request Logging

Every query is logged to a `request_log` table for offline analysis:
```sql
CREATE TABLE request_log (
    request_id      TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    sub_queries     TEXT,              -- JSON array (if decomposed)
    crag_verdict    TEXT,
    faithfulness    TEXT,
    retrieved_chunks TEXT,             -- JSON array of chunk_ids
    response_length INTEGER,
    total_latency_ms INTEGER,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## J. API DESIGN

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Query the RAG system (SSE streaming, OpenAI-compatible) |
| `POST` | `/v1/ingest/file` | Upload a file for ingestion |
| `POST` | `/v1/ingest/url` | Submit a URL for crawling and ingestion |
| `GET`  | `/v1/memory/preferences` | List active user preferences |
| `PUT`  | `/v1/memory/preferences` | Add or update a preference |
| `GET`  | `/v1/memory/conversations` | List past conversation summaries |
| `GET`  | `/v1/admin/stats` | System stats (chunk count, RAM, cache hit rate) |
| `GET`  | `/v1/admin/health` | Health check |
| `GET`  | `/v1/admin/failed-ingests` | List failed ingestions |

---

## K. SECURITY + PRIVACY

- **Local-first:** All SQLite databases, vectors, and documents stay on local SSD
- **API Key auth:** Static API key for FastAPI endpoints (single user)
- **Cloud LLM calls:** Only the query + retrieved context is sent to cloud — never the full database
- **No telemetry:** Zero data collection, zero external analytics

---

## L. DEPLOYMENT

### Native Python (no Docker)

```powershell
# start.ps1
$ErrorActionPreference = "Stop"

# Check RAM headroom
$freeRAM = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
if ($freeRAM -lt 6) {
    Write-Error "Insufficient RAM. Need at least 6 GB free. Currently: $([math]::Round($freeRAM, 1)) GB"
    exit 1
}

# Activate environment
& "$PSScriptRoot\venv\Scripts\Activate.ps1"

# Set ONNX thread limits
$env:OMP_NUM_THREADS = "4"
$env:TOKENIZERS_PARALLELISM = "false"

# Boot
python -m uvicorn app.api.routes:app --host 0.0.0.0 --port 8000
```

### RAM Budget (Warm State)

| Component | RAM Usage |
|---|---|
| OS + browser (Windows 11) | ~4.5 GB |
| Python + ONNX runtime | ~0.8 GB |
| bge-small-en-v1.5 model | ~0.15 GB |
| FlashRank model | ~0.3 GB |
| sqlite-vec mmap working set (100K chunks) | ~0.5 GB |
| FTS5 index | ~0.1 GB |
| Semantic cache (500 entries) | ~0.05 GB |
| App + FastAPI + buffers | ~0.5 GB |
| **Total** | **~6.9 GB** |
| **Available headroom** | **~9.1 GB** |

### Scale Ceiling

| Corpus Size | Feasibility |
|---|---|
| < 50K chunks | ✅ Comfortable — well within all limits |
| 50K – 200K chunks | ✅ Feasible — mmap working set grows to ~1.5 GB |
| 200K – 500K chunks | ⚠️ Tight — brute-force search reaches 300-500ms, consider migrating to LanceDB |
| > 500K chunks | ❌ Requires ANN index — migrate to LanceDB or Qdrant |

---

## M. EVALUATION FRAMEWORK

### Golden Set Requirements

Create `tests/golden_set/queries.json` with **30 minimum** test cases:

```json
[
  {
    "query_id": "q001",
    "query": "What is the chunking strategy used in this project?",
    "expected_answer_contains": ["sentence-level semantic", "ICC", "DCC"],
    "relevant_doc_ids": ["doc_uuid_1"],
    "relevant_chunk_ids": ["chunk_uuid_1", "chunk_uuid_2"],
    "difficulty": "simple"
  },
  {
    "query_id": "q002",
    "query": "Compare the retrieval strategies in Document A vs Document B",
    "expected_answer_contains": ["hybrid search", "BM25"],
    "relevant_doc_ids": ["doc_uuid_a", "doc_uuid_b"],
    "relevant_chunk_ids": ["chunk_a1", "chunk_b1"],
    "difficulty": "multi-hop"
  }
]
```

### Metrics & Targets

| Metric | Target | Tool |
|---|---|---|
| Context Precision | > 0.85 | Ragas |
| Context Recall | > 0.90 | Ragas |
| Faithfulness | > 0.95 | Ragas |
| Answer Relevance | > 0.85 | Ragas |
| MRR@5 | > 0.70 | Custom script |
| CRAG pass rate | > 80% | Custom logging |
| Abstention rate on unanswerable | > 90% | Custom test set |
| TTFT | < 2000 ms | Custom timer |
