# Phase 6 — Architectural Self-Critique & Evolution

## 1. Deconstructing the "Thin-Client" Architecture

The architecture defined in Phases 3 and 4 relies heavily on a "Thin-Client" paradigm: Orchestration (Async Python Pipeline), Vector DB (`sqlite-vec`), and Embeddings (`bge-small`) are local, while Generation (Ollama) is offloaded to the cloud due to the strict 16GB RAM / 0 VRAM hardware constraint.

While theoretically sound for hardware bypass, a rigorous evaluation reveals critical engineering flaws, CPU bottlenecks, and maintenance traps.

---

## 2. Critical Vulnerabilities & Bottlenecks

### A. The Cross-Encoder Inference Trap
**The Flaw:** Sending 100 retrieved chunks over the internet to Cohere API for reranking is deeply inefficient.
- It's not the HTTP upload bandwidth that hurts (50k tokens is only ~200KB), it's the **inference time and billing**. Cross-encoder matrix math on 100 chunks is massively expensive and slow on cloud GPUs.
- **Failure Point:** RAG systems must feel instantaneous. High cloud inference latency destroys the UX and racks up API bills fast.

### B. API Fragility
**The Flaw:** The architecture's uptime relies heavily on the Generation LLM cloud provider (e.g., Ollama Cloud).
- **Failure Point:** If the LLM provider times out or rate-limits you, the entire pipeline crashes, leaving the user with a hung terminal.

### C. Complexity Traps in Multi-Agent Frameworks
**The Flaw:** Using multi-agent cyclical graphs (Supervisor, Research Agent, Critic) introduces a catastrophic complexity trap for a single-user system.
- **The Trap:** If the "Critic" node evaluates the "Research" context and finds it lacking, it sends the graph back to Research. If the documents simply *don't exist* in `sqlite-vec`, the system enters an infinite API loop, burning cloud tokens until the max recursion limit is hit. 

### D. Concurrency & `database is locked`
**The Flaw:** We rely on SQLite for the main document store, vector store, and pipeline checkpointer.
- While SQLite handles reads brilliantly, concurrent writes are bottlenecked by locks. If a background ingestion thread updates documents while the main thread caches a new chunk, the system will throw `database is locked` exceptions.

### E. Memory Drift
**The Flaw:** Long-term preferences decay. If you tell the system "I prefer Python" and two months later "I am now using Rust," the preference store retrieves conflicting rules. 

---

## 3. Scaling Risks & Future-Proofing

*   **Context Window Obsolescence:** In late-2026, API models handle 1M-2M+ tokens natively. Why build RAG? Because stuffing a massive 500k-token SQLite database into an API call takes 15-20+ seconds for Time To First Token (TTFT) and costs dollars per query. This architecture guarantees sub-2-second TTFT, near-zero per-query API costs, and total local data ownership.
*   **CPU Contention:** Because this is a 0-VRAM architecture, everything except Generation runs on your CPU. If you drop a massive 500-page PDF into the ingestion folder, the local `bge-small` embedding pass will consume CPU cycles, requiring careful core management.

---
---

## 4. Evolving the Architecture (The "V2 Elite" Pivot)

To mitigate these flaws, we must optimize API inference, simplify state management, and introduce aggressive CPU scheduling. Here is the improved V2 Architecture.

### Improvement 1: Two-Stage Reranking (Cost & Inference Optimization)
*   **The Fix:** Never send 100 chunks to a cloud cross-encoder.
*   **The Implementation:**
    1. **Stage 1 (Local):** `sqlite-vec` + BM25 retrieves the Top 100.
    2. **Stage 2 (Local):** Implement `FlashRank` (a nano-reranker running purely on CPU/RAM) to rerank the 100 chunks down to the Top 15 instantly.
    3. **Stage 3 (Cloud / Local Final):** If Cohere (optional) is enabled, send the Top 15 to Cohere for a precise cross-encoder rerank down to Top 5. If Cohere is disabled, simply take the Top 5 of the FlashRank 15.
*   **Impact:** Cuts cloud inference latency and costs by 85%.

### Improvement 2: Graceful Degradation (Raw Evidence Fallback)
*   **The Fix:** API failures must not crash the local UX, but running a 1.5B LLM fallback on CPU is too slow and hallucination-prone.
*   **The Implementation:** If the generation API times out, we trigger one retry with backoff. If it still fails, the pipeline traps the error and **surfaces the Top-5 reranked chunks directly to the user as raw evidence**, with a "generation unavailable" notice. This preserves 100% uptime without relying on a dangerous micro-model.

### Improvement 3: Predictable Linear Routing (Complexity Reduction)
*   **The Fix:** Stop using dynamic confidence routers and multi-agent loops. Cosine similarity measures lexical overlap, not factual confidence.
*   **The Implementation:** Unconditionally route every query through the linear retrieval pipeline and into FlashRank. Do not use hardcoded cosine thresholds to skip reranking.

### Improvement 4: Deterministic Preference Deduplication & WAL
*   **The Fix:** Prevent SQLite write-locks and eliminate unsupervised LLM memory corruption.
*   **The Implementation:** 
    1. The SQLite database is strictly initialized with `PRAGMA journal_mode=WAL;` to allow concurrent reads and writes.
    2. Preferences are managed via a deterministic, append-only versioned schema. When a user updates a preference, the old row is marked `superseded`, bypassing the need for a dangerous, hallucination-prone offline LLM cron job.

### Improvement 5: In-Process CPU Contention Management
*   **The Fix:** Do not let background indexing starve foreground query latency.
*   **The Implementation:** Move ingestion to a strictly managed in-process background thread. Apply a CPU Contention Rule: indexing is limited to `N-2` threads and is aggressively paused/niced via semaphores whenever an active user query enters the pipeline.
