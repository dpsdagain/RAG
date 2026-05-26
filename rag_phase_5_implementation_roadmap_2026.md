# Phase 5 — Implementation Roadmap & Evaluation Framework

## 1. Phase-by-Phase Implementation Roadmap & Priority Order

To ensure a smooth transition from a basic setup to an elite, agentic Thin-Client RAG system, the implementation must be rolled out in prioritized phases. 

### Step 1: Core Foundation (MVP) - Priority: HIGH
**Goal:** Establish the basic local-to-cloud pipeline and ensure end-to-end functionality without complex routing.
*   **Action 1:** Set up the local Python native environment (`venv`), install `sqlite-vec`, and configure the document store with WAL mode.
*   **Action 2:** Integrate `Cohere` API (`embed-english-v3.0`) for high-quality dense embeddings.
*   **Action 3:** Connect to Cloud Ollama and set up a basic prompt for generative answering.
*   **Action 4:** Build a basic CLI or Gradio UI to test the ingestion-retrieval-generation loop.

### Step 2: Advanced Retrieval & Reranking - Priority: HIGH
**Goal:** Improve the quality of the context injected into the LLM.
*   **Action 1:** Implement Hybrid Search (Dense Vectors via `sqlite-vec` + Keyword BM25).
*   **Action 2:** Integrate `FlashRank` for local CPU Cross-Encoder Reranking of the top retrieved results.
*   **Action 3:** Upgrade ingestion pipeline to use pymupdf4llm natively and LlamaParse API as a fallback for complex formats.
*   **Action 4:** Implement Semantic Markdown splitting to preserve context across headers.

### Step 3: Linear Orchestration & True Memory - Priority: MEDIUM
**Goal:** Introduce predictable linear routing and temporal episodic memory.
*   **Action 1:** Implement Linear Pipeline (`Retrieve -> Rerank -> Evaluate Context -> Generate`).
*   **Action 2:** Add single-retry self-correction logic (Context Evaluator node).
*   **Action 3:** Implement CRAG-style Tavily Web Fallback for out-of-domain queries.
*   **Action 4:** Implement Mem0-style episodic memory (`created_at`, `supersedes_id`) in SQLite.

### Step 4: Optimization & Benchmarking - Priority: LOW (but essential for scale)
**Goal:** Finalize the system for daily driver usage, ensuring it stays within the 16GB RAM / 0 VRAM constraints.
*   **Action 1:** Implement an in-process semantic cache (dict + cosine threshold), without Redis.
*   **Action 2:** Finalize dynamic query decomposition for multi-hop questions.
*   **Action 3:** Conduct thorough latency profiling and evaluation (see below).

---

## 2. MVP vs Advanced Features

| Feature Category | MVP (Phase 1) | Advanced (Phases 2-4) |
| :--- | :--- | :--- |
| **Ingestion** | PyPDF / pymupdf4llm | pymupdf4llm + LlamaParse API |
| **Chunking** | Fixed-size (e.g., 512 tokens + overlap) | Semantic Markdown splitting on LlamaParse/pymupdf4llm headers |
| **Retrieval** | Single Dense Vector Search | Hybrid (BM25 + Dense) + CRAG Web Fallback |
| **Reranking** | None (Raw DB scores) | FlashRank (Local CPU) |
| **Orchestration** | Linear Script (Input -> VectorDB -> LLM) | Linear Pipeline + Single Self-Correction Retry |
| **Memory** | None / Session history only | Mem0-style Episodic Memory |

### Which Features Matter Most?
1.  **Good Ingestion:** If your raw data is parsed poorly, no LLM will rescue it.
2.  **Reranking (FlashRank):** Reranking provides the highest ROI for retrieval accuracy. Retrieving 100 docs locally and reranking them down drastically reduces hallucination.
3.  **High-Quality Embeddings:** Using an API embedder (Cohere) drastically improves initial retrieval over local models.
4.  **Evaluation Harness (Ragas):** Having a Golden Set to benchmark against prevents blind prompt tuning.

---

## 3. Common Mistakes & Pitfalls

*   **Don't run *any* model on the GPU:** With a 0-VRAM baseline constraint, reranking must run strictly on the CPU, while heavy generation and embedding are offloaded to the cloud. Trying to shoehorn models onto the GPU will break the system.
*   **Ignoring Table and Image Data:** Standard PDF parsers destroy tables. Ensure the ingestion pipeline falls back to LlamaParse when pymupdf4llm struggles.
*   **"Blind" Retrieval:** Trusting the Vector DB's top 3 results without a Cross-Encoder Reranker usually results in sub-optimal context and hallucinations.
*   **Multi-Agent Infinite Loops:** Complex cyclical agent graphs often lead to endless loops and massive API bills. Use a predictable linear pipeline with a strict single-retry limit instead.

---

## 4. Evaluation Framework & Metrics

To ensure the system is elite, you must evaluate it systematically using frameworks like **Ragas** or **TruLens**.

### A. Retrieval Accuracy Evaluation
*   **Context Precision:** Measures if the retrieved chunks are relevant to the user query. (High precision = no useless noise in the prompt).
*   **Context Recall:** Measures if *all* the necessary information to answer the question was successfully retrieved from the database.
*   **Hit Rate & MRR (Mean Reciprocal Rank):** Determines how often the most relevant document appears in the Top 1 or Top 3 retrieved chunks before and after FlashRank Reranking.

### B. Hallucination & Generation Evaluation
*   **Faithfulness (Hallucination check):** Measures whether the LLM's final answer is strictly derived from the retrieved context. If it uses outside knowledge, it fails.
*   **Answer Relevance:** Measures how directly the LLM answered the specific user prompt, punishing evasive or overly verbose answers.

### C. Latency Testing
*   **Time to First Token (TTFT):** Essential for UI responsiveness. Target: < 1.0 - 3.0 seconds (network bound).
*   **Retrieval Latency:** Time taken by `sqlite-vec` + FlashRank. Target: < 1000ms.

### D. Memory Quality Evaluation
*   **State Retention Accuracy:** Evaluating if LangGraph accurately remembers user preferences across a 10-turn conversation.

---

## 5. Performance Benchmarks (Targets for 2026 Thin-Client)

| Metric | Target Benchmark |
| :--- | :--- |
| **Embedder API Latency** | < 200ms |
| **Total Resident RAM Usage** | ~6.5-8.5 GB (sqlite-vec + orchestration + python) |
| **Indexing Speed** | > 30-400 chunks/s batched |
| **Retrieval Speed (Local DB)** | < 50ms per query (brute force on ~100k vectors) |
| **Local FlashRank rerank (100 chunks)** | < 800ms |
| **Reranking Latency (Cohere)** | < 300ms for 15 chunks (Optional upgrade) |
| **Faithfulness Score** | > 0.95 (Ragas metric) |
| **Context Recall** | > 0.90 (Ragas metric) |
