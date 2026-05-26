# Phase 5 — Implementation Roadmap & Evaluation Framework

## 1. Phase-by-Phase Implementation Roadmap & Priority Order

To ensure a smooth transition from a basic setup to an elite, agentic Thin-Client RAG system, the implementation must be rolled out in prioritized phases. 

### Step 1: Core Foundation (MVP) - Priority: HIGH
**Goal:** Establish the basic local-to-cloud pipeline and ensure end-to-end functionality without complex routing.
*   **Action 1:** Set up the local Python native environment (`venv`), install `sqlite-vec`, and configure the document store with WAL mode.
*   **Action 2:** Integrate `MiniLM-L6` for local embeddings and configure the `onnxruntime` CPU provider.
*   **Action 3:** Connect to Cloud Ollama and set up a basic prompt for generative answering.
*   **Action 4:** Build a basic CLI or Gradio UI to test the ingestion-retrieval-generation loop.

### Step 2: Advanced Retrieval & Reranking - Priority: HIGH
**Goal:** Improve the quality of the context injected into the LLM.
*   **Action 1:** Implement Hybrid Search (Dense Vectors via `sqlite-vec` + Keyword BM25).
*   **Action 2:** Integrate `FlashRank` for local CPU Cross-Encoder Reranking of the top retrieved results.
*   **Action 3:** Upgrade ingestion pipeline to use PyMuPDF natively and LlamaParse API as a fallback for complex formats.
*   **Action 4:** Implement Semantic Markdown splitting to preserve context across headers.

### Step 3: Agentic Orchestration & Memory - Priority: MEDIUM
**Goal:** Introduce reasoning, routing, and persistent memory using LangGraph.
*   **Action 1:** Implement LangGraph Orchestrator with a routing node (Simple vs. Complex queries).
*   **Action 2:** Add multi-agent capabilities: Research Agent, Critic/Evaluator Agent.
*   **Action 3:** Implement Working Memory (LangGraph State checkpointing via SQLite).
*   **Action 4:** Integrate persistent user-editable preference store (SQLite KV).

### Step 4: Optimization & Benchmarking - Priority: LOW (but essential for scale)
**Goal:** Finalize the system for daily driver usage, ensuring it stays within the 16GB RAM / 0 VRAM constraints.
*   **Action 1:** Implement an in-process semantic cache (dict + cosine threshold), without Redis.
*   **Action 2:** Finalize dynamic query decomposition for multi-hop questions.
*   **Action 3:** Conduct thorough latency profiling and evaluation (see below).

---

## 2. MVP vs Advanced Features

| Feature Category | MVP (Phase 1) | Advanced (Phases 2-4) |
| :--- | :--- | :--- |
| **Ingestion** | PyPDF / simple text extraction | PyMuPDF + LlamaParse API |
| **Chunking** | Fixed-size (e.g., 512 tokens + overlap) | Semantic Markdown splitting on LlamaParse/PyMuPDF headers |
| **Retrieval** | Single Dense Vector Search | Hybrid (BM25 + Dense) + Reciprocal Rank Fusion (RRF) |
| **Reranking** | None (Raw DB scores) | FlashRank (Local CPU) |
| **Orchestration** | Linear Script (Input -> VectorDB -> LLM) | LangGraph Cyclic Agent State Machine |
| **Memory** | None / Session history only | Versioned preference store |

### Which Features Matter Most?
1.  **Good Ingestion:** If your raw data is parsed poorly, no LLM will rescue it.
2.  **Reranking (FlashRank):** Reranking provides the highest ROI for retrieval accuracy. Retrieving 100 docs locally and reranking them down drastically reduces hallucination.
3.  **Local Embeddings (MiniLM-L6):** Keeping embeddings local saves massive API costs and ensures privacy for your semantic index.

---

## 3. Common Mistakes & Pitfalls

*   **Don't run *any* model on the GPU:** With a 0-VRAM baseline constraint, embeddings and reranking must run strictly on the CPU, while heavy generation is offloaded to the cloud. Trying to shoehorn models onto the GPU will break the system.
*   **Ignoring Table and Image Data:** Standard PDF parsers destroy tables. Ensure the ingestion pipeline falls back to LlamaParse when PyMuPDF struggles.
*   **"Blind" Retrieval:** Trusting the Vector DB's top 3 results without a Cross-Encoder Reranker usually results in sub-optimal context and hallucinations.
*   **Agent Infinite Loops:** In LangGraph, failing to set strict exit conditions or max recursion depths for the Critic Agent can lead to endless loops and massive API bills.

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
| **Embedder RAM** | < 0.5 GB; VRAM unused (0) |
| **Total Resident RAM Usage** | ~7-9 GB (sqlite-vec + orchestration + python) |
| **Indexing Speed** | > 30-400 chunks/s batched |
| **Retrieval Speed (Local DB)** | < 50ms per query (brute force on ~100k vectors) |
| **Local FlashRank rerank (100 chunks)** | < 800ms |
| **Reranking Latency (Cohere)** | < 300ms for 15 chunks (Optional upgrade) |
| **Faithfulness Score** | > 0.95 (Ragas metric) |
| **Context Recall** | > 0.90 (Ragas metric) |
