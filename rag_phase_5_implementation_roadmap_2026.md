# Phase 5 — Implementation Roadmap & Evaluation Framework

## 1. Phase-by-Phase Implementation Roadmap & Priority Order

To ensure a smooth transition from a basic setup to an elite, agentic Thin-Client RAG system, the implementation must be rolled out in prioritized phases. 

### Step 1: Core Foundation (MVP) - Priority: HIGH
**Goal:** Establish the basic local-to-cloud pipeline and ensure end-to-end functionality without complex routing.
*   **Action 1:** Set up the local Python environment, install `sqlite-vec`, and configure the document store.
*   **Action 2:** Integrate `Jina v5-text-small` for local embeddings and configure chunking (basic overlap for now).
*   **Action 3:** Connect to Cloud Ollama and set up a basic prompt for generative answering.
*   **Action 4:** Build a basic CLI or Gradio UI to test the ingestion-retrieval-generation loop.

### Step 2: Advanced Retrieval & Reranking - Priority: HIGH
**Goal:** Improve the quality of the context injected into the LLM.
*   **Action 1:** Implement Hybrid Search (Dense Vectors via `sqlite-vec` + Keyword BM25).
*   **Action 2:** Integrate Cohere API for Cross-Encoder Reranking of the top retrieved results.
*   **Action 3:** Upgrade ingestion pipeline to use LlamaParse API for processing PDFs, Tables, and complex formats.
*   **Action 4:** Implement Jina "Late Chunking" to preserve context across chunk boundaries.

### Step 3: Agentic Orchestration & Memory - Priority: MEDIUM
**Goal:** Introduce reasoning, routing, and persistent memory using LangGraph.
*   **Action 1:** Implement LangGraph Orchestrator with a routing node (Simple vs. Complex queries).
*   **Action 2:** Add multi-agent capabilities: Research Agent, Critic/Evaluator Agent.
*   **Action 3:** Implement Working Memory (LangGraph State checkpointing via `PostgresSaver`).
*   **Action 4:** Integrate Temporal/Graph memory (Graphiti) for user preference retention.

### Step 4: Optimization & Benchmarking - Priority: LOW (but essential for scale)
**Goal:** Finalize the system for daily driver usage, ensuring it stays within the 16GB RAM / 2GB VRAM constraints.
*   **Action 1:** Implement Semantic Caching (Redis/local alternative) for frequent queries.
*   **Action 2:** Finalize dynamic query decomposition for multi-hop questions.
*   **Action 3:** Conduct thorough latency profiling and evaluation (see below).

---

## 2. MVP vs Advanced Features

| Feature Category | MVP (Phase 1) | Advanced (Phases 2-4) |
| :--- | :--- | :--- |
| **Ingestion** | PyPDF / simple text extraction | LlamaParse VLM + Markdown structure preservation |
| **Chunking** | Fixed-size (e.g., 512 tokens + overlap) | Context-Aware Jina Late Chunking |
| **Retrieval** | Single Dense Vector Search | Hybrid (BM25 + Dense) + Reciprocal Rank Fusion (RRF) |
| **Reranking** | None (Raw DB scores) | Cohere Cross-Encoder |
| **Orchestration** | Linear Script (Input -> VectorDB -> LLM) | LangGraph Cyclic Agent State Machine |
| **Memory** | None / Session history only | CoALA architecture (Working, Semantic, Episodic) |

### Which Features Matter Most?
1.  **LlamaParse / Good Ingestion:** If your raw data is parsed poorly (e.g., messy tables), no LLM will rescue it. Garbage in = Garbage out.
2.  **Reranking (Cohere):** Reranking provides the highest ROI for retrieval accuracy. Retrieving 100 docs locally and reranking the top 5 in the cloud drastically reduces hallucination.
3.  **Local Embeddings (Jina):** Keeping embeddings local saves massive API costs and ensures privacy for your semantic index.

---

## 3. Common Mistakes & Pitfalls

*   **Ignoring Table and Image Data:** Standard PDF parsers destroy tables. Relying on basic PyPDF instead of LlamaParse will ruin technical document retrieval.
*   **Over-Chunking:** Slicing documents too small loses semantic context. Under-chunking overwhelms the LLM. *Fix:* Use Late Chunking.
*   **"Blind" Retrieval:** Trusting the Vector DB's top 3 results without a Cross-Encoder Reranker usually results in sub-optimal context and hallucinations.
*   **Agent Infinite Loops:** In LangGraph, failing to set strict exit conditions or max recursion depths for the Critic Agent can lead to endless loops and massive API bills.
*   **VRAM Overflow:** Trying to run the LLM locally on 2GB VRAM instead of offloading to Ollama Cloud. Stick to the Thin-Client architecture.

---

## 4. Evaluation Framework & Metrics

To ensure the system is elite, you must evaluate it systematically using frameworks like **Ragas** or **TruLens**.

### A. Retrieval Accuracy Evaluation
*   **Context Precision:** Measures if the retrieved chunks are relevant to the user query. (High precision = no useless noise in the prompt).
*   **Context Recall:** Measures if *all* the necessary information to answer the question was successfully retrieved from the database.
*   **Hit Rate & MRR (Mean Reciprocal Rank):** Determines how often the most relevant document appears in the Top 1 or Top 3 retrieved chunks before and after Cohere Reranking.

### B. Hallucination & Generation Evaluation
*   **Faithfulness (Hallucination check):** Measures whether the LLM's final answer is strictly derived from the retrieved context. If it uses outside knowledge, it fails.
*   **Answer Relevance:** Measures how directly the LLM answered the specific user prompt, punishing evasive or overly verbose answers.

### C. Latency Testing
*   **Time to First Token (TTFT):** Essential for UI responsiveness. Target: < 1.0 seconds.
*   **Retrieval Latency:** Time taken by `sqlite-vec` + Cohere Rerank. Target: < 500ms.
*   **Total End-to-End Latency:** Target < 3-5 seconds for complex multi-hop queries.

### D. Memory Quality Evaluation
*   **State Retention Accuracy:** Evaluating if LangGraph accurately remembers user preferences across a 10-turn conversation.
*   **Graph Interference:** Ensuring Episodic memory (Graphiti) doesn't inject outdated or contradictory rules into the current Working Memory.

---

## 5. Performance Benchmarks (Targets for 2026 Thin-Client)

| Metric | Target Benchmark |
| :--- | :--- |
| **Local VRAM Usage** | < 1.5 GB (Jina Embeddings only) |
| **Local RAM Usage** | < 4 GB (sqlite-vec + orchestration) |
| **Indexing Speed** | > 50 pages / minute (depending on LlamaParse limits) |
| **Retrieval Speed (Local DB)** | < 50ms per query |
| **Reranking Latency (Cohere)** | < 300ms for 100 chunks |
| **Faithfulness Score** | > 0.95 (Ragas metric) |
| **Context Recall** | > 0.90 (Ragas metric) |
