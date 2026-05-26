# Phase 2 — The Ideal 2026 RAG Architecture
## A Principal Architect's System Design

> [!NOTE]
> This document outlines the definitive production-grade architecture for an elite, single-power-user RAG system in 2026. It discards toy framework abstractions in favor of a decoupled, highly optimized, multi-agent reasoning engine built on state-of-the-art 2026 technologies.

> [!IMPORTANT]
> This phase describes the unconstrained 2026 frontier; see Phase 3's 'Carried forward vs. dropped' for what the CPU-first build actually uses.

---

## A. INGESTION PIPELINE

The 2026 standard abandons the "one-size-fits-all" parser. Ingestion is now a **Tiered Routing Pipeline** where documents are dynamically routed to specialized extractors based on their type, layout complexity, and modality.

### 1. Parsing & Extraction Stack
- **Enterprise & General Documents (PDFs, DOCX, PPTX):** **Docling** (IBM). The default choice. Uses the Granite-Docling-258M VLM for fast, single-pass extraction of text, tables, and markup. Preserves reading order and hierarchical structure.
- **Complex Scientific & Academic Papers:** **MinerU (v2.5)**. Unmatched for dense layouts, multi-column CJK text, and intricate LaTeX math extraction.
- **Codebases:** **Tree-sitter**. AST-based chunking that breaks code at semantic boundaries (functions, classes) rather than token limits, preserving structural integrity.
- **Websites:** **Firecrawl** (managed) or **Crawl4AI** (self-hosted). Converts dynamic sites to clean Markdown, handling JS rendering and anti-bot protections.

### 2. Multimodal Ingestion
- **Images & Diagrams:** Extracted via vision models (e.g., **Qwen3-VL** or **Gemini 2.5/3.0**), which generate dense descriptive summaries and preserve spatial bounding box metadata.
- **Audio:** **Whisper Large v3 Turbo** coupled with **pyannote.audio** for speaker diarization.
- **Video:** Segmented by scene detection algorithms. Keyframes are extracted at scene changes, passed to VLMs for captioning, and temporally aligned with Whisper transcripts.

### 3. OCR Routing
- **Tier 1 (Clean/Standard):** **PaddleOCR** for high-throughput, structured extraction (forms, invoices).
- **Tier 2 (Messy/Complex):** Vision-Language Models (VLMs) used as fallback for handwriting or severely degraded scans.

### 4. Advanced Chunking & Knowledge Extraction
- **Adaptive Chunking:** Evaluates each document dynamically to select the best strategy based on Intrachunk Cohesion (ICC) and Document Contextual Coherence (DCC).
- **Late Chunking:** Powered by **Jina AI (v3)**. The entire document is passed through a long-context embedding model first to capture global semantics, then token embeddings are mean-pooled at logical boundaries. Resolves the "dangling pronoun" problem perfectly.
- **Knowledge Graph Extraction:** LLM-based extraction (e.g., GPT-4o or DeepSeek-R1) pulls Subject-Predicate-Object triples, validated against a predefined ontology, to build explicit relational links between concepts.

---

## B. STORAGE LAYER

The storage layer is disaggregated, utilizing specialized databases for different memory formats.

### 1. Vector DB (Dense + Sparse Semantic Storage)
**Primary Recommendation: Qdrant (Local/Self-hosted) or Weaviate**
- **Qdrant:** Written in Rust, offering sub-5ms p50 latency. Unmatched for filtered search (combining metadata constraints with semantic search).
- **Alternative for Postgres environments:** **pgvector** + **pgvectorscale** (DiskANN). Ideal if the user wants to maintain a unified relational + vector architecture up to ~50M vectors.

### 2. Graph DB (Relational & Relational Memory)
**Primary Recommendation: Neo4j**
- The enterprise standard for GraphRAG. It combines vector similarity and complex graph traversals in a single Cypher query. Enables "In-index filtering" and multi-hop reasoning.

### 3. Unified / Embedded Storage
**Primary Recommendation: LanceDB or SurrealDB**
- **LanceDB:** Embedded, serverless, columnar storage (Lance format) that natively handles both multimodal raw data and vectors. Exceptional for local-first/edge execution.
- **SurrealDB:** A multi-model database that unifies vector, graph, and document storage, eliminating DB fragmentation for agentic systems.

### 4. Cache Layer
**Primary Recommendation: Redis**
- Used for Semantic Caching. Embeddings of incoming queries are matched against cached queries. If similarity is >0.95, it bypasses the LLM entirely, saving ~90% cost and yielding sub-10ms response times.

---

## C. RETRIEVAL LAYER

Retrieval is orchestrated dynamically by the Agent Layer, abandoning naive top-k vector search.

### 1. Hybrid Search (The Baseline)
Combines dense vectors (semantic intent) with sparse BM25 / SPLADE (exact keyword/ID matching). Merged using **Reciprocal Rank Fusion (RRF)**.

### 2. Retrieval Orchestration & Routing
- **Adaptive Routing:** A lightweight classifier determines query complexity.
  - *Simple facts:* Routed to standard Hybrid Search.
  - *Complex reasoning:* Routed to Graph Traversal or Multi-query decomposition.
- **Decomposition:** Planner agents break complex questions ("Compare X's impact on Y to Z") into atomic sub-queries, retrieving concurrently for each.

### 3. Multi-Stage Reranking
1. **Stage 1 (Recall):** Hybrid search retrieves Top 100-200 candidates.
2. **Stage 2 (Precision):** A Cross-Encoder model (e.g., **Qwen3-Reranker-4B** or **Cohere Rerank**) performs full attention between the query and each chunk, trimming to the Top 10-20.

### 4. Small-to-Big Retrieval
Retrieval targets highly specific, granular chunks (for semantic precision). Upon a match, the system injects the surrounding "Parent Context" (e.g., the whole section or page) into the LLM prompt to ensure coherent generation.

---

## D. MEMORY SYSTEM

Memory is treated as a cognitive architecture (CoALA framework), not just a chat history log.

### 1. The Four-Tier Hierarchy
- **L1 (Model Memory):** Parametric weights and optimized KV caching.
- **L2 (Working Memory):** The active context window, utilizing sub-quadratic models (Mamba/Jamba) for extreme context.
- **L3 (Episodic & Semantic System Memory):** The persistent vector/graph stores.
- **L4 (Procedural & Metacognitive Memory):** The agent's stored reflections, learned workflows, and tool-use strategies.

### 2. Temporal & Graph Memory
- Utilizing **Graphiti (Zep)** bi-temporal modeling: Facts track both *Event Time* (when it happened) and *Ingestion Time* (when the AI learned it), allowing for queries like "What did we think the architecture was last month?"

### 3. Sleep-Time Compute & Consolidation
Agents consolidate memory during idle time ("Digital Sleep"):
- **Light Sleep:** Deduplicates recent conversation traces.
- **Deep Sleep:** Summarizes episodic memory into hardened semantic rules and facts.
- **Pruning:** Evicts or compresses stale context using token-level compression algorithms like **LLMLingua-2**.

---

## E. AGENT LAYER

The system is governed by a multi-agent orchestration framework, primarily built on **LangGraph**.

### 1. Orchestration Pattern
**Supervisor with Specialized Workers (Cyclic Graphs):**
- A Supervisor agent receives the query and delegates it to specialized sub-agents (e.g., Research Agent, Coding Agent, Indexing Agent).
- State is managed via shared Pydantic models. **PostgresSaver** is used for persistent checkpointing, enabling time-travel debugging and human-in-the-loop interventions.

### 2. Retrieval as a Tool (MCP)
Retrieval is executed via the **Model Context Protocol (MCP)**. RAG is simply an MCP tool that agents can invoke dynamically alongside web search, calculators, and API calls.

### 3. Corrective RAG (CRAG) & Self-RAG
- **CRAG:** A quality gate *before* generation. If retrieved docs are deemed irrelevant by an evaluator model, the agent discards them and falls back to a web search.
- **Self-RAG:** The generation model uses internal reflection tokens (`[IsSupported]`, `[IsUseful]`) to self-critique its reasoning during output generation.

---

## F. REASONING LAYER

### 1. Speculative RAG
To minimize latency while maximizing reasoning depth:
- A small, fast "Drafter" model generates multiple parallel reasoning chains from different subsets of the retrieved context.
- A massive "Verifier" model (e.g., Llama 4 Scout or Claude 3.5) reviews the drafts, selects the most accurate, and synthesizes the final response.

### 2. Grounded Reasoning & Citation Verification
All outputs must include explicit lineage. The LLM is prompted via Context Distillation (e.g., the RADIO framework) to explicitly state *why* a piece of evidence justifies its claim, appending bounding-box or line-number citations to every fact.

---

## G. MULTIMODAL RAG

### ColPali / Late Interaction Visual Retrieval
For complex visual documents (financial reports, schematics), the 2026 standard bypasses OCR entirely for the retrieval phase.
- Documents are rendered as high-resolution images.
- Models like **ColQwen2.5** slice the image into patches and generate multi-vector embeddings.
- Retrieval is performed via late-interaction patch-matching (MaxSim), perfectly preserving spatial layout, charts, and figures that traditional text extraction destroys.

---

## H. PERFORMANCE OPTIMIZATION

For a power-user operating locally or in a hybrid cloud setup:

### 1. Inference Engines
- **SGLang (RadixAttention):** The optimal engine for RAG. It aggressively caches prompt prefixes (system prompts, retrieved context) across multi-turn agent conversations, achieving massively higher throughput than standard vLLM for this specific workload.

### 2. Local LLMs & Quantization
- **Primary Reasoning Model:** **Llama 4 Scout (109B MoE)**. Running locally with GGUF Q4_K_M quantization, it only activates ~17B parameters per token, fitting comfortably in a 24GB VRAM GPU (e.g., RTX 4090/5090) while offering frontier-level intelligence and a 10M token context window.
- **Routing/Drafting Models:** Phi-4 or Qwen 3.5 8B for lightning-fast classification and planning tasks.

### 3. Async Pipeline & Streaming
- Decoupled architecture: Ingestion, Retrieval, and Generation are entirely asynchronous.
- **Incremental Context Feeding:** Context chunks are streamed into the LLM's context window as soon as they are retrieved and reranked. The LLM begins generation via Server-Sent Events (SSE) before the final database query even finishes, minimizing Time-To-First-Token (TTFT).
