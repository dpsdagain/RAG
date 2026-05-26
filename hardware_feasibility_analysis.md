# Hardware Feasibility Analysis
## Can Phase 1 & Phase 2 Be Implemented on 16 GB RAM / 500 GB SSD / 0 VRAM?

> [!NOTE]
> **Hardware budget:** After Windows OS + browser (~4.5 GB), you have roughly **11 GB of usable RAM**. After Python runtime overhead (~0.5 GB), you have **~10.5 GB** for your entire RAG system. Every component below is evaluated against this real budget.

---

## Summary Scorecard

| Category | Total Components | ✅ YES | ⚠️ PARTIAL | ❌ NO |
|---|---|---|---|---|
| Ingestion Pipeline | 10 | 5 | 3 | 2 |
| Storage Layer | 7 | 4 | 1 | 2 |
| Retrieval Layer | 8 | 6 | 2 | 0 |
| Memory System | 9 | 4 | 3 | 2 |
| Agent Layer | 5 | 2 | 1 | 2 |
| Reasoning Layer | 3 | 1 | 1 | 1 |
| Multimodal RAG | 2 | 0 | 1 | 1 |
| Performance Optimization | 4 | 2 | 1 | 1 |
| **TOTAL** | **48** | **24 (50%)** | **13 (27%)** | **11 (23%)** |

**Bottom line: You can implement roughly 50% fully and another 27% partially. Only 23% is truly impossible on this hardware.**

---

# A. INGESTION PIPELINE (Phase 2, Section A)

### 1. Docling (IBM) — General Document Parsing
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~300-500 MB during parsing (loads Granite-Docling-258M model) |
| **Why it works** | The Granite-Docling-258M VLM is only 258M parameters. At FP16 that's ~516 MB. On CPU with ONNX quantization (INT8), it drops to ~260 MB. Fits comfortably. |
| **Caveat** | Parsing is slow on CPU (~5-15 seconds per page vs <1s on GPU). Fine for batch ingestion, not real-time. |

### 2. MinerU (v2.5) — Scientific/Academic Papers
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~1.5-2.5 GB during parsing (loads multiple detection models) |
| **Why it's partial** | MinerU loads a layout detection model + OCR model + table recognition model simultaneously. On CPU, each model runs sequentially but all stay resident. At 2.5 GB, it's feasible ONLY if you unload other models first and don't run it during query time. |
| **Recommendation** | Use only for batch ingestion. Unload embedding model while MinerU runs. Reload after. |

### 3. Tree-sitter — Code Parsing
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~10-30 MB |
| **Why it works** | Tree-sitter is a pure C parser with tiny grammars (~100 KB each). Near-zero resource usage. This is trivially feasible. |

### 4. Firecrawl / Crawl4AI — Website Crawling
| Verdict | ✅ YES (Crawl4AI) |
|---|---|
| **RAM cost** | ~200-400 MB (headless browser for JS rendering) |
| **Why it works** | Crawl4AI runs a lightweight headless Chromium. 400 MB is fine during batch ingestion. Firecrawl managed API is even lighter (just HTTP calls). |
| **Caveat** | Don't crawl while serving queries — the headless browser will eat CPU/RAM. |

### 5. Image/Diagram Extraction via VLMs (Qwen3-VL / Gemini)
| Verdict | ❌ NO (locally) / ✅ YES (via API) |
|---|---|
| **RAM cost** | Qwen3-VL smallest variant (3B) needs ~3-6 GB RAM on CPU. Inference would take 30-60 seconds per image. |
| **Why it fails locally** | VLM inference on CPU is agonizingly slow and would consume your entire RAM budget. |
| **Alternative** | Use **Gemini 2.5 Flash API** or **GPT-4o-mini API** for image description. Costs ~$0.001 per image. Trivial cost, fast, no local RAM. **This works.** |

### 6. Whisper Large v3 Turbo — Audio Transcription
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~1.5-2 GB (Whisper Large v3 Turbo, INT8 quantized, CPU) |
| **Why it's partial** | Whisper Large v3 Turbo at INT8 fits in RAM (~1.5 GB). BUT: inference speed on CPU is ~0.3-0.5x real-time. A 10-minute audio file takes 20-30 minutes to transcribe. |
| **Alternative** | Use **Whisper base or small** (~150 MB / ~500 MB) for decent quality at 2-5x real-time on CPU. OR use **Groq Whisper API** (free tier, near-instant). |
| **Recommendation** | Use Whisper small (500 MB) locally for privacy-sensitive audio. Use Groq API for speed. |

### 7. pyannote.audio — Speaker Diarization
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~1-1.5 GB (loads multiple PyTorch models) |
| **Why it fails** | pyannote requires PyTorch + multiple neural network models running simultaneously. On CPU, diarization of a 10-minute file takes 15-20 minutes and consumes ~1.5 GB RAM. Combined with Whisper, that's 3+ GB just for audio processing. |
| **Alternative** | Skip diarization for v1. Or use a cloud API (e.g., AssemblyAI) that provides transcription + diarization together. |

### 8. PaddleOCR — Clean Document OCR
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~300-500 MB |
| **Why it works** | PaddleOCR's lightweight models (detection + recognition + angle classifier) total ~100 MB on disk, ~400 MB resident. CPU inference is fast for clean documents (1-3 seconds per page). |

### 9. VLM-based OCR Fallback — Messy/Handwriting OCR
| Verdict | ✅ YES (via API) |
|---|---|
| **Why it works** | Same as #5 — use Gemini Flash or GPT-4o-mini API. Send the page image, get text back. |

### 10. Late Chunking (Jina v3)
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~2-4 GB (Jina v3 is a ~560M param model with 8K context) |
| **Why it fails** | Late chunking requires passing the **entire document** through a long-context embedding model to get token-level embeddings, then pooling them into chunks. On CPU, embedding a 20-page document through a 560M model takes minutes and consumes several GB. |
| **Alternative** | **Sentence-level semantic chunking with bge-small** achieves 80% of the benefit: embed every sentence (~5ms each), compute inter-sentence cosine similarity, split where similarity drops. This runs in milliseconds on CPU. |

### 11. Adaptive Chunking (ICC/DCC metrics)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | Negligible (uses existing embedding model) |
| **Why it works** | ICC and DCC are computed using the same bge-small embeddings you already have. The chunking algorithm itself is pure Python logic — cosine similarity calculations between consecutive chunk embeddings. Zero additional model cost. |

### 12. Knowledge Graph Extraction (LLM-based triples)
| Verdict | ⚠️ PARTIAL (via cloud LLM) |
|---|---|
| **RAM cost** | 0 locally (cloud LLM call) |
| **Why it's partial** | You CAN extract Subject-Predicate-Object triples by sending chunks to your cloud LLM with a structured extraction prompt. The LLM call works. BUT: validating against a predefined ontology, maintaining graph consistency, and running graph traversal queries adds significant engineering complexity. |
| **Recommendation** | Extract triples via cloud LLM during ingestion. Store in a simple SQLite adjacency table. Query with basic SQL joins. Skip Neo4j, skip ontology validation for v1. |

---

# B. STORAGE LAYER (Phase 2, Section B)

### 1. Qdrant (Local/Self-hosted)
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~500 MB base + ~1-2 GB for 100K vectors with payload |
| **Why it's partial** | Qdrant is a Rust binary that runs as a background server. At 100K vectors it's feasible (~1.5 GB total). At 500K+ vectors it starts consuming 3-4 GB. The server overhead is wasteful for a single-user system. |
| **Recommendation** | sqlite-vec is the correct choice for this hardware. Qdrant adds operational overhead (background process, port management, crashes) for marginal benefit at your scale. Only consider Qdrant if you exceed 200K vectors and need HNSW. |

### 2. Neo4j — Graph Database
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~2-4 GB minimum (JVM-based, heap allocation) |
| **Why it fails** | Neo4j runs on the JVM. The minimum recommended heap is 2 GB. With OS, Python, embedding model, and SQLite already consuming ~6-8 GB, adding a JVM process pushes you into swap territory. |
| **Alternative** | **NetworkX** (in-memory Python graph library): ~50-100 MB for a 50K-node graph. Store graph edges in a SQLite table, load into NetworkX on startup. |

### 3. LanceDB — Embedded Columnar Storage
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~200-500 MB (embedded, mmap-based) |
| **Why it works** | LanceDB is embedded (no server), uses memory-mapped files (doesn't load entire index into RAM), and supports vector search natively. It's actually a viable *alternative* to sqlite-vec with better ANN support. |
| **Consideration** | LanceDB could replace sqlite-vec if you need ANN indexing beyond 200K vectors. Worth evaluating as a future upgrade path. |

### 4. SurrealDB — Multi-model Database
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~1-2 GB base (Rust binary, but complex multi-model engine) |
| **Why it fails** | SurrealDB tries to be a vector DB + graph DB + document DB + relational DB simultaneously. The memory overhead of maintaining all these query engines is excessive for your constraint. You get worse performance at everything than dedicated solutions. |

### 5. Redis — Semantic Cache
| Verdict | ✅ YES (but unnecessary) |
|---|---|
| **RAM cost** | ~100-300 MB for semantic caching |
| **Why it works** | Redis is lightweight. A cache of 1000 query-response pairs with 384-dim embeddings is ~50 MB. |
| **But** | For a single-user system, an **in-process Python dictionary** with LRU eviction achieves the same result with zero infrastructure. Redis adds a background server process. Use a `dict` + `functools.lru_cache` instead. |

### 6. sqlite-vec — Local Vector DB
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~200 MB - 1 GB (mmap scales with corpus size) |
| **Why it works** | Already validated in Phase 7. Embedded, no server, mmap-based. Perfect for this hardware up to ~200K vectors. |

### 7. SQLite FTS5 — BM25 Search
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~50-100 MB for 100K documents |
| **Why it works** | FTS5 is a built-in SQLite extension. Near-zero overhead. This should be in the MVP — there is absolutely no hardware reason to defer it. |

---

# C. RETRIEVAL LAYER (Phase 2, Section C)

### 1. Hybrid Search (Dense + Sparse BM25 + RRF)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | 0 additional (uses existing sqlite-vec + FTS5) |
| **Why it works** | RRF is a pure Python rank-merging algorithm (~20 lines of code). Dense search and BM25 both run against SQLite. Zero additional RAM. |

### 2. Adaptive Query Routing (Complexity Classifier)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | 0-100 MB depending on approach |
| **Options** | **Option A:** Send query to cloud LLM with "classify this as simple/complex" — costs 1 API call (~50 tokens). **Option B:** Local heuristic (query length, question word detection, keyword presence) — zero RAM. **Option C:** Tiny local classifier (distilbert-base, ~250 MB) — feasible but overkill. |
| **Recommendation** | Use a simple local heuristic for v1. If insufficient, upgrade to a cloud LLM classification call. |

### 3. Query Decomposition (Planner Agents)
| Verdict | ✅ YES (via cloud LLM) |
|---|---|
| **RAM cost** | 0 locally |
| **Why it works** | Send the query to your cloud LLM: "Break this into 1-3 focused sub-queries." One API call, ~100 tokens. No local compute required. |

### 4. Multi-Stage Reranking (Cross-Encoder)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~200-500 MB for FlashRank |
| **Details** | **FlashRank** (nano cross-encoder): ~200 MB, reranks 100 chunks in <1 second on CPU. ✅ **GTE-Reranker-ModernBERT-Base** (149M params): ~300 MB, higher quality. ✅ **Qwen3-Reranker-4B**: ~4 GB quantized — ❌ too large. **Cohere Rerank API**: 0 local RAM — ✅ as optional cloud tier. |

### 5. Small-to-Big Retrieval (Parent Context)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | Negligible (SQLite query to fetch parent chunk) |
| **Why it works** | This is purely a storage/retrieval pattern. Store `parent_chunk_id` in the chunks table. On retrieval, JOIN to get parent content. Zero additional model cost. |

### 6. Semantic Caching
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~10-50 MB (in-process cache) |
| **Why it works** | Cache recent query embeddings + responses in a Python dict. On new query, compute cosine similarity against cached query embeddings. If > 0.95, return cached response. Uses existing bge-small model. |

### 7. ColBERT / Late Interaction Retrieval
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~500 MB - 1.5 GB (model) + massive storage (multi-vector per document) |
| **Why it's partial** | ColBERTv2 model (~110M params) fits in RAM (~220 MB FP16, ~110 MB INT8). BUT: ColBERT generates **one vector per token**. A 200-token chunk generates 200 vectors. At 100K chunks, that's 20M vectors × 128 dims × 4 bytes = **~10 GB on disk** and significant index RAM. |
| **Verdict** | Feasible only for small corpora (<10K chunks). Not practical at scale on this hardware. |

### 8. SPLADE / Learned Sparse Retrieval
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~300-500 MB (SPLADE model) |
| **Why it's partial** | SPLADE models are ~110M params (BERT-sized). Fits in RAM easily. CPU inference is ~50-100ms per query. However, SPLADE generates sparse vectors that need inverted index storage. At 100K docs, this is manageable (~200 MB). At 500K+, the inverted index becomes large. |
| **Recommendation** | Worth considering as a BM25 replacement. SPLADE provides learned query expansion ("car" → "vehicle", "automobile") which BM25 cannot. But stick with FTS5/BM25 for v1 — it's simpler and nearly as good. |

---

# D. MEMORY SYSTEM (Phase 2, Section D)

### 1. L1 — Model Memory (KV Cache)
| Verdict | ✅ YES (implicitly — handled by cloud LLM) |
|---|---|
| **Why** | Your cloud LLM manages its own KV cache. No local action needed. |

### 2. L2 — Working Memory (Context Window)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~10-50 MB (JSON state in SQLite) |
| **Why it works** | Storing the last N conversation turns in a SQLite table is trivially feasible. |

### 3. L3 — Episodic Memory (Past Experiences)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | ~50-100 MB (conversation embeddings in sqlite-vec) |
| **Implementation** | After each conversation, generate a summary, embed it with bge-small, store in a dedicated sqlite-vec table. On new queries, retrieve relevant past conversations. This is just another vector search — uses existing infrastructure. |
| **Why it was wrongly dropped** | Phase 7 dropped this entirely, but it requires zero new models, zero new infrastructure, and ~50 lines of code. |

### 4. L3 — Semantic Memory (Facts/Knowledge)
| Verdict | ✅ YES (this IS your RAG knowledge base) |
|---|---|
| **Why** | Your sqlite-vec chunk store IS semantic memory. Already implemented. |

### 5. L4 — Procedural Memory (Tool Strategies/Workflows)
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~10 MB (SQLite table) |
| **Why it's partial** | You CAN store learned procedures/workflows in a SQLite table (e.g., "when user asks about code, always include file paths"). The storage is trivial. The hard part is **learning** new procedures — that requires the LLM to reflect on past interactions and generate procedure rules. This works via cloud LLM but adds complexity. |
| **Recommendation** | Start with manually-authored procedure rules (essentially prompt templates). Add LLM-generated procedures in v2. |

### 6. Graphiti Bi-Temporal Modeling
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~200-500 MB (Graphiti uses Neo4j OR can use in-memory graphs) |
| **Why it's partial** | Full Graphiti requires Neo4j (❌ on this hardware). BUT: the **concept** of bi-temporal modeling (tracking event time + ingestion time) can be implemented with two timestamp columns on every fact row in SQLite. You lose Graphiti's graph traversal and community detection, but you gain temporal reasoning. |
| **Recommendation** | Add `event_time` and `ingestion_time` columns to your chunks/facts tables. Query with SQL temporal predicates. Skip the full Graphiti framework. |

### 7. Sleep-Time Compute / Digital Sleep
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | 0 additional (uses cloud LLM during idle time) |
| **Why it's partial** | **Light Sleep** (deduplication): Purely local — find duplicate/near-duplicate chunks via embedding similarity, merge them. ✅ Works. **Deep Sleep** (summarization): Send episodic memory batches to cloud LLM for consolidation into semantic rules. ✅ Works, but costs API tokens. **Pruning** (LLMLingua-2 compression): ❌ LLMLingua-2 needs a local model (~500 MB). Alternative: use cloud LLM to summarize and compress. |
| **Recommendation** | Implement light sleep (dedup) locally. Run deep sleep (summarization) as a nightly cloud LLM batch job. Skip LLMLingua-2. |

### 8. A-MEM (Zettelkasten Memory)
| Verdict | ❌ NO (as described) / ✅ YES (simplified) |
|---|---|
| **Why it fails as described** | Full A-MEM requires autonomous note construction, causal/thematic link generation, and memory evolution — all requiring continuous LLM inference. On cloud APIs, this becomes expensive for every interaction. |
| **Simplified version that works** | Store each interaction as a "note" in SQLite with metadata tags. Manually or periodically link related notes via embedding similarity. This gives you 60% of A-MEM's value with 10% of the complexity. |

### 9. LLMLingua-2 Token Compression
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~500 MB - 1 GB (small language model for token scoring) |
| **Why it fails** | LLMLingua-2 uses a small BERT-like model to score every token by perplexity, then strips low-information tokens. The model fits in RAM, but CPU inference on a 10K-token context takes 5-10 seconds — unacceptable for real-time query processing. |
| **Alternative** | Use **extractive summarization** via cloud LLM: "Summarize this context, keeping only facts relevant to the query." One API call. Achieves similar compression without local model cost. |

---

# E. AGENT LAYER (Phase 2, Section E)

### 1. Multi-Agent Supervisor + Workers (LangGraph)
| Verdict | ❌ NO |
|---|---|
| **Why it fails** | Not a RAM issue — it's an **architecture failure risk**. Multi-agent loops with cyclic graphs cause infinite API calls, unpredictable latency, and massive token costs. Phase 6 correctly identified this as a trap. The linear pipeline is the right choice. |

### 2. RAG as MCP Tool
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | Negligible (MCP is a protocol, not a model) |
| **Why it works** | MCP is just a JSON-RPC protocol for tool invocation. Your RAG pipeline becomes an MCP server that the cloud LLM calls. Zero additional RAM. This is worth implementing — it makes your RAG system interoperable with any MCP-compatible agent. |

### 3. CRAG (Corrective RAG)
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | 0 locally (cloud LLM call) |
| **Implementation** | After retrieval, send retrieved chunks + query to cloud LLM: "Are these documents relevant and sufficient to answer this query? Rate: CORRECT / AMBIGUOUS / INCORRECT." One API call, ~500 tokens. If INCORRECT, fall back to web search or abstain. |
| **Why this was wrongly dropped** | CRAG is the single highest-impact hallucination prevention mechanism. It costs one cloud LLM call. There is zero hardware reason to exclude it. |

### 4. Self-RAG (Reflection Tokens)
| Verdict | ❌ NO |
|---|---|
| **Why it fails** | Self-RAG requires a fine-tuned model that generates special reflection tokens (`[IsSupported]`, `[IsUseful]`). You can't fine-tune models on this hardware, and cloud APIs don't support custom reflection tokens. |
| **Alternative** | Achieve 80% of Self-RAG's benefit with a **post-generation verification prompt**: Send the generated response + source chunks to the cloud LLM: "Is every claim in this response supported by the provided evidence? Flag unsupported claims." One extra API call. |

### 5. PostgresSaver for Checkpointing
| Verdict | ✅ YES (using SQLite instead) |
|---|---|
| **Why** | SQLite checkpointing replaces Postgres. Already in Phase 7. Works fine. |

---

# F. REASONING LAYER (Phase 2, Section F)

### 1. Speculative RAG (Drafter + Verifier)
| Verdict | ❌ NO |
|---|---|
| **Why it fails** | Requires a fast local "drafter" model generating multiple parallel reasoning chains. Any local model on CPU is too slow. You'd need to run the drafter on cloud too, which defeats the latency benefit. |
| **Alternative** | Use **single-pass generation with CRAG** instead. The quality gate before generation achieves most of the accuracy benefit without the complexity. |

### 2. Grounded Reasoning / RADIO Citation Framework
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | 0 locally |
| **Why it works** | This is a **prompting technique**, not a model. Instruct your cloud LLM: "For every claim in your response, cite the specific chunk ID and quote the supporting text." Zero RAM, zero additional API calls — it's part of your generation prompt. |
| **Why this was wrongly dropped** | There is literally no cost to implementing citation enforcement. It's a prompt modification. |

### 3. Context Distillation
| Verdict | ⚠️ PARTIAL |
|---|---|
| **Why it's partial** | Full context distillation (training a student model on the RAG pipeline outputs) requires local training — ❌. But **prompt-based distillation** (asking the cloud LLM to extract reasoning rationales from retrieved context before generating the final answer) works perfectly. |

---

# G. MULTIMODAL RAG (Phase 2, Section G)

### 1. ColPali / ColQwen2.5 — Visual Document Retrieval
| Verdict | ❌ NO |
|---|---|
| **RAM cost** | ~2-4 GB (ColQwen2.5 model) + massive multi-vector storage |
| **Why it fails** | ColQwen2.5 generates patch-level embeddings for document images. On CPU, encoding a single page takes 10-30 seconds. Storage requirements for multi-vector representations are enormous (hundreds of vectors per page). |
| **Alternative** | For visual documents: render pages as images → send to cloud VLM API (Gemini Flash) for description → embed the description text with bge-small. You lose spatial layout matching but gain feasibility. |

### 2. Video Processing (Scene Detection + VLM Captioning)
| Verdict | ⚠️ PARTIAL |
|---|---|
| **RAM cost** | ~200-500 MB for scene detection (PySceneDetect) |
| **Why it's partial** | **Scene detection** (PySceneDetect): ✅ Pure OpenCV, lightweight, CPU-friendly. **Keyframe extraction**: ✅ OpenCV `VideoCapture`, trivial. **VLM captioning of keyframes**: ✅ via cloud API (send images to Gemini Flash). **Whisper transcription of audio track**: ⚠️ slow on CPU (see audio section above). |
| **Recommendation** | Feasible as a batch ingestion pipeline. Not real-time. Process videos overnight. |

---

# H. PERFORMANCE OPTIMIZATION (Phase 2, Section H)

### 1. SGLang (RadixAttention)
| Verdict | ❌ NO |
|---|---|
| **Why** | SGLang is a GPU inference engine. It requires VRAM for model hosting. Completely irrelevant for 0-VRAM. |

### 2. Local LLMs (Llama 4 Scout 109B MoE)
| Verdict | ❌ NO |
|---|---|
| **Why** | Even with Q4_K_M quantization, Llama 4 Scout requires ~24 GB VRAM. Phase 2 explicitly states "RTX 4090/5090." Impossible on your hardware. |
| **What you CAN run locally** | **Phi-3-mini (3.8B, Q4)**: ~2.5 GB RAM on CPU. Speed: ~3-5 tokens/sec. Usable for classification/routing only, not generation. **Qwen2.5-0.5B (Q8)**: ~600 MB RAM. Speed: ~10 tokens/sec. Useful for query classification. |

### 3. Async Pipeline & Streaming
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | Negligible |
| **Why it works** | Python asyncio + SSE streaming is purely architectural. No model cost. Already in Phase 7. |

### 4. Incremental Context Feeding
| Verdict | ✅ YES |
|---|---|
| **RAM cost** | Negligible |
| **Why it works** | Stream retrieved chunks to the cloud LLM as they arrive. The LLM begins generating before all chunks are reranked. Reduces TTFT by 200-500ms. Pure architectural optimization. |

---

# FINAL SUMMARY: What Your System CAN Be

## The Realistic Maximum Architecture on 16 GB / 0 VRAM

```
┌─────────────────────────────────────────────────────┐
│              INGESTION (Batch, Background)           │
├─────────────────────────────────────────────────────┤
│ Local: Docling (258M, CPU) → Text PDFs              │
│ Local: PaddleOCR (400 MB) → Clean scanned docs      │
│ Local: Tree-sitter → Code                            │
│ Local: Crawl4AI → Websites                           │
│ Cloud: Gemini Flash API → Images, diagrams, messy OCR│
│ Cloud: LlamaParse API → Complex visual PDFs          │
│ Local: PySceneDetect → Video scene detection         │
│ Local/Cloud: Whisper small/Groq → Audio transcription│
├─────────────────────────────────────────────────────┤
│           CHUNKING & KNOWLEDGE EXTRACTION            │
├─────────────────────────────────────────────────────┤
│ Local: Sentence-level semantic chunking (bge-small)  │
│ Local: Adaptive chunking with ICC/DCC metrics        │
│ Cloud: LLM-based triple extraction → SQLite graph    │
│ Local: Bi-temporal timestamps on all chunks          │
├─────────────────────────────────────────────────────┤
│              STORAGE (All Local)                     │
├─────────────────────────────────────────────────────┤
│ sqlite-vec: Dense vectors (384-dim, bge-small)       │
│ SQLite FTS5: BM25 sparse index                       │
│ SQLite: Document store, metadata, conversation logs  │
│ SQLite: Knowledge graph (adjacency table)            │
│ Python dict: In-process semantic cache               │
├─────────────────────────────────────────────────────┤
│              RETRIEVAL (All Local)                    │
├─────────────────────────────────────────────────────┤
│ Hybrid: Dense (sqlite-vec) + Sparse (FTS5) + RRF     │
│ Reranking: FlashRank (CPU, 200 MB)                   │
│ Small-to-Big: Parent context injection               │
│ Semantic cache check (in-process dict)               │
├─────────────────────────────────────────────────────┤
│              REASONING (Cloud LLM)                   │
├─────────────────────────────────────────────────────┤
│ Cloud: CRAG quality gate (1 LLM call)                │
│ Cloud: Query decomposition (1 LLM call)              │
│ Cloud: Generation with RADIO citation prompt         │
│ Cloud: Post-generation faithfulness check (1 LLM call)│
│ Fallback: Raw evidence surfacing on API failure      │
├─────────────────────────────────────────────────────┤
│              MEMORY (Local + Cloud)                  │
├─────────────────────────────────────────────────────┤
│ L2: Working memory (SQLite, last N turns)            │
│ L3: Episodic memory (conversation embeddings)        │
│ L3: Semantic memory (RAG knowledge base)             │
│ L4: Procedural memory (manual rules, SQLite)         │
│ Local: Light sleep deduplication (cron job)           │
│ Cloud: Deep sleep summarization (nightly LLM batch)  │
├─────────────────────────────────────────────────────┤
│              RAM BUDGET ESTIMATE                     │
├─────────────────────────────────────────────────────┤
│ OS + Browser:           ~4.5 GB                      │
│ Python + ONNX:          ~1.0 GB                      │
│ bge-small model:        ~0.15 GB                     │
│ FlashRank model:        ~0.3 GB                      │
│ sqlite-vec mmap:        ~0.5-1.0 GB                  │
│ FTS5 + SQLite:          ~0.1 GB                      │
│ App + cache + buffers:  ~0.5 GB                      │
│ ────────────────────────────────────                 │
│ TOTAL:                  ~7.0-7.5 GB                  │
│ HEADROOM:               ~8.5 GB free                 │
└─────────────────────────────────────────────────────┘
```

## The Honest Numbers

| Metric | Phase 2 Ideal | What You Can Actually Build |
|---|---|---|
| Components from Phase 1/2 | 48 | **37** (24 full + 13 partial) |
| Retrieval quality (vs ideal) | 100% | **~85%** (hybrid + rerank, no ColBERT, no graph traversal) |
| Memory sophistication (vs ideal) | 100% | **~45%** (episodic + working + procedural, no Graphiti, no A-MEM) |
| Hallucination prevention | Full CRAG + Self-RAG | **~70%** (CRAG + citation + post-gen check, no Self-RAG) |
| Multimodal capability | Full local VLM | **~40%** (cloud APIs for images/audio, no ColPali, limited video) |
| Multi-hop reasoning | GraphRAG (87%) | **~50%** (query decomposition + lightweight SQLite graph, no Neo4j) |
| Ingestion speed | GPU-accelerated | **~15-30%** of GPU speed (CPU parsing is slower, but runs in background) |

> [!IMPORTANT]
> **The key insight:** Your hardware doesn't prevent you from building a sophisticated system. It prevents you from running large **models** locally. Every technique that is purely **algorithmic** (chunking strategies, RRF fusion, small-to-big retrieval, temporal indexing, citation prompting) works perfectly. Every technique that requires a **large model** must be routed to a cloud API — and most of them can be.
>
> **The real constraint is not RAM — it's API cost.** Each query with CRAG + decomposition + generation + post-gen check = 4 cloud LLM calls. At ~$0.01-0.03 per query, this is manageable for a single power user but not negligible.
