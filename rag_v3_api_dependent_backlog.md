# V3 Backlog — API-Cost-Dependent Features
## Implement After Evaluating Cloud API Costs

> [!NOTE]
> These 13 features are **technically feasible** on 16 GB RAM / 0 VRAM but require either heavy cloud API usage, significant additional local RAM during specific operations, or complex engineering that should come after the core system is proven. Each item includes estimated API cost so you can make informed decisions.

> [!IMPORTANT]
> **Priority guidance:** Items marked 🔴 HIGH have the biggest impact on system quality. Items marked 🟡 MEDIUM add meaningful capability. Items marked 🟢 LOW are nice-to-have refinements.

---

## Feature Backlog

### 1. 🔴 HIGH — MinerU (v2.5) for Scientific Papers
| Aspect | Detail |
|---|---|
| **What it does** | Extracts text from dense multi-column academic papers, CJK text, LaTeX math |
| **Why it's partial** | Loads 3 models simultaneously: layout detection + OCR + table recognition. Peak RAM ~2.5 GB |
| **Hardware constraint** | Must unload bge-small + FlashRank before running MinerU, reload after. Cannot run during query serving. |
| **API cost** | $0 (fully local) |
| **Engineering cost** | Model loading/unloading lifecycle management. ~1 day of work. |
| **RAM during operation** | ~2.5 GB (models) + ~0.5 GB (processing) = ~3 GB peak |
| **When to add** | After core pipeline is stable. Only if you ingest many scientific papers. |

---

### 2. 🟡 MEDIUM — Whisper Audio Transcription (Local CPU)
| Aspect | Detail |
|---|---|
| **What it does** | Transcribes audio/video audio tracks to text |
| **Why it's partial** | Whisper Large v3 Turbo = ~1.5 GB RAM, runs at 0.3-0.5x real-time on CPU. Whisper Small = ~500 MB, runs at 2-5x real-time. |
| **Options** | **Option A (Local):** Use Whisper Small (~500 MB). A 10-min audio takes 2-5 minutes to transcribe. Free. **Option B (Cloud):** Use Groq Whisper API. Near-instant. Free tier: 20 requests/min. **Option C (Cloud):** Use OpenAI Whisper API. $0.006/minute of audio. |
| **Estimated API cost** | Option B: Free (within limits). Option C: ~$0.36 per hour of audio |
| **When to add** | After core pipeline is stable. Only if you ingest audio content. |

---

### 3. 🟢 LOW — pyannote Speaker Diarization
| Aspect | Detail |
|---|---|
| **What it does** | Identifies who is speaking when in multi-speaker audio |
| **Why it's partial** | Requires PyTorch + multiple neural models. ~1.5 GB RAM. Very slow on CPU (15-20 min for 10-min audio). |
| **Options** | **Option A (Cloud):** Use AssemblyAI API — provides transcription + diarization together. $0.37/hour of audio. **Option B:** Skip diarization entirely. Label all speech as single speaker. |
| **Estimated API cost** | $0.37/hour of audio (AssemblyAI) |
| **When to add** | Only if multi-speaker meeting transcription is a priority use case. |

---

### 4. 🔴 HIGH — Knowledge Graph Extraction (LLM-based triples)
| Aspect | Detail |
|---|---|
| **What it does** | Extracts (Subject, Predicate, Object) triples from text during ingestion. Enables multi-hop reasoning by following relationship chains. |
| **Why it's partial** | The extraction itself works fine (send chunk to cloud LLM with extraction prompt). The partial complexity is: building a consistent graph, merging duplicate entities, handling contradictions, and traversing the graph at query time. |
| **Implementation plan** | 1. During ingestion: send each chunk to cloud LLM with prompt "Extract factual relationships as JSON triples." 2. Store triples in a SQLite `knowledge_triples` table. 3. At query time: extract entities from query, look up connected triples via SQL joins. 4. Inject relevant triples into generation context. |
| **Estimated API cost** | ~$0.001-0.003 per chunk ingested (one LLM call per chunk). For 10K chunks: ~$10-30 one-time ingestion cost. |
| **Schema** | `knowledge_triples(triple_id, subject, predicate, object, source_chunk_id, confidence, created_at)` |
| **When to add** | After core retrieval quality is validated. Biggest impact for multi-hop questions. |

---

### 5. 🟡 MEDIUM — Image/Diagram Description via Cloud VLM
| Aspect | Detail |
|---|---|
| **What it does** | Sends extracted images/diagrams from PDFs to a cloud vision model (Gemini Flash, GPT-4o-mini) for textual description. The description is then embedded and stored as a searchable chunk. |
| **Why it's partial** | Requires API calls for every image. Cost depends on image count. |
| **Implementation plan** | 1. During PDF ingestion: extract embedded images using pymupdf4llm. 2. Filter images > 100x100px (skip tiny icons/decorations). 3. Send each image to Gemini 2.5 Flash API: "Describe this diagram/figure in detail." 4. Store description as a chunk with `source_type='image'` and link to parent document. |
| **Estimated API cost** | Gemini 2.5 Flash: ~$0.001 per image. For a 100-page PDF with 20 diagrams: ~$0.02 |
| **When to add** | After PDF text ingestion is stable. High value for technical documents with architecture diagrams, charts, flowcharts. |

---

### 6. 🟢 LOW — SPLADE Learned Sparse Retrieval
| Aspect | Detail |
|---|---|
| **What it does** | Replaces or augments BM25 with a neural sparse retriever. SPLADE learns query expansion ("car" → "vehicle", "automobile") that BM25 cannot do. |
| **Why it's partial** | SPLADE model (~110M params) = ~300 MB RAM. CPU inference ~50-100ms/query. Feasible but adds RAM pressure and inverted index storage. |
| **Tradeoff** | BM25 via FTS5 gives ~85% of SPLADE's quality at 0 MB additional RAM. SPLADE's 15% improvement may not justify the complexity for v1. |
| **Estimated API cost** | $0 (fully local) |
| **RAM cost** | ~300-500 MB permanent |
| **When to add** | Only after measuring BM25 performance with your actual corpus. If BM25 recall is insufficient, evaluate SPLADE. |

---

### 7. 🟢 LOW — ColBERT Late Interaction Retrieval
| Aspect | Detail |
|---|---|
| **What it does** | Token-level multi-vector matching. Bridges cross-encoder accuracy with bi-encoder speed. |
| **Why it's partial** | Model fits (~110 MB INT8). BUT: generates 1 vector per token. A 200-token chunk = 200 vectors. At 100K chunks → 20M vectors × 128 dims × 4 bytes = **~10 GB on disk**. Massive storage and index overhead. |
| **Feasibility** | Only for small corpora (< 10K chunks). At 10K chunks: ~2M vectors = ~1 GB disk. Manageable. |
| **Estimated API cost** | $0 (fully local) |
| **When to add** | Only if FlashRank reranking proves insufficient AND corpus is small. Very niche use case. |

---

### 8. 🔴 HIGH — Graphiti-Style Bi-Temporal Modeling (Simplified)
| Aspect | Detail |
|---|---|
| **What it does** | Tracks when facts were true (`event_time`) vs when the AI learned them (`ingested_at`). Enables queries like "What did I know about X last month?" |
| **Why it's partial** | Full Graphiti requires Neo4j (❌). But the **core concept** — two timestamp columns — is trivially implementable in SQLite. The partial complexity is building temporal query logic. |
| **Implementation plan** | 1. `event_time` and `ingested_at` columns already in V3 chunk schema. 2. Add temporal query modifiers: "as of [date]" → filter `ingested_at <= date`. 3. Add freshness weighting: `freshness_boost = 1 / (days_since_ingestion + 1)`. Multiply with RRF scores. |
| **Estimated API cost** | $0 (fully local, just SQL) |
| **Engineering cost** | ~4 hours for temporal filtering. ~1 day for freshness-weighted ranking. |
| **When to add** | After core retrieval works. High value for evolving knowledge bases. |

---

### 9. 🟡 MEDIUM — Sleep-Time Compute (Digital Sleep)
| Aspect | Detail |
|---|---|
| **What it does** | Background consolidation during idle time: deduplication, summarization, pruning. |
| **Why it's partial** | **Light Sleep** (deduplication): Fully local. Find near-duplicate chunks via embedding cosine similarity > 0.95, merge them. ✅ FREE. **Deep Sleep** (summarization): Send episodic memory batches to cloud LLM: "Consolidate these conversation summaries into key facts." API cost. **Pruning**: Identify stale/low-value chunks by age + access frequency, mark as inactive. ✅ FREE. |
| **Implementation plan** | Run as a nightly cron job (or on-demand script): 1. Light sleep: O(n²) pairwise cosine on recent chunks → merge duplicates. 2. Deep sleep: batch 20 conversation summaries → 1 cloud LLM call → extract durable facts. 3. Pruning: DELETE FROM chunks WHERE ingested_at < 6 months AND access_count = 0. |
| **Estimated API cost** | Deep sleep: ~$0.01 per nightly run (one LLM call with ~2000 tokens) |
| **When to add** | After episodic memory is working and accumulating data (1-2 weeks of daily use). |

---

### 10. 🟢 LOW — A-MEM Zettelkasten Memory (Simplified)
| Aspect | Detail |
|---|---|
| **What it does** | Each interaction generates a structured "note" with metadata tags. Notes are linked by causal/thematic/logical connections. Memory evolves when new info conflicts with old notes. |
| **Why it's partial** | Storing notes is trivial (SQLite table). Generating quality tags and links requires LLM calls per interaction. Memory evolution (conflict detection + resolution) requires periodic LLM review. |
| **Simplified version** | 1. After each conversation, store a "note" with auto-generated tags (extracted keywords). 2. Link notes by embedding similarity (top 3 most similar existing notes). 3. Skip conflict detection for v1. |
| **Estimated API cost** | ~$0.002 per conversation (one LLM call for tag generation) |
| **When to add** | After basic episodic memory is working for 2+ weeks. |

---

### 11. 🟡 MEDIUM — Video Processing Pipeline
| Aspect | Detail |
|---|---|
| **What it does** | Scene detection → keyframe extraction → VLM captioning → audio transcription → temporal alignment |
| **Why it's partial** | **Scene detection** (PySceneDetect): ✅ Local, ~200 MB, CPU-friendly. **Keyframe extraction** (OpenCV): ✅ Local, trivial. **VLM captioning** (cloud API): ~$0.001 per keyframe image. **Audio transcription**: See Whisper item above. **Temporal alignment**: Pure Python logic. ✅ Free. |
| **Estimated API cost** | For a 1-hour video with 30 scene changes: ~$0.03 (VLM captions) + $0.36 (audio transcription if using OpenAI Whisper API) = ~$0.39 |
| **When to add** | Only if video ingestion is a real use case. |

---

### 12. 🟡 MEDIUM — Context Distillation (Prompt-Based)
| Aspect | Detail |
|---|---|
| **What it does** | Before generation, asks the cloud LLM to extract explicit reasoning paths from retrieved context. "Why is this document relevant to the query?" The rationale is then used to improve generation quality. |
| **Why it's partial** | Adds one additional cloud LLM call in the pipeline (after CRAG gate, before generation). Increases per-query cost and latency by ~30%. |
| **Estimated API cost** | ~$0.003-0.005 per query (one LLM call with ~1000 tokens) |
| **When to add** | After measuring faithfulness scores. If faithfulness < 0.90, add context distillation to improve grounding. |

---

### 13. 🟡 MEDIUM — Procedural Memory Learning (LLM-Generated Rules)
| Aspect | Detail |
|---|---|
| **What it does** | Instead of manually writing procedural rules, the LLM reflects on past interactions and generates new rules automatically. "User frequently asks for Python code → add rule: always provide Python examples." |
| **Why it's partial** | Requires periodic LLM review of conversation logs. Risk of generating incorrect or unhelpful rules (memory corruption). |
| **Implementation plan** | 1. Weekly cron: send last 50 conversation summaries to cloud LLM. 2. Prompt: "Based on these interactions, suggest 3 new procedural rules for improving responses." 3. Rules are proposed but NOT auto-applied — user must approve. |
| **Estimated API cost** | ~$0.01-0.02 per weekly run (one LLM call with ~3000 tokens) |
| **When to add** | After 4+ weeks of daily use with enough conversation history to learn from. |

---

## Cost Summary

### One-Time Ingestion Costs (Per 10K Chunks)

| Feature | Cost |
|---|---|
| Knowledge Graph Extraction | ~$10-30 |
| Image/Diagram Description | ~$0.20 (for ~200 images) |
| Audio Transcription (OpenAI) | ~$0.36/hour of audio |
| **Total (typical corpus)** | **~$15-35** |

### Per-Query Costs (if ALL partial features enabled)

| Feature | Cost Per Query |
|---|---|
| Context Distillation | ~$0.004 |
| A-MEM note generation | ~$0.002 |
| **Additional per-query cost** | **~$0.006** |
| (Core V3 already includes: CRAG, decomposition, generation, faithfulness) | ~$0.02-0.05 |
| **Total per-query with all features** | **~$0.03-0.06** |

### Periodic Background Costs

| Feature | Frequency | Cost |
|---|---|---|
| Sleep-Time Compute (Deep Sleep) | Nightly | ~$0.01/run |
| Procedural Memory Learning | Weekly | ~$0.02/run |
| **Monthly background cost** | — | **~$0.38** |

---

## Implementation Priority Order

After core V3 is stable and evaluated:

| Sprint | Feature | Impact | Effort |
|---|---|---|---|
| **Backlog Sprint 1** | Bi-Temporal Modeling (#8) | 🔴 HIGH | 1 day |
| **Backlog Sprint 1** | Knowledge Graph Extraction (#4) | 🔴 HIGH | 2-3 days |
| **Backlog Sprint 1** | Image/Diagram Description (#5) | 🟡 MEDIUM | 4 hours |
| **Backlog Sprint 2** | Sleep-Time Compute (#9) | 🟡 MEDIUM | 1 day |
| **Backlog Sprint 2** | Context Distillation (#12) | 🟡 MEDIUM | 4 hours |
| **Backlog Sprint 2** | Whisper Audio (#2) | 🟡 MEDIUM | 4 hours |
| **Backlog Sprint 3** | MinerU Scientific Parsing (#1) | 🔴 HIGH | 1 day |
| **Backlog Sprint 3** | Video Processing (#11) | 🟡 MEDIUM | 1-2 days |
| **Backlog Sprint 3** | Procedural Memory Learning (#13) | 🟡 MEDIUM | 4 hours |
| **Later** | A-MEM Simplified (#10) | 🟢 LOW | 1-2 days |
| **Later** | SPLADE (#6) | 🟢 LOW | 1-2 days |
| **Later** | ColBERT (#7) | 🟢 LOW | 2-3 days |
| **Later** | pyannote Diarization (#3) | 🟢 LOW | 4 hours |
