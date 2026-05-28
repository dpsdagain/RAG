"""Golden-set of (query, expected substrings) pairs against the V3 doc.

Each entry has:
- `id`: unique short slug (used as the pytest parametrize id)
- `query`: what the user types
- `must_retrieve`: lowercase substrings that should appear in at least one
   chunk among the top-K retrieval results (semantic relevance check)
- `must_generate`: lowercase substrings the LLM answer should contain
   (end-to-end answer grounding check). At least one of these substrings
   must show up — they're treated as alternatives, not all required.

All checks are intentionally LOOSE ("at least one substring") so LLM
paraphrasing doesn't cause false failures. The point is to catch
regressions ("none of the expected facts appeared"), not exact wording.

Corpus: rag_v3_implementation_architecture.md (the V3 spec).
28 chunks after ingestion via SemanticChunker.
"""
from __future__ import annotations

GOLDEN_SET = [
    # ── FACTUAL / CONSTANT LOOKUPS ───────────────────────────────────────
    # Test BM25 + dense retrieval on terse, specific values.
    {
        "id": "chunking",
        "query": "What is the chunking strategy?",
        "must_retrieve": ["sentence-level", "semantic chunking"],
        "must_generate": ["sentence", "0.3", "semantic"],
    },
    {
        "id": "max_chunk_tokens",
        "query": "What is the maximum chunk token size?",
        "must_retrieve": ["max_chunk_tokens", "512"],
        "must_generate": ["512"],
    },
    {
        "id": "min_chunk_tokens",
        "query": "What is the minimum chunk token size before chunks get merged?",
        "must_retrieve": ["min_chunk_tokens", "50"],
        "must_generate": ["50"],
    },
    {
        "id": "semantic_threshold",
        "query": "What inter-sentence similarity threshold triggers a new chunk?",
        "must_retrieve": ["semantic_threshold", "0.3"],
        "must_generate": ["0.3"],
    },
    {
        "id": "embedder",
        "query": "Which embedding model does the system use?",
        "must_retrieve": ["bge-small", "384"],
        "must_generate": ["bge-small", "384"],
    },
    {
        "id": "embed_dimensions",
        "query": "How many dimensions are the embedding vectors?",
        "must_retrieve": ["384", "dimensions"],
        "must_generate": ["384"],
    },
    {
        "id": "database",
        "query": "What database does the system use for storage?",
        "must_retrieve": ["sqlite", "vec"],
        "must_generate": ["sqlite", "vec"],
    },
    {
        "id": "rrf_k",
        "query": "What is the RRF k constant value in the fusion formula?",
        "must_retrieve": ["rrf", "k=60", "k = 60", "60"],
        "must_generate": ["60"],
    },
    {
        "id": "context_budget",
        "query": "What is the maximum context budget for the LLM in tokens?",
        "must_retrieve": ["12,000", "12000", "12k", "budget"],
        "must_generate": ["12,000", "12000", "12k"],
    },
    {
        "id": "rerank_top_k",
        "query": "How many chunks does the FlashRank reranker reduce to?",
        "must_retrieve": ["15", "rerank", "flashrank"],
        "must_generate": ["15"],
    },
    {
        "id": "working_memory_size",
        "query": "How many conversation turns does working memory keep?",
        "must_retrieve": ["working memory", "10 turns", "10"],
        "must_generate": ["10"],
    },
    {
        "id": "cache_threshold",
        "query": "What cosine similarity threshold triggers a semantic cache hit?",
        "must_retrieve": ["cache", "0.95", "similarity"],
        "must_generate": ["0.95", "0.85"],  # 0.85 after this session's lowering
    },

    # ── BEHAVIORAL / "HOW DOES X WORK" ────────────────────────────────────
    # Test semantic search + reasoning over multiple chunks.
    {
        "id": "hybrid_search",
        "query": "How does hybrid search combine dense and sparse retrieval?",
        "must_retrieve": ["bm25", "rrf"],
        "must_generate": ["bm25", "rrf", "fusion"],
    },
    {
        "id": "parent_context",
        "query": "How does parent-context injection work?",
        "must_retrieve": ["parent", "section", "small-to-big"],
        "must_generate": ["parent", "section", "small-to-big"],
    },
    {
        "id": "crag_gate",
        "query": "What is the CRAG quality gate used for?",
        "must_retrieve": ["crag", "sufficient"],
        "must_generate": ["crag", "sufficient", "insufficient", "retrieval"],
    },
    {
        "id": "faithfulness_check",
        "query": "What does the post-generation faithfulness check verify?",
        "must_retrieve": ["faithfulness", "supported"],
        "must_generate": ["claim", "supported", "verified", "context"],
    },
    {
        "id": "query_decomposition",
        "query": "When does the system decompose a query into sub-queries?",
        "must_retrieve": ["decomposition", "complex", "sub-quer"],
        "must_generate": ["complex", "sub-quer", "compare"],
    },
    {
        "id": "pdf_tiers",
        "query": "What are the three PDF parsing tiers?",
        "must_retrieve": ["pymupdf4llm", "docling", "llamaparse"],
        "must_generate": ["pymupdf4llm", "docling", "llamaparse"],
    },
    {
        "id": "code_parsing",
        "query": "How does the system parse source code files?",
        "must_retrieve": ["tree-sitter", "ast"],
        "must_generate": ["tree-sitter", "ast", "function", "class"],
    },
    {
        "id": "semantic_cache",
        "query": "How does the in-process semantic cache work?",
        "must_retrieve": ["semantic cache", "cosine", "embedding"],
        "must_generate": ["cosine", "similarity", "embedding"],
    },
    {
        "id": "abstention",
        "query": "When does the system abstain from answering?",
        "must_retrieve": ["abstain", "insufficient", "don't have"],
        "must_generate": ["abstain", "insufficient", "don't have enough"],
    },
    {
        "id": "ingestion_worker",
        "query": "How does the background ingestion worker avoid blocking queries?",
        "must_retrieve": ["semaphore", "thread", "ingestion"],
        "must_generate": ["semaphore", "thread", "cpu", "block"],
    },
    {
        "id": "rrf_formula",
        "query": "What is the Reciprocal Rank Fusion scoring formula?",
        "must_retrieve": ["1/", "rrf", "rank"],
        "must_generate": ["1 / (k", "1/(k", "rank"],
    },
    {
        "id": "wal_pragmas",
        "query": "What SQLite PRAGMAs does the system set on every connection?",
        "must_retrieve": ["wal", "journal_mode", "pragma"],
        "must_generate": ["wal", "synchronous", "foreign_keys"],
    },

    # ── ARCHITECTURAL / MEMORY ───────────────────────────────────────────
    {
        "id": "memory_layers",
        "query": "What memory layers does the system have?",
        "must_retrieve": ["working memory", "episodic", "procedural"],
        "must_generate": ["working", "episodic", "procedural", "preference"],
    },
    {
        "id": "episodic_vs_working",
        "query": "What is the difference between working memory and episodic memory?",
        "must_retrieve": ["working memory", "episodic"],
        "must_generate": ["working", "episodic", "summari", "cross-session"],
    },
    {
        "id": "procedural_rules",
        "query": "What are procedural rules used for?",
        "must_retrieve": ["procedural", "rule", "trigger"],
        "must_generate": ["rule", "trigger", "inject", "instruction"],
    },
    {
        "id": "deployment_ram",
        "query": "What is the RAM budget for the warm running system?",
        "must_retrieve": ["ram", "gb", "warm"],
        "must_generate": ["gb", "6.9", "7", "memory"],
    },

    # ── ADVERSARIAL / EDGE CASES ─────────────────────────────────────────
    {
        "id": "no_results_path",
        "query": "What happens when retrieval returns zero results?",
        "must_retrieve": ["bm25-only", "no relevant", "abstain"],
        "must_generate": ["bm25", "fallback", "abstain", "no relevant"],
    },
    {
        "id": "flashrank_failure",
        "query": "What happens if the FlashRank reranker fails?",
        "must_retrieve": ["flashrank", "exception", "fallback"],
        "must_generate": ["fallback", "without reranking", "hybrid"],
    },
]
