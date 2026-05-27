"""Golden-set of (query, expected substrings) pairs against the V3 doc.

Each query has:
- `query`: what the user types
- `must_retrieve`: lowercase substrings that should appear in at least one
   chunk among the top-K retrieval results (semantic relevance check)
- `must_generate`: lowercase substrings the LLM answer should contain
   (end-to-end answer grounding check). At least one of these substrings
   must show up — they're treated as alternatives, not all required.
"""
from __future__ import annotations

GOLDEN_SET = [
    {
        "id": "chunking",
        "query": "What is the chunking strategy?",
        "must_retrieve": ["sentence-level", "semantic chunking"],
        "must_generate": ["sentence", "0.3", "semantic"],
    },
    {
        "id": "database",
        "query": "What database does the system use for storage?",
        "must_retrieve": ["sqlite", "vec"],
        "must_generate": ["sqlite", "vec"],
    },
    {
        "id": "hybrid_search",
        "query": "How does hybrid search combine dense and sparse retrieval?",
        "must_retrieve": ["bm25", "rrf"],
        "must_generate": ["bm25", "rrf", "fusion"],
    },
    {
        "id": "embedder",
        "query": "Which embedding model does the system use?",
        "must_retrieve": ["bge-small", "384"],
        "must_generate": ["bge-small", "384"],
    },
    {
        "id": "crag_gate",
        "query": "What is the CRAG quality gate used for?",
        "must_retrieve": ["crag", "sufficient"],
        "must_generate": ["crag", "sufficient", "insufficient", "retrieval"],
    },
]
