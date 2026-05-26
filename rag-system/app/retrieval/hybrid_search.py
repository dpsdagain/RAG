"""Hybrid search engine combining dense vector retrieval and sparse BM25 retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both retrieval
methods into a single high-quality result set.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict

from app.infrastructure.database import Database, ChunkResult
from app.infrastructure.observability import get_logger, metrics
from app.models.embedder import Embedder

logger = get_logger("retrieval.hybrid_search")

# ---------------------------------------------------------------------------
# Defaults – used when the caller does not supply explicit overrides
# ---------------------------------------------------------------------------
_DEFAULT_DENSE_TOP_K: int = 50
_DEFAULT_SPARSE_TOP_K: int = 50
_DEFAULT_RRF_K: int = 60
_DEFAULT_RRF_TOP_N: int = 100


class HybridSearchEngine:
    """Executes hybrid retrieval via dense + BM25 search and fuses with RRF.

    The engine runs both retrieval legs concurrently, merges their ranked lists
    using Reciprocal Rank Fusion, and returns the top-N combined results.
    """

    def __init__(self, db: Database, embedder: Embedder) -> None:
        """Initialise the hybrid search engine.

        Args:
            db: Database handle exposing ``vector_search`` and ``bm25_search``.
            embedder: Text embedder for producing dense query vectors.
        """
        self._db = db
        self._embedder = embedder
        logger.info("hybrid_search_engine.initialised")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        dense_top_k: int = _DEFAULT_DENSE_TOP_K,
        sparse_top_k: int = _DEFAULT_SPARSE_TOP_K,
        rrf_k: int = _DEFAULT_RRF_K,
        top_n: int = _DEFAULT_RRF_TOP_N,
    ) -> list[ChunkResult]:
        """Execute hybrid search: dense + BM25 + RRF fusion.

        Steps:
            1. Embed the query text.
            2. Concurrently execute dense (vector) and sparse (BM25) search.
            3. Merge both ranked lists with Reciprocal Rank Fusion.
            4. Fetch full ``ChunkResult`` objects for the top *top_n* items.
            5. Record latency and result-count metrics.

        Args:
            query: Natural-language search query.
            dense_top_k: Number of results from the dense retrieval leg.
            sparse_top_k: Number of results from the sparse retrieval leg.
            rrf_k: RRF constant controlling rank damping.
            top_n: Maximum number of fused results to return.

        Returns:
            Ranked list of ``ChunkResult`` objects (highest RRF score first).
        """
        t0 = time.perf_counter()

        # 1. Embed query
        query_embedding: list[float] = self._embedder.embed(query)

        # 2. Run dense and sparse searches concurrently
        dense_task = asyncio.create_task(
            self._db.vector_search(query_embedding, dense_top_k)
        )
        sparse_task = asyncio.create_task(
            self._db.bm25_search(query, sparse_top_k)
        )
        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task
        )

        logger.debug(
            "hybrid_search.legs_complete",
            dense_count=len(dense_results),
            sparse_count=len(sparse_results),
        )

        # 3. Prepare ranked id/score tuples and fuse
        dense_ranked: list[tuple[str, float]] = [
            (r.chunk_id, r.score) for r in dense_results
        ]
        sparse_ranked: list[tuple[str, float]] = [
            (r.chunk_id, r.score) for r in sparse_results
        ]
        fused = self.reciprocal_rank_fusion(
            dense_ranked, sparse_ranked, k=rrf_k, top_n=top_n
        )

        # 4. Build a lookup of already-fetched chunks for efficiency
        chunk_map: dict[str, ChunkResult] = {
            r.chunk_id: r for r in dense_results
        }
        chunk_map.update({r.chunk_id: r for r in sparse_results})

        # Fetch any missing chunks (unlikely, but defensive)
        merged_results: list[ChunkResult] = []
        for chunk_id, rrf_score in fused:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                fetched = await self._db.get_chunk_by_id(chunk_id)
                if fetched is None:
                    logger.warning(
                        "hybrid_search.missing_chunk", chunk_id=chunk_id
                    )
                    continue
                # Wrap ChunkRecord → ChunkResult with the RRF score
                chunk = ChunkResult(
                    chunk_id=fetched.chunk_id,
                    doc_id=fetched.doc_id,
                    content=fetched.content,
                    score=rrf_score,
                    chunk_index=fetched.chunk_index,
                    parent_chunk_id=getattr(fetched, "parent_chunk_id", None),
                    section_title=getattr(fetched, "section_title", None),
                    source_page=getattr(fetched, "source_page", None),
                    token_count=getattr(fetched, "token_count", 0),
                    source_uri=getattr(fetched, "source_uri", None),
                    source_type=getattr(fetched, "source_type", None),
                )
            else:
                # Override score with the fused RRF score
                chunk = ChunkResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=rrf_score,
                    chunk_index=chunk.chunk_index,
                    parent_chunk_id=getattr(chunk, "parent_chunk_id", None),
                    section_title=getattr(chunk, "section_title", None),
                    source_page=getattr(chunk, "source_page", None),
                    token_count=getattr(chunk, "token_count", 0),
                    source_uri=getattr(chunk, "source_uri", None),
                    source_type=getattr(chunk, "source_type", None),
                )
            merged_results.append(chunk)

        # 5. Metrics
        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics.record_latency("hybrid_search", elapsed_ms)
        metrics.increment("hybrid_search_count")
        logger.info(
            "hybrid_search.complete",
            query_len=len(query),
            result_count=len(merged_results),
            latency_ms=round(elapsed_ms, 2),
        )

        return merged_results

    async def search_multi_query(
        self,
        queries: list[str],
        **kwargs: int,
    ) -> list[ChunkResult]:
        """Execute hybrid search for multiple sub-queries, deduplicate and re-rank.

        Each sub-query is searched independently.  Results are deduplicated by
        ``chunk_id`` and then re-ranked via a second RRF pass so that chunks
        appearing across many sub-queries are promoted.

        Args:
            queries: List of sub-query strings.
            **kwargs: Forwarded to :py:meth:`search` (e.g. ``dense_top_k``).

        Returns:
            Deduplicated, RRF-fused list of ``ChunkResult`` objects.
        """
        if not queries:
            return []

        t0 = time.perf_counter()
        rrf_k: int = kwargs.get("rrf_k", _DEFAULT_RRF_K)
        top_n: int = kwargs.get("top_n", _DEFAULT_RRF_TOP_N)

        # Run all sub-queries concurrently
        tasks = [
            asyncio.create_task(self.search(q, **kwargs)) for q in queries
        ]
        all_results: list[list[ChunkResult]] = await asyncio.gather(*tasks)

        # Collect per-query ranked lists as (chunk_id, score) pairs
        ranked_lists: list[list[tuple[str, float]]] = [
            [(c.chunk_id, c.score) for c in results] for results in all_results
        ]

        # Fuse across all sub-query result lists using iterative RRF
        combined_scores: dict[str, float] = defaultdict(float)
        for ranked in ranked_lists:
            for rank_pos, (cid, _score) in enumerate(ranked, start=1):
                combined_scores[cid] += 1.0 / (rrf_k + rank_pos)

        sorted_ids = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        # Build chunk lookup from all results
        chunk_map: dict[str, ChunkResult] = {}
        for results in all_results:
            for c in results:
                if c.chunk_id not in chunk_map or c.score > chunk_map[c.chunk_id].score:
                    chunk_map[c.chunk_id] = c

        merged: list[ChunkResult] = []
        for chunk_id, fused_score in sorted_ids:
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            merged.append(
                ChunkResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=fused_score,
                    chunk_index=chunk.chunk_index,
                    parent_chunk_id=getattr(chunk, "parent_chunk_id", None),
                    section_title=getattr(chunk, "section_title", None),
                    source_page=getattr(chunk, "source_page", None),
                    token_count=getattr(chunk, "token_count", 0),
                    source_uri=getattr(chunk, "source_uri", None),
                    source_type=getattr(chunk, "source_type", None),
                )
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        metrics.record_latency("multi_query_search", elapsed_ms)
        logger.info(
            "multi_query_search.complete",
            query_count=len(queries),
            result_count=len(merged),
            latency_ms=round(elapsed_ms, 2),
        )
        return merged

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def reciprocal_rank_fusion(
        dense_results: list[tuple[str, float]],
        sparse_results: list[tuple[str, float]],
        k: int = 60,
        top_n: int = 100,
    ) -> list[tuple[str, float]]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        For each item appearing in a ranked list at position *r* (1-based),
        its RRF score contribution is ``1 / (k + r)``.  Contributions are
        summed across all lists in which the item appears.

        Args:
            dense_results: Ranked (chunk_id, score) pairs from dense retrieval.
            sparse_results: Ranked (chunk_id, score) pairs from sparse retrieval.
            k: RRF damping constant (typically 60).
            top_n: Maximum number of merged results to return.

        Returns:
            Merged (chunk_id, rrf_score) pairs sorted descending by score.
        """
        scores: dict[str, float] = defaultdict(float)

        for rank_pos, (chunk_id, _score) in enumerate(dense_results, start=1):
            scores[chunk_id] += 1.0 / (k + rank_pos)

        for rank_pos, (chunk_id, _score) in enumerate(sparse_results, start=1):
            scores[chunk_id] += 1.0 / (k + rank_pos)

        sorted_results = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_results[:top_n]
