"""Cross-encoder reranker using FlashRank for CPU-friendly reranking.

Takes the top-100 RRF results and reranks them with a cross-encoder model,
returning the top-15 most relevant chunks.
"""
from __future__ import annotations

import time
from typing import Any

from app.infrastructure.database import ChunkResult
from app.infrastructure.observability import get_logger, metrics

logger = get_logger("retrieval.reranker")

_flashrank_available = True
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    _flashrank_available = False
    logger.warning("flashrank_not_installed", msg="FlashRank not available; reranking will pass through")


class Reranker:
    """CPU cross-encoder reranker using FlashRank.

    Falls back to passthrough (returns input unchanged) if FlashRank
    is not installed.
    """

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        """Initialize the reranker.

        Args:
            model_name: FlashRank model name. Downloaded on first use.
        """
        self._model_name = model_name
        self._ranker: Any = None

        if _flashrank_available:
            try:
                self._ranker = Ranker(model_name=model_name)
                logger.info("reranker_loaded", model=model_name)
            except Exception as e:
                logger.error("reranker_load_failed", error=str(e))

    async def rerank(
        self,
        query: str,
        chunks: list[ChunkResult],
        top_k: int = 15,
    ) -> list[ChunkResult]:
        """Rerank chunks using cross-encoder scoring.

        Args:
            query: The original user query.
            chunks: Candidate chunks from hybrid search.
            top_k: Number of top results to return after reranking.

        Returns:
            Reranked list of ChunkResult objects trimmed to top_k.
        """
        if not chunks:
            return []

        if self._ranker is None:
            logger.warning("reranker_passthrough", reason="No ranker loaded")
            return chunks[:top_k]

        t0 = time.perf_counter()

        try:
            # Build FlashRank input
            passages = [
                {"id": chunk.chunk_id, "text": chunk.content}
                for chunk in chunks
            ]
            rerank_request = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(rerank_request)

            # Build reranked list with updated scores
            chunk_map = {c.chunk_id: c for c in chunks}
            reranked: list[ChunkResult] = []

            for result in results[:top_k]:
                chunk_id = result.get("id", result.get("metadata", {}).get("id", ""))
                original = chunk_map.get(chunk_id)
                if original is None:
                    continue
                reranked.append(ChunkResult(
                    chunk_id=original.chunk_id,
                    doc_id=original.doc_id,
                    content=original.content,
                    score=float(result.get("score", 0.0)),
                    chunk_index=original.chunk_index,
                    parent_chunk_id=original.parent_chunk_id,
                    section_title=original.section_title,
                    source_page=original.source_page,
                    token_count=original.token_count,
                    source_uri=original.source_uri,
                    source_type=original.source_type,
                ))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.record_latency("rerank", elapsed_ms)
            logger.info(
                "rerank_complete",
                input_count=len(chunks),
                output_count=len(reranked),
                latency_ms=round(elapsed_ms, 2),
            )
            return reranked

        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            metrics.increment("rerank_errors")
            return chunks[:top_k]
