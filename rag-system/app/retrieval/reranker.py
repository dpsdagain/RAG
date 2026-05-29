"""Reranker with three backends: Voyage cloud, ST CrossEncoder, FlashRank.

Takes the top-N RRF results and reranks them, returning the top-K most
relevant chunks.

Backend selection (auto, by model name):
  * Name starts with 'rerank-' (e.g. 'rerank-2.5') → Voyage AI cloud reranker.
    Uses the same Voyage key as the embedder (RAG_EMBEDDING__API_KEY). No local
    model, so it sidesteps the heavy CPU cost of cross-encoders. Falls back to
    local FlashRank ms-marco if the key or SDK is unavailable.
  * Name looks like a HuggingFace repo (contains '/') OR contains
    'jina-reranker' / 'bge-reranker' → sentence-transformers CrossEncoder
    (modern 2024+ rerankers — strong on code AND prose).
  * Otherwise → FlashRank (the 2022-era ms-marco family, lightweight local).

If no backend is available, reranking degrades to passthrough.
"""
from __future__ import annotations

import asyncio
import os
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

_sentence_transformers_available = True
try:
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
except ImportError:
    _sentence_transformers_available = False


def _looks_like_hf_reranker(model_name: str) -> bool:
    """Heuristic: HuggingFace-style names need sentence-transformers."""
    m = model_name.lower()
    if "/" in model_name:
        return True
    return any(tag in m for tag in ("jina-reranker", "bge-reranker", "bge-rerank"))


def _looks_like_voyage_reranker(model_name: str) -> bool:
    """Voyage rerankers are named 'rerank-*' (rerank-2.5, rerank-2.5-lite…)."""
    return model_name.lower().startswith("rerank-") and "/" not in model_name


class Reranker:
    """CPU cross-encoder reranker.

    Dual-backend: sentence-transformers CrossEncoder (modern HF models) or
    FlashRank (legacy ms-marco). Falls back to passthrough if neither
    backend can load the configured model.
    """

    def __init__(
        self, model_name: str = "ms-marco-MiniLM-L-12-v2", api_key: str = ""
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: Voyage reranker ('rerank-2.5'), a HF repo id
                        ('BAAI/bge-reranker-v2-m3'), or a FlashRank built-in
                        name ('ms-marco-MiniLM-L-12-v2').
            api_key: Voyage API key (only used for 'rerank-*' models). Falls
                     back to RAG_EMBEDDING__API_KEY / VOYAGE_API_KEY env vars.
        """
        self._model_name = model_name
        self._backend: str = "passthrough"  # 'voyage' | 'ce' | 'flashrank' | 'passthrough'
        self._ranker: Any = None
        self._client: Any = None

        # 1) Voyage cloud reranker (rerank-2.5 etc.) — same key as embeddings.
        #    No local model; degrades to local FlashRank ms-marco if the key or
        #    SDK is missing so reranking never hard-fails.
        if _looks_like_voyage_reranker(model_name):
            key = (
                api_key
                or os.getenv("RAG_EMBEDDING__API_KEY", "")
                or os.getenv("VOYAGE_API_KEY", "")
            )
            if key:
                try:
                    import voyageai
                    self._client = voyageai.Client(api_key=key)
                    self._backend = "voyage"
                    self._max_retries = 3
                    logger.info("reranker_loaded", model=model_name, backend="voyage")
                    return
                except Exception as e:
                    logger.warning("voyage_reranker_load_failed", model=model_name, error=str(e))
            else:
                logger.warning(
                    "voyage_reranker_no_key", model=model_name,
                    msg="No Voyage API key; falling back to local FlashRank ms-marco.",
                )
            # Local fallback target (the Voyage name is meaningless to FlashRank).
            model_name = "ms-marco-MiniLM-L-12-v2"
            self._model_name = model_name

        # Prefer sentence-transformers when the model name looks like a HF
        # cross-encoder. Otherwise prefer FlashRank.
        prefer_ce = _looks_like_hf_reranker(model_name)

        if prefer_ce and _sentence_transformers_available:
            try:
                # CrossEncoder downloads from HF on first use (~280-560 MB
                # depending on model). Stored under ~/.cache/huggingface/.
                self._ranker = CrossEncoder(model_name, device="cpu")
                self._backend = "ce"
                logger.info("reranker_loaded", model=model_name, backend="cross_encoder")
                return
            except Exception as e:
                logger.warning("ce_load_failed_falling_back", model=model_name, error=str(e))

        if _flashrank_available:
            try:
                self._ranker = Ranker(model_name=model_name)
                self._backend = "flashrank"
                logger.info("reranker_loaded", model=model_name, backend="flashrank")
                return
            except Exception as e:
                logger.warning("flashrank_load_failed", model=model_name, error=str(e))

        logger.warning(
            "reranker_unavailable",
            model=model_name,
            msg=(
                "Neither sentence-transformers nor FlashRank could load this "
                "model. Reranking will passthrough."
            ),
        )

    async def rerank(
        self,
        query: str,
        chunks: list[ChunkResult],
        top_k: int = 15,
    ) -> list[ChunkResult]:
        """Rerank chunks using cross-encoder scoring.

        The actual scoring runs in a thread (asyncio.to_thread) so the
        event loop isn't blocked for hundreds of milliseconds.
        """
        if not chunks:
            return []

        # Passthrough only when nothing loaded. Note the Voyage backend keeps
        # its handle in self._client (not self._ranker), so guard on both.
        if self._backend == "passthrough" or (self._ranker is None and self._client is None):
            return chunks[:top_k]

        t0 = time.perf_counter()

        try:
            if self._backend == "voyage":
                reranked = await asyncio.to_thread(
                    self._rerank_with_voyage, query, chunks, top_k
                )
            elif self._backend == "ce":
                reranked = await asyncio.to_thread(
                    self._rerank_with_ce, query, chunks, top_k
                )
            else:  # flashrank
                reranked = await asyncio.to_thread(
                    self._rerank_with_flashrank, query, chunks, top_k
                )

            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.record_latency("rerank", elapsed_ms)
            logger.info(
                "rerank_complete",
                input_count=len(chunks),
                output_count=len(reranked),
                latency_ms=round(elapsed_ms, 2),
                backend=self._backend,
            )
            return reranked

        except Exception as e:
            logger.error("rerank_failed", error=str(e), backend=self._backend)
            metrics.increment("rerank_errors")
            return chunks[:top_k]

    # ------------------------------------------------------------------
    # Backend implementations (sync — call via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _rerank_with_voyage(
        self, query: str, chunks: list[ChunkResult], top_k: int
    ) -> list[ChunkResult]:
        """Voyage cloud reranker (rerank-2.5).

        Voyage returns results pre-sorted by relevance; each result's `index`
        maps back into the input `chunks` list, and `top_k` is applied server
        side. One network call per rerank (no local model).
        """
        documents = [c.content for c in chunks]
        # Retry transient network/rate-limit errors with backoff (mirrors the
        # embedder). The outer rerank() handler only falls back to passthrough
        # after these retries are exhausted.
        last_err: Exception | None = None
        resp = None
        for attempt in range(self._max_retries):
            try:
                resp = self._client.rerank(
                    query, documents, model=self._model_name, top_k=top_k
                )
                break
            except Exception as e:
                last_err = e
                if attempt < self._max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))
        if resp is None:
            raise RuntimeError(
                f"Voyage rerank failed after {self._max_retries} attempts: {last_err}"
            ) from last_err

        reranked: list[ChunkResult] = []
        for res in resp.results:
            original = chunks[res.index]
            reranked.append(ChunkResult(
                chunk_id=original.chunk_id,
                doc_id=original.doc_id,
                content=original.content,
                score=float(res.relevance_score),
                chunk_index=original.chunk_index,
                parent_chunk_id=original.parent_chunk_id,
                section_title=original.section_title,
                source_page=original.source_page,
                token_count=original.token_count,
                source_uri=original.source_uri,
                source_type=original.source_type,
            ))
        return reranked

    def _rerank_with_ce(
        self, query: str, chunks: list[ChunkResult], top_k: int
    ) -> list[ChunkResult]:
        """sentence-transformers CrossEncoder scoring."""
        pairs = [(query, c.content) for c in chunks]
        scores = self._ranker.predict(pairs, show_progress_bar=False)

        # Sort by score descending, keep top_k
        scored = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
        reranked: list[ChunkResult] = []
        for score, original in scored[:top_k]:
            reranked.append(ChunkResult(
                chunk_id=original.chunk_id,
                doc_id=original.doc_id,
                content=original.content,
                score=float(score),
                chunk_index=original.chunk_index,
                parent_chunk_id=original.parent_chunk_id,
                section_title=original.section_title,
                source_page=original.source_page,
                token_count=original.token_count,
                source_uri=original.source_uri,
                source_type=original.source_type,
            ))
        return reranked

    def _rerank_with_flashrank(
        self, query: str, chunks: list[ChunkResult], top_k: int
    ) -> list[ChunkResult]:
        """FlashRank scoring (legacy ms-marco family)."""
        passages = [
            {"id": chunk.chunk_id, "text": chunk.content}
            for chunk in chunks
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(rerank_request)

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
        return reranked
