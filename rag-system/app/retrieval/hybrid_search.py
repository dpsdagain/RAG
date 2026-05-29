"""Hybrid search engine combining dense vector retrieval and sparse BM25 retrieval.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from both retrieval
methods into a single high-quality result set.
"""
from __future__ import annotations

import asyncio
import re
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


# ---------------------------------------------------------------------------
# Symbol Guarantee — ported and adapted from new_llm_v2_modified
# ---------------------------------------------------------------------------
# A query like "What is RETRIEVER_K?" contains a code symbol. The chunk
# that DEFINES that symbol (`RETRIEVER_K = 12`) is short, has weak dense
# signal, and may lose RRF to longer usage-site chunks. The Symbol
# Guarantee detects code-shaped identifiers in the query and force-
# includes the BM25-best matching chunk past the RRF cut.
#
# Strict no-op on natural-language queries — only fires when the query
# contains ALL_CAPS (≥4 chars) or snake_case (with underscore, ≥4 chars).

# Match ALL_CAPS constants like RETRIEVER_K, MAX_CACHE_CHECKPOINTS
_ALL_CAPS_RE = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")
# Match snake_case identifiers like compress_chat_history, get_embedding_model
_SNAKE_CASE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")

# Match a filename in the query — letters/digits/underscore + dot + 1-5 char extension.
# Common extensions we expect to see in queries about codebases / docs.
_FILENAME_RE = re.compile(
    r"\b([A-Za-z0-9_-]+\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cpp|h|hpp|md|txt|csv|json|yaml|yml|toml|sql|sh|ps1|html|css|cfg|ini))\b",
    re.IGNORECASE,
)


def extract_filename_candidates(query: str) -> list[str]:
    """Return filenames (with extension) mentioned in the query.

    Deduplicated, original case preserved. Empty list when no filenames.
    """
    found = _FILENAME_RE.findall(query)
    seen: set[str] = set()
    out: list[str] = []
    for fname in found:
        if fname not in seen:
            seen.add(fname)
            out.append(fname)
    return out


def extract_symbol_candidates(query: str) -> list[str]:
    """Return code-shaped identifiers from the query.

    Returns a deduplicated list combining ALL_CAPS constants and
    snake_case identifiers. Empty list for prose queries.
    """
    caps = _ALL_CAPS_RE.findall(query)
    snake = _SNAKE_CASE_RE.findall(query)
    # Dedup while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for sym in caps + snake:
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


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

        # 1. Embed query (input_type='query' lets Voyage apply its query
        #    prompt; the local ONNX embedder ignores the kwarg)
        query_embedding: list[float] = self._embedder.embed(query, input_type="query")

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

        # 5. Symbol Guarantee — rescue exact-symbol-named chunks past RRF cut.
        merged_results = await self._apply_symbol_guarantee(
            query, merged_results, top_n
        )

        # 5b. Filename Guarantee — when the user names a file, ensure at
        # least one chunk from that file is in the results.
        merged_results = await self._apply_filename_guarantee(
            query, merged_results, top_n
        )

        # 5c. Call-graph propagation: for queries that look like "where is X
        # used?" / "what calls Y?" / "trace flow of Z", hit the dedicated
        # chunks_symbols_fts index for every symbol in the query and rescue
        # the top hits past the RRF cut. No-op when no symbols in the query.
        merged_results = await self._apply_call_graph_rescue(
            query, merged_results, top_n
        )

        # 6. Metrics
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

    async def _apply_symbol_guarantee(
        self,
        query: str,
        current_results: list[ChunkResult],
        top_n: int,
    ) -> list[ChunkResult]:
        """Force-include chunks defining code symbols mentioned in the query.

        For each ALL_CAPS or snake_case identifier in the query, do a
        targeted BM25 lookup for that exact symbol. If the best match
        isn't already in current_results, prepend it. Bounded by the
        original top_n (the rescued chunks displace lowest-ranked
        existing ones, so total never exceeds top_n).

        Strict no-op on prose queries: no symbols → no extra BM25 calls,
        original ranking returned unchanged.
        """
        symbols = extract_symbol_candidates(query)
        if not symbols:
            return current_results

        existing_ids = {c.chunk_id for c in current_results}
        rescued: list[ChunkResult] = []

        # Cap to first 3 symbols to bound work — a single query with 10
        # identifiers is unusual and probably a copy-paste.
        for sym in symbols[:3]:
            try:
                # Use the symbol itself as a BM25 query — FTS5's porter+
                # unicode61 tokenizer preserves underscores, so RETRIEVER_K
                # and compress_chat_history match exactly.
                hits = await self._db.bm25_search(sym, top_k=3)
            except Exception as e:
                logger.warning("symbol_guarantee_bm25_failed", symbol=sym, error=str(e))
                continue

            for hit in hits:
                if hit.chunk_id in existing_ids:
                    continue  # already in results, no need to rescue
                # First not-already-present hit for this symbol is enough.
                rescued.append(hit)
                existing_ids.add(hit.chunk_id)
                break

        if not rescued:
            return current_results

        metrics.increment("symbol_guarantee_rescues", value=len(rescued))
        logger.info(
            "symbol_guarantee_applied",
            symbols=symbols[:3],
            rescued=len(rescued),
        )

        # Prepend rescued chunks. Trim from the tail to keep top_n.
        combined = rescued + current_results
        return combined[:top_n]

    async def _apply_filename_guarantee(
        self,
        query: str,
        current_results: list[ChunkResult],
        top_n: int,
    ) -> list[ChunkResult]:
        """Force-include at least one chunk from each filename in the query.

        A query like "what's in config.py?" should always surface a chunk
        from config.py even if BM25/dense don't rank it. Tiny zero-chunk
        files lose ranking to multi-chunk files; this rescues them.

        Strict no-op when the query contains no filename.
        """
        filenames = extract_filename_candidates(query)
        if not filenames:
            return current_results

        # Which filenames are already represented?
        present_basenames = set()
        for c in current_results:
            uri = c.source_uri or ""
            if uri:
                # Use the basename so absolute-path source_uri matches the
                # bare filename the user typed.
                from os.path import basename
                present_basenames.add(basename(uri).lower())

        rescued: list[ChunkResult] = []
        existing_ids = {c.chunk_id for c in current_results}

        for fname in filenames[:3]:  # cap, same reasoning as symbol guarantee
            if fname.lower() in present_basenames:
                continue  # already represented in results

            try:
                # Rank chunks by filename match. We rely on the BM25 search
                # but also need to filter to that file's chunks. Cheapest:
                # use BM25 with the file's basename — chunks from that file
                # often mention it (especially after Contextual Retrieval
                # headers are added).
                hits = await self._db.bm25_search(fname, top_k=5)
            except Exception as e:
                logger.warning("filename_guarantee_bm25_failed", filename=fname, error=str(e))
                continue

            for hit in hits:
                if hit.chunk_id in existing_ids:
                    continue
                # Prefer a hit whose source_uri ACTUALLY matches the filename.
                from os.path import basename
                if hit.source_uri and basename(hit.source_uri).lower() == fname.lower():
                    rescued.append(hit)
                    existing_ids.add(hit.chunk_id)
                    break
            else:
                # Fallback: take the first non-present hit even if source_uri
                # doesn't match — better than no rescue.
                for hit in hits:
                    if hit.chunk_id not in existing_ids:
                        rescued.append(hit)
                        existing_ids.add(hit.chunk_id)
                        break

        if not rescued:
            return current_results

        metrics.increment("filename_guarantee_rescues", value=len(rescued))
        logger.info(
            "filename_guarantee_applied",
            filenames=filenames[:3],
            rescued=len(rescued),
        )

        combined = rescued + current_results
        return combined[:top_n]

    # Keywords that signal a propagation / call-graph query. Symbols in
    # the query are routed through chunks_symbols_fts when ANY of these
    # phrases is present. Strict: requires both a symbol AND a propagation
    # keyword to avoid hijacking simple lookup queries.
    _PROPAGATION_KEYWORDS = (
        "where is", "who calls", "what calls", "what uses", "where used",
        "trace ", "propagat", "callers of", "called by", "flow of",
        "how is.* used", "what references", "where does.*come from",
    )

    async def _apply_call_graph_rescue(
        self,
        query: str,
        current_results: list[ChunkResult],
        top_n: int,
    ) -> list[ChunkResult]:
        """For propagation queries naming a code symbol, rescue caller
        chunks via the dedicated chunks_symbols_fts index.

        Only fires when:
          (a) the query contains a code-shaped symbol (ALL_CAPS or
              snake_case identifier), AND
          (b) the query contains a propagation phrase ("where is", "trace",
              "what calls", etc.)

        Strict no-op on lookup queries — those are already handled by the
        Symbol Guarantee.
        """
        symbols = extract_symbol_candidates(query)
        if not symbols:
            return current_results

        q_lower = query.lower()
        is_propagation = any(re.search(kw, q_lower) for kw in self._PROPAGATION_KEYWORDS)
        if not is_propagation:
            return current_results

        existing_ids = {c.chunk_id for c in current_results}
        rescued: list[ChunkResult] = []

        for sym in symbols[:3]:
            try:
                hits = await self._db.search_by_symbol(sym, top_k=5)
            except Exception as e:
                logger.warning("call_graph_rescue_failed", symbol=sym, error=str(e))
                continue
            for hit in hits:
                if hit.chunk_id in existing_ids:
                    continue
                rescued.append(hit)
                existing_ids.add(hit.chunk_id)
                if len(rescued) >= 5:  # cap rescues per query
                    break
            if len(rescued) >= 5:
                break

        if not rescued:
            return current_results

        metrics.increment("call_graph_rescues", value=len(rescued))
        logger.info(
            "call_graph_rescue_applied",
            symbols=symbols[:3],
            rescued=len(rescued),
        )

        combined = rescued + current_results
        return combined[:top_n]

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
