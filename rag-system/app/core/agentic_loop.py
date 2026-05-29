"""Agentic ReAct retrieval loop.

Replaces the one-shot "search once → rerank → generate" retrieval stage with
an iterative loop where the LLM *drives* retrieval. Each step the model sees
the question and the evidence gathered so far, then chooses one action:

    search(query)   — run another hybrid search to fill a gap
    read(chunk_id)  — pull a chunk's full text + parent section
    answer          — stop; the evidence on hand is enough

The loop is purely a smarter *retrieval* stage: it accumulates a pool of
candidate chunks and hands them back to the pipeline, which then reranks,
injects parent context, and generates exactly as in the one-shot path. That
keeps streaming, faithfulness, and citations untouched.

Why ReAct text-parsing and not native tool-calling: LLMClient only exposes
generate()/generate_stream() against an OpenAI-compatible endpoint whose
function-calling support is not guaranteed (gpt-oss on ollama.com). A strict
JSON action protocol is portable across every provider the client supports.

Hardware note: this adds cloud LLM round-trips (one cheap planning call per
iteration) but NO new local model. Local cost is just running the existing
hybrid search N≤max_iterations times — bounded and CPU-cheap.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, AsyncIterator

from app.infrastructure.config import Settings
from app.infrastructure.database import Database, ChunkResult
from app.infrastructure.observability import get_logger, metrics
from app.models.embedder import Embedder
from app.models.llm_client import LLMClient
from app.retrieval.hybrid_search import HybridSearchEngine

logger = get_logger("core.agentic_loop")

# How many chunks the seed search and each follow-up search contribute, and
# the hard cap on the evidence pool. The pool is the candidate set the
# downstream reranker trims to rerank_top_k, so it only needs to be a few
# multiples of that — keeping it small also keeps each planning prompt lean.
_SEED_TAKE = 12
_SEARCH_TAKE = 8
_POOL_CAP = 40

# Chars of a chunk shown to the planner. Truncated so a 5-iteration loop
# doesn't balloon the planning prompt; READ expands a chunk past this.
_SNIPPET_CHARS = 200
_EXPANDED_CHARS = 800

# Most chunks listed in a single planning prompt (highest score first).
_EVIDENCE_SHOWN = 20


_PLANNER_SYSTEM = """\
You are a retrieval planner for a RAG system. Your job is to gather enough \
evidence from a document corpus to answer the user's QUESTION. You do NOT \
answer the question yourself — you only choose the next retrieval action.

Respond with ONLY a single JSON object and nothing else. Valid actions:

  {"action": "search", "query": "<a focused search query>"}
      Search the corpus for more evidence. Use a DIFFERENT angle or more \
specific terms than previous searches to fill a gap.

  {"action": "read", "chunk_id": "<id from the EVIDENCE list>"}
      Pull the full text and surrounding section of a chunk you have only \
seen as a snippet, when it looks highly relevant but truncated.

  {"action": "answer"}
      Stop. The EVIDENCE already gathered is enough to answer the QUESTION. \
Prefer this as soon as you have enough — do not over-search.

Rules:
- One concept per search. Keep queries targeted.
- The corpus may simply not contain the answer. If two searches in a row add \
nothing useful, choose "answer" — the system will abstain gracefully.
- Never invent a chunk_id; only "read" ids shown in EVIDENCE."""


class AgenticRetriever:
    """Iterative LLM-driven retrieval. See module docstring."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        search_engine: HybridSearchEngine,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._search = search_engine
        self._llm = llm
        self._settings = settings
        self._max_iterations = max(
            1, getattr(settings.pipeline, "agentic_max_iterations", 5)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def gather(self, query: str) -> list[ChunkResult]:
        """Run the loop and return the accumulated evidence pool.

        Convenience wrapper over :meth:`run` for the non-streaming path:
        consumes the event stream, discards status events, and returns the
        final chunk list.
        """
        chunks: list[ChunkResult] = []
        async for event in self.run(query):
            if event["type"] == "result":
                chunks = event["chunks"]
        return chunks

    async def run(self, query: str) -> AsyncIterator[dict]:
        """Drive the ReAct loop, yielding progress then a final result.

        Yields dicts:
            {"type": "status", "msg": str}            — human-readable step
            {"type": "result", "chunks": list[ChunkResult]}  — final (last)

        The result event is always emitted exactly once, at the end, even
        when the loop errors out (it falls back to whatever was gathered).
        """
        r = self._settings.retrieval
        pool: dict[str, ChunkResult] = {}
        expanded: dict[str, str] = {}
        history: list[str] = []

        # ---- Seed: one ordinary hybrid search on the raw query. -------
        # Guarantees the pool is never worse than the one-shot path even
        # if the planner immediately answers.
        yield {"type": "status", "msg": f"Searching: {query[:80]}"}
        try:
            seed = await self._search.search(
                query,
                dense_top_k=r.dense_top_k,
                sparse_top_k=r.sparse_top_k,
                rrf_k=r.rrf_k,
                top_n=r.rrf_top_n,
            )
        except Exception as e:
            logger.warning("agentic_seed_search_failed", error=str(e))
            seed = []
        self._add_to_pool(pool, seed[:_SEED_TAKE])
        history.append(f'search("{query}") → {len(seed)} hits')

        # ---- Iterate: let the model plan follow-up actions. -----------
        iterations = 0
        for step in range(self._max_iterations):
            action = await self._decide(query, pool, expanded, history)
            act = action.get("action", "answer")

            if act == "answer":
                history.append("answer")
                break

            iterations += 1

            if act == "search":
                sub_q = (action.get("query") or "").strip()
                if not sub_q:
                    history.append("search(<empty>) → skipped")
                    continue
                yield {"type": "status", "msg": f"Searching: {sub_q[:80]}"}
                try:
                    hits = await self._search.search(
                        sub_q,
                        dense_top_k=r.dense_top_k,
                        sparse_top_k=r.sparse_top_k,
                        rrf_k=r.rrf_k,
                        top_n=r.rrf_top_n,
                    )
                except Exception as e:
                    logger.warning("agentic_search_failed", query=sub_q, error=str(e))
                    hits = []
                added = self._add_to_pool(pool, hits[:_SEARCH_TAKE])
                history.append(f'search("{sub_q}") → {len(hits)} hits, {added} new')

            elif act == "read":
                cid = (action.get("chunk_id") or "").strip()
                yield {"type": "status", "msg": f"Reading chunk {cid[:24]}"}
                ok = await self._read_chunk(cid, pool, expanded)
                history.append(
                    f'read("{cid}") → {"expanded" if ok else "not found"}'
                )

            else:
                # Unknown action — treat as a no-op and stop to avoid spinning.
                history.append(f"invalid action {act!r} → answer")
                break

        metrics.increment("agentic_loops")
        metrics.increment("agentic_iterations_total", value=iterations)
        logger.info(
            "agentic_loop_complete",
            query=query[:100],
            iterations=iterations,
            pool_size=len(pool),
            reads=len(expanded),
        )

        # Highest-scoring first so the reranker sees the best candidates.
        result = sorted(pool.values(), key=lambda c: c.score, reverse=True)
        yield {"type": "result", "chunks": result}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _add_to_pool(pool: dict[str, ChunkResult], chunks: list[ChunkResult]) -> int:
        """Merge chunks into the pool (keep best score), enforce the cap.

        Returns the count of genuinely new chunk_ids added.
        """
        added = 0
        for c in chunks:
            existing = pool.get(c.chunk_id)
            if existing is None:
                pool[c.chunk_id] = c
                added += 1
            elif c.score > existing.score:
                pool[c.chunk_id] = c

        # Enforce cap by dropping the lowest-scoring chunks.
        if len(pool) > _POOL_CAP:
            keep = sorted(pool.values(), key=lambda c: c.score, reverse=True)[:_POOL_CAP]
            pool.clear()
            pool.update({c.chunk_id: c for c in keep})
        return added

    async def _read_chunk(
        self,
        chunk_id: str,
        pool: dict[str, ChunkResult],
        expanded: dict[str, str],
    ) -> bool:
        """Expand a chunk: store its full text + parent section for the planner.

        Returns True if the chunk was found. Also ensures the chunk is in the
        pool so a READ can rescue a chunk the planner saw but that fell off.
        """
        if not chunk_id:
            return False
        try:
            record = await self._db.get_chunk_by_id(chunk_id)
        except Exception as e:
            logger.warning("agentic_read_failed", chunk_id=chunk_id, error=str(e))
            return False
        if record is None:
            return False

        full = record.content or ""
        # Pull the parent section if there is one, for surrounding context.
        if record.parent_chunk_id:
            try:
                parent = await self._db.get_chunk_by_id(record.parent_chunk_id)
                if parent is not None and parent.content:
                    full = f"{full}\n\n[Parent section]\n{parent.content}"
            except Exception as e:
                logger.warning(
                    "agentic_read_parent_failed",
                    chunk_id=chunk_id,
                    parent_id=record.parent_chunk_id,
                    error=str(e),
                )

        expanded[chunk_id] = full

        # Make sure a READ chunk survives into the final evidence even if it
        # never came from a search (it would otherwise not be in the pool).
        if chunk_id not in pool:
            pool[chunk_id] = ChunkResult(
                chunk_id=record.chunk_id,
                doc_id=record.doc_id,
                content=record.content,
                score=0.0,
                chunk_index=record.chunk_index,
                parent_chunk_id=record.parent_chunk_id,
                section_title=record.section_title,
                source_page=record.source_page,
                token_count=record.token_count,
            )
        return True

    async def _decide(
        self,
        query: str,
        pool: dict[str, ChunkResult],
        expanded: dict[str, str],
        history: list[str],
    ) -> dict[str, Any]:
        """Ask the LLM for the next action. Fail-safe to {"action": "answer"}."""
        evidence_block = self._format_evidence(pool, expanded)
        history_block = "\n".join(f"- {h}" for h in history) or "(none yet)"

        user = (
            f"QUESTION:\n{query}\n\n"
            f"ACTIONS TAKEN SO FAR:\n{history_block}\n\n"
            f"EVIDENCE GATHERED ({len(pool)} chunks):\n{evidence_block}\n\n"
            f"What is your next action? Respond with ONLY the JSON object."
        )
        messages = [
            {"role": "system", "content": _PLANNER_SYSTEM},
            {"role": "user", "content": user},
        ]

        try:
            raw = await self._llm.generate(messages, temperature=0.0, max_tokens=150)
        except Exception as e:
            # A failed planning call must not break the request — stop the
            # loop and let the pipeline answer with what's already gathered.
            logger.warning("agentic_decide_failed", error=str(e))
            return {"action": "answer"}

        return self._parse_action(raw)

    def _format_evidence(
        self,
        pool: dict[str, ChunkResult],
        expanded: dict[str, str],
    ) -> str:
        """Render the evidence pool for the planning prompt (highest score first)."""
        if not pool:
            return "(nothing yet)"
        ordered = sorted(pool.values(), key=lambda c: c.score, reverse=True)
        lines: list[str] = []
        for c in ordered[:_EVIDENCE_SHOWN]:
            src = os.path.basename(c.source_uri) if c.source_uri else "?"
            section = c.section_title or ""
            head = f"[{c.chunk_id}] {src}"
            if section:
                head += f" :: {section}"
            if c.chunk_id in expanded:
                body = expanded[c.chunk_id][:_EXPANDED_CHARS].replace("\n", " ")
            else:
                body = (c.content or "")[:_SNIPPET_CHARS].replace("\n", " ")
            lines.append(f"{head}\n  {body}")
        return "\n".join(lines)

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        """Extract the first JSON object from the model's reply.

        Tolerates ```json fences and surrounding prose. Falls back to
        {"action": "answer"} on any parse failure so the loop always
        terminates cleanly rather than spinning on malformed output.
        """
        if not text:
            return {"action": "answer"}
        # Strip code fences if present.
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        # Grab the first {...} block.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"action": "answer"}
        try:
            obj = json.loads(cleaned[start : end + 1])
        except (json.JSONDecodeError, ValueError):
            return {"action": "answer"}
        if not isinstance(obj, dict):
            return {"action": "answer"}

        action = str(obj.get("action", "")).lower().strip()
        if action not in ("search", "read", "answer"):
            return {"action": "answer"}
        return obj
