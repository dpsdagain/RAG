"""RAG Pipeline: the main 10-step orchestrator.

Executes: Cache → Route → Decompose → Retrieve → Rerank →
Parent Inject → CRAG → Generate → Faithfulness → Cache → Return

This is the core engine of the entire RAG system.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from app.api.schemas import ChatResponse, ChatStreamEvent, SourceCitation
from app.core.crag_gate import CRAGGate
from app.core.faithfulness_checker import FaithfulnessChecker
from app.core.prompt_templates import generation, query_rewrite
from app.core.query_decomposer import QueryDecomposer
from app.core.query_router import QueryRouter
from app.infrastructure.config import Settings
from app.infrastructure.database import ChunkResult, Database
from app.infrastructure.observability import get_logger, metrics
from app.memory.episodic_memory import EpisodicMemory
from app.memory.preference_store import PreferenceStore
from app.memory.procedural_rules import ProceduralRuleEngine
from app.memory.working_memory import WorkingMemory
from app.models.embedder import Embedder
from app.models.llm_client import LLMClient
from app.retrieval.hybrid_search import HybridSearchEngine
from app.retrieval.parent_context import ParentContextInjector
from app.retrieval.reranker import Reranker
from app.retrieval.semantic_cache import SemanticCache

logger = get_logger("core.pipeline")


class RAGPipeline:
    """Linear async pipeline: the main RAG orchestrator.

    Steps:
    1. Semantic cache check
    2. Query routing (conversational → skip retrieval)
    3. Query decomposition (if complex)
    4. Hybrid retrieval (dense + BM25 + RRF)
    5. FlashRank reranking (100 → 15)
    6. Parent context injection (with budget cap)
    7. CRAG quality gate (with retry on INSUFFICIENT)
    8. Generation with RADIO citations
    9. Faithfulness check
    10. Cache + log + return
    """

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._settings = settings

        # Initialize all sub-components
        self.search_engine = HybridSearchEngine(db, embedder)
        self.reranker = Reranker(settings.retrieval.rerank_model)
        self.parent_injector = ParentContextInjector(db, embedder)
        self.cache = SemanticCache(
            max_entries=settings.cache.max_entries,
            similarity_threshold=settings.cache.similarity_threshold,
            ttl_seconds=settings.cache.ttl_seconds,
        )
        self.working_memory = WorkingMemory(db, settings.memory.working_memory_max_turns)
        self.episodic_memory = EpisodicMemory(
            db, embedder, llm, settings.memory.episodic_memory_top_k
        )
        self.preferences = PreferenceStore(db)
        self.rules = ProceduralRuleEngine(db)
        self.router = QueryRouter()
        self.crag = CRAGGate(llm)
        self.decomposer = QueryDecomposer(llm)
        self.faithfulness = FaithfulnessChecker(llm)

        logger.info("rag_pipeline_initialized")

    async def execute(
        self,
        query: str,
        conversation_id: str | None = None,
        stream: bool = False,
    ) -> ChatResponse | AsyncIterator[ChatStreamEvent]:
        """Execute the full 10-step RAG pipeline.

        Args:
            query: User's query text.
            conversation_id: Thread ID for conversation continuity.
            stream: If True, return an async iterator of stream events.

        Returns:
            ChatResponse for non-streaming, or AsyncIterator for streaming.
        """
        t0 = time.perf_counter()
        request_id = str(uuid.uuid4())

        # Get or create conversation thread
        thread_id = await self.working_memory.get_or_create_thread(conversation_id)

        try:
            # ============================================================
            # Step 1: Semantic Cache Check
            # ============================================================
            query_embedding = self._embedder.embed(query)
            cached = self.cache.lookup(query_embedding)

            if cached is not None:
                logger.info("pipeline_cache_hit", request_id=request_id)
                metrics.increment("pipeline_cache_hits")

                # Store the turn even for cache hits
                await self.working_memory.add_turn(thread_id, "user", query)
                await self.working_memory.add_turn(thread_id, "assistant", cached.response)

                latency_ms = (time.perf_counter() - t0) * 1000
                return ChatResponse(
                    response=cached.response,
                    sources=[SourceCitation(**s) for s in cached.sources],
                    conversation_id=thread_id,
                    crag_verdict="CACHED",
                    latency_ms=round(latency_ms, 1),
                    query_type="cached",
                )

            # ============================================================
            # Step 2: Query Routing
            # ============================================================
            query_type = self.router.classify(query)
            logger.info("query_classified", query_type=query_type, request_id=request_id)

            if query_type == "conversational":
                return await self._handle_conversational(query, thread_id, t0)

            # ============================================================
            # Step 3: Query Decomposition (if complex)
            # ============================================================
            sub_queries = [query]
            if (
                query_type == "complex_retrieval"
                and self._settings.pipeline.decomposition_enabled
            ):
                sub_queries = await self.decomposer.decompose(query)
                logger.info("query_decomposed", sub_queries=len(sub_queries))

            # ============================================================
            # Step 4: Hybrid Retrieval
            # ============================================================
            if len(sub_queries) > 1:
                retrieval_results = await self.search_engine.search_multi_query(
                    sub_queries,
                    dense_top_k=self._settings.retrieval.dense_top_k,
                    sparse_top_k=self._settings.retrieval.sparse_top_k,
                    rrf_k=self._settings.retrieval.rrf_k,
                    top_n=self._settings.retrieval.rrf_top_n,
                )
            else:
                retrieval_results = await self.search_engine.search(
                    query,
                    dense_top_k=self._settings.retrieval.dense_top_k,
                    sparse_top_k=self._settings.retrieval.sparse_top_k,
                    rrf_k=self._settings.retrieval.rrf_k,
                    top_n=self._settings.retrieval.rrf_top_n,
                )

            logger.info("retrieval_complete", results=len(retrieval_results))

            if not retrieval_results:
                return await self._build_abstention_response(query, [], thread_id, t0)

            # ============================================================
            # Step 5: FlashRank Reranking (100 → 15)
            # ============================================================
            reranked = await self.reranker.rerank(
                query, retrieval_results, top_k=self._settings.retrieval.rerank_top_k
            )

            # ============================================================
            # Step 6: Parent Context Injection
            # ============================================================
            enriched_chunks = await self.parent_injector.inject_parent_context(
                reranked, budget_tokens=self._settings.retrieval.context_budget_tokens
            )
            context_str = self.parent_injector.format_context_for_prompt(enriched_chunks)

            # ============================================================
            # Step 7: CRAG Quality Gate
            # ============================================================
            crag_verdict = "SKIPPED"
            if self._settings.pipeline.crag_enabled:
                crag_verdict = await self.crag.evaluate(query, reranked)
                logger.info("crag_verdict", verdict=crag_verdict)

                if crag_verdict == "INSUFFICIENT":
                    # Retry with query rewrite
                    rewritten_query, retry_results = await self._retry_with_rewrite(
                        query, enriched_chunks
                    )
                    if retry_results:
                        reranked = await self.reranker.rerank(
                            rewritten_query, retry_results,
                            top_k=self._settings.retrieval.rerank_top_k,
                        )
                        enriched_chunks = await self.parent_injector.inject_parent_context(
                            reranked, budget_tokens=self._settings.retrieval.context_budget_tokens,
                        )
                        context_str = self.parent_injector.format_context_for_prompt(enriched_chunks)

                        # Re-evaluate CRAG
                        crag_verdict = await self.crag.evaluate(rewritten_query, reranked)
                        if crag_verdict == "INSUFFICIENT":
                            return await self._build_abstention_response(
                                query, reranked, thread_id, t0
                            )

            # ============================================================
            # Step 8: Generation with RADIO Citations
            # ============================================================
            working_mem_str = await self.working_memory.format_for_prompt(thread_id)
            episodic_str = await self.episodic_memory.format_for_prompt(query_embedding)
            prefs_str = await self.preferences.format_for_prompt()
            rules_str = await self.rules.format_for_prompt(query)

            messages = generation.build_messages(
                query=query,
                retrieved_chunks=context_str,
                working_memory=working_mem_str,
                procedural_rules=rules_str,
                active_preferences=prefs_str,
                episodic_memory=episodic_str,
            )

            response_text = await self._llm.generate(
                messages,
                temperature=self._settings.llm.temperature,
                max_tokens=self._settings.llm.max_tokens,
            )

            # ============================================================
            # Step 9: Faithfulness Check
            # ============================================================
            faith_result: dict[str, Any] = {"verified": True, "raw_result": "SKIPPED"}
            if self._settings.pipeline.faithfulness_check_enabled:
                faith_result = await self.faithfulness.check(response_text, enriched_chunks)

            # ============================================================
            # Step 10: Cache + Log + Return
            # ============================================================
            # Build source citations
            sources = self._build_citations(reranked)

            # Store in cache
            self.cache.store(
                query_embedding=query_embedding,
                response=response_text,
                sources=[s.model_dump() for s in sources],
                conversation_id=thread_id,
            )

            # Store conversation turns
            await self.working_memory.add_turn(thread_id, "user", query)
            await self.working_memory.add_turn(thread_id, "assistant", response_text)

            # Log request
            latency_ms = (time.perf_counter() - t0) * 1000
            metrics.record_latency("pipeline_total", latency_ms)
            metrics.increment("pipeline_executions")

            try:
                await self._db.log_request(
                    request_id=request_id,
                    query=query,
                    sub_queries=sub_queries if len(sub_queries) > 1 else None,
                    crag_verdict=crag_verdict,
                    faithfulness=faith_result.get("raw_result", ""),
                    retrieved_chunks=[c.chunk_id for c in reranked],
                    response_length=len(response_text),
                    total_latency_ms=int(latency_ms),
                )
            except Exception as e:
                logger.warning("request_log_failed", error=str(e))

            logger.info(
                "pipeline_complete",
                request_id=request_id,
                query_type=query_type,
                crag=crag_verdict,
                faithful=faith_result.get("verified", True),
                sources=len(sources),
                latency_ms=round(latency_ms, 1),
            )

            return ChatResponse(
                response=response_text,
                sources=sources,
                conversation_id=thread_id,
                crag_verdict=crag_verdict,
                faithfulness_result=faith_result.get("raw_result"),
                latency_ms=round(latency_ms, 1),
                query_type=query_type,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "pipeline_error",
                request_id=request_id,
                error=str(e),
                latency_ms=round(latency_ms, 1),
            )
            metrics.increment("pipeline_errors")

            return ChatResponse(
                response=f"I encountered an error processing your request. Please try again. Error: {str(e)}",
                sources=[],
                conversation_id=thread_id,
                crag_verdict="ERROR",
                latency_ms=round(latency_ms, 1),
            )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    async def _handle_conversational(
        self,
        query: str,
        thread_id: str,
        t0: float,
    ) -> ChatResponse:
        """Handle conversational queries without retrieval."""
        working_mem = await self.working_memory.format_for_prompt(thread_id)
        prefs = await self.preferences.format_for_prompt()

        messages = [
            {"role": "system", "content": f"You are a helpful assistant.\n{prefs}"},
            {"role": "user", "content": f"Conversation:\n{working_mem}\n\nUser: {query}"},
        ]

        response = await self._llm.generate(messages)

        await self.working_memory.add_turn(thread_id, "user", query)
        await self.working_memory.add_turn(thread_id, "assistant", response)

        latency_ms = (time.perf_counter() - t0) * 1000
        return ChatResponse(
            response=response,
            sources=[],
            conversation_id=thread_id,
            crag_verdict="CONVERSATIONAL",
            latency_ms=round(latency_ms, 1),
            query_type="conversational",
        )

    async def _retry_with_rewrite(
        self,
        query: str,
        context: list[dict],
    ) -> tuple[str, list[ChunkResult]]:
        """Rewrite query and retry retrieval on INSUFFICIENT."""
        context_summary = "\n".join(
            c.get("content", "")[:200] for c in context[:5]
        )

        messages = query_rewrite.build_messages(query, context_summary)

        try:
            rewritten = await self._llm.generate(messages, temperature=0.0, max_tokens=200)
            rewritten = rewritten.strip()

            if rewritten and rewritten != query:
                logger.info("query_rewritten", original=query[:100], rewritten=rewritten[:100])
                results = await self.search_engine.search(rewritten)
                return rewritten, results
        except Exception as e:
            logger.warning("query_rewrite_failed", error=str(e))

        return query, []

    async def _build_abstention_response(
        self,
        query: str,
        chunks: list[ChunkResult],
        thread_id: str,
        t0: float,
    ) -> ChatResponse:
        """Build abstention response when context is insufficient."""
        response = (
            "I don't have enough relevant information in my knowledge base to "
            "answer this question accurately. The documents I searched through "
            "didn't contain sufficient context to provide a reliable answer. "
            "Please try rephrasing your question or ingesting relevant documents first."
        )

        await self.working_memory.add_turn(thread_id, "user", query)
        await self.working_memory.add_turn(thread_id, "assistant", response)

        latency_ms = (time.perf_counter() - t0) * 1000
        metrics.increment("abstentions")

        return ChatResponse(
            response=response,
            sources=[],
            conversation_id=thread_id,
            crag_verdict="INSUFFICIENT",
            latency_ms=round(latency_ms, 1),
            query_type="abstention",
        )

    @staticmethod
    def _build_citations(chunks: list[ChunkResult]) -> list[SourceCitation]:
        """Build source citations from reranked chunks."""
        citations: list[SourceCitation] = []
        for chunk in chunks[:10]:  # Limit citations
            snippet = chunk.content[:300].strip()
            if len(chunk.content) > 300:
                snippet += "..."

            citations.append(SourceCitation(
                chunk_id=chunk.chunk_id,
                content_snippet=snippet,
                source_uri=chunk.source_uri or "unknown",
                source_type=chunk.source_type or "unknown",
                page=chunk.source_page,
                score=round(min(chunk.score, 1.0), 4),
            ))
        return citations
