"""Parent context injector for enriching retrieved chunks.

For each reranked chunk, fetches its parent chunk (the full section) and
builds context payloads within a strict token budget.
"""
from __future__ import annotations

from app.infrastructure.database import ChunkResult, Database
from app.infrastructure.observability import get_logger, metrics
from app.models.embedder import Embedder

logger = get_logger("retrieval.parent_context")


class ParentContextInjector:
    """Enriches retrieved chunks with parent section context.

    Operates within a strict token budget to prevent LLM context overflow.
    Chunks are processed in rank order (highest score first), and parent
    content is only added when budget allows.
    """

    def __init__(self, db: Database, embedder: Embedder) -> None:
        self._db = db
        self._embedder = embedder

    async def inject_parent_context(
        self,
        chunks: list[ChunkResult],
        budget_tokens: int = 12000,
    ) -> list[dict]:
        """Enrich chunks with parent context within a token budget.

        Args:
            chunks: Reranked chunks sorted by score (highest first).
            budget_tokens: Maximum total tokens for the context payload.

        Returns:
            List of enriched chunk dicts with keys: chunk_id, content,
            parent_content, section_title, source_uri, source_page,
            score, token_count.
        """
        enriched: list[dict] = []
        cumulative_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.token_count or self._embedder.count_tokens(chunk.content)

            # Stop if even the chunk itself exceeds remaining budget
            if cumulative_tokens + chunk_tokens > budget_tokens:
                break

            entry: dict = {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "parent_content": None,
                "section_title": chunk.section_title,
                "source_uri": chunk.source_uri,
                "source_page": chunk.source_page,
                "source_type": chunk.source_type,
                "score": chunk.score,
                "token_count": chunk_tokens,
            }
            cumulative_tokens += chunk_tokens

            # Try to fetch and add parent content if budget allows
            if chunk.parent_chunk_id:
                try:
                    parent = await self._db.get_chunk_by_id(chunk.parent_chunk_id)
                    if parent is not None:
                        parent_tokens = parent.token_count or self._embedder.count_tokens(parent.content)
                        if cumulative_tokens + parent_tokens <= budget_tokens:
                            entry["parent_content"] = parent.content
                            entry["token_count"] += parent_tokens
                            cumulative_tokens += parent_tokens
                except Exception as e:
                    logger.warning(
                        "parent_fetch_failed",
                        chunk_id=chunk.chunk_id,
                        parent_id=chunk.parent_chunk_id,
                        error=str(e),
                    )

            enriched.append(entry)

        metrics.increment("parent_context_injections")
        logger.info(
            "parent_context_injected",
            chunks_enriched=len(enriched),
            total_tokens=cumulative_tokens,
            budget=budget_tokens,
        )
        return enriched

    @staticmethod
    def format_context_for_prompt(enriched_chunks: list[dict]) -> str:
        """Format enriched chunks into a string for LLM prompt injection.

        Args:
            enriched_chunks: Output of inject_parent_context().

        Returns:
            Formatted context string with source metadata and content.
        """
        if not enriched_chunks:
            return "No relevant context found."

        parts: list[str] = []
        for i, chunk in enumerate(enriched_chunks, 1):
            source = chunk.get("source_uri", "unknown")
            page = chunk.get("source_page")
            chunk_id = chunk.get("chunk_id", "?")
            section = chunk.get("section_title", "")

            header = f"[Source {i}: {source}"
            if page is not None:
                header += f" | Page: {page}"
            if section:
                header += f" | Section: {section}"
            header += f" | Chunk: {chunk_id}]"

            content = chunk.get("content", "")
            parent = chunk.get("parent_content")

            block = f"{header}\n{content}"
            if parent:
                block += f"\n[Parent Context:]\n{parent}"
            block += "\n---"
            parts.append(block)

        return "\n\n".join(parts)
