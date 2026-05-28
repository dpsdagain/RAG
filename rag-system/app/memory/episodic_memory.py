"""Episodic memory: long-term conversation recall via semantic search.

Summarizes completed conversations using the LLM, embeds the summary,
and stores it for future semantic retrieval of relevant past context.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.infrastructure.database import Database
from app.infrastructure.observability import get_logger, metrics
from app.models.embedder import Embedder
from app.models.llm_client import LLMClient

logger = get_logger("memory.episodic_memory")

SUMMARIZE_PROMPT = (
    "Summarize this conversation in 2-3 sentences focusing on key facts "
    "discussed and decisions made. Be concise and factual.\n\n"
    "Conversation:\n{conversation}"
)


class EpisodicMemory:
    """Long-term memory that stores and recalls past conversation summaries.

    Uses LLM summarization and vector search to enable the system to
    recall relevant context from previous interactions.
    """

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        llm: LLMClient,
        top_k: int = 3,
    ) -> None:
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._top_k = top_k

    async def store_conversation(
        self,
        conversation_id: str,
        turns: list[dict],
    ) -> None:
        """Summarize and store a completed conversation.

        Args:
            conversation_id: Unique conversation ID.
            turns: List of turn dicts with 'role' and 'content' keys.
        """
        if not turns:
            return

        # Format conversation for summarization
        formatted = "\n".join(
            f"{'User' if t.get('role') == 'user' else 'Assistant'}: {t.get('content', '')}"
            for t in turns
        )

        # Generate summary via LLM
        try:
            summary = await self._llm.generate(
                messages=[{
                    "role": "user",
                    "content": SUMMARIZE_PROMPT.format(conversation=formatted),
                }],
                temperature=0.1,
                max_tokens=200,
            )
        except Exception as e:
            logger.error("episodic_summarize_failed", error=str(e))
            summary = f"Conversation with {len(turns)} turns."

        # Embed summary for future semantic recall
        summary_embedding = self._embedder.embed(summary)

        # Store in conversations table
        await self._db.execute(
            """INSERT OR REPLACE INTO conversations (conversation_id, summary, turn_count, started_at)
               VALUES (?, ?, ?, ?)""",
            (conversation_id, summary, len(turns), datetime.now(timezone.utc).isoformat()),
        )

        # Store embedding in conversations_vec
        import struct
        blob = struct.pack(f"{len(summary_embedding)}f", *summary_embedding)
        try:
            await self._db.execute(
                "INSERT OR REPLACE INTO conversations_vec (conversation_id, summary_embedding) VALUES (?, ?)",
                (conversation_id, blob),
            )
        except Exception as e:
            logger.warning("episodic_vec_store_failed", error=str(e))

        metrics.increment("episodic_conversations_stored")
        logger.info(
            "episodic_conversation_stored",
            conversation_id=conversation_id,
            turn_count=len(turns),
            summary_length=len(summary),
        )

    async def recall(self, query_embedding: list[float]) -> list[dict]:
        """Recall relevant past conversations by semantic similarity.

        Same K-NN-as-subquery pattern as Database.vector_search — the vec0
        virtual table needs to see a bare `MATCH ? LIMIT ?` so it can do
        the K-NN; joining + filtering in an outer query keeps it happy.
        """
        import struct
        blob = struct.pack(f"{len(query_embedding)}f", *query_embedding)
        knn_limit = max(self._top_k * 2, self._top_k + 5)

        try:
            rows = await self._db.fetch_all(
                """SELECT cv.conversation_id, cv.distance, c.summary,
                          c.turn_count, c.started_at
                   FROM (
                       SELECT conversation_id, distance
                       FROM conversations_vec
                       WHERE summary_embedding MATCH ?
                       ORDER BY distance
                       LIMIT ?
                   ) cv
                   JOIN conversations c ON c.conversation_id = cv.conversation_id
                   ORDER BY cv.distance
                   LIMIT ?""",
                (blob, knn_limit, self._top_k),
            )
        except Exception as e:
            logger.warning("episodic_recall_failed", error=str(e))
            return []

        results: list[dict] = []
        for row in rows:
            # sqlite-vec L2² distance → [0, 1] similarity (same formula as
            # Database.vector_search for L2-normalized embeddings).
            distance = float(row["distance"])
            similarity = max(0.0, min(1.0, 1.0 - distance / 2.0))
            results.append({
                "conversation_id": row["conversation_id"],
                "summary": row["summary"],
                "similarity": round(similarity, 4),
                "turn_count": row["turn_count"],
                "started_at": row["started_at"],
            })

        metrics.increment("episodic_recalls")
        return results

    async def format_for_prompt(self, query_embedding: list[float]) -> str:
        """Format recalled memories for prompt injection.

        Args:
            query_embedding: Embedding of the current query.

        Returns:
            Formatted string of relevant past conversation summaries.
        """
        memories = await self.recall(query_embedding)
        if not memories:
            return ""

        parts: list[str] = []
        for mem in memories:
            parts.append(
                f"[Past conversation ({mem['started_at']}): {mem['summary']}]"
            )

        return "\n".join(parts)

    async def get_all_conversations(self, limit: int = 50) -> list[dict]:
        """List all past conversations with summaries.

        Args:
            limit: Maximum number of conversations to return.

        Returns:
            List of conversation dicts.
        """
        rows = await self._db.fetch_all(
            "SELECT * FROM conversations ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [
            {
                "conversation_id": row["conversation_id"],
                "summary": row["summary"],
                "turn_count": row["turn_count"],
                "started_at": row["started_at"],
            }
            for row in rows
        ]
