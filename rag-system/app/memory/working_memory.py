"""Working memory: manages per-conversation turn history.

Stores and retrieves recent conversation turns for a thread, enforcing
a max_turns limit with FIFO eviction of oldest turn pairs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from app.infrastructure.database import Database
from app.infrastructure.observability import get_logger

logger = get_logger("memory.working_memory")


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    turn_index: int
    timestamp: str


class WorkingMemory:
    """Manages short-term conversation context per thread.

    Stores turn history as a JSON blob in the working_memory table.
    Enforces a maximum number of turns, evicting oldest user-assistant
    pairs when the limit is exceeded.
    """

    def __init__(self, db: Database, max_turns: int = 10) -> None:
        """Initialize working memory.

        Args:
            db: Database handle.
            max_turns: Maximum conversation turns to retain per thread.
        """
        self._db = db
        self._max_turns = max_turns

    async def get_or_create_thread(self, thread_id: str | None = None) -> str:
        """Get an existing thread or create a new one.

        Args:
            thread_id: Existing thread ID, or None to create a new one.

        Returns:
            The thread ID (existing or newly created).
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        row = await self._db.fetch_one(
            "SELECT thread_id FROM working_memory WHERE thread_id = ?",
            (thread_id,),
        )
        if row is None:
            state: dict = {"turns": [], "next_index": 0}
            await self._db.execute(
                "INSERT INTO working_memory (thread_id, state_blob) VALUES (?, ?)",
                (thread_id, json.dumps(state)),
            )
            logger.info("thread_created", thread_id=thread_id)
        return thread_id

    async def get_turns(self, thread_id: str) -> list[ConversationTurn]:
        """Get conversation turns for a thread.

        Args:
            thread_id: The thread identifier.

        Returns:
            List of ConversationTurn objects, most recent max_turns.
        """
        row = await self._db.fetch_one(
            "SELECT state_blob FROM working_memory WHERE thread_id = ?",
            (thread_id,),
        )
        if row is None:
            return []

        state = json.loads(row["state_blob"])
        turns = state.get("turns", [])

        return [
            ConversationTurn(
                role=t["role"],
                content=t["content"],
                turn_index=t["turn_index"],
                timestamp=t["timestamp"],
            )
            for t in turns[-self._max_turns:]
        ]

    async def add_turn(self, thread_id: str, role: str, content: str) -> None:
        """Add a conversation turn to a thread.

        Evicts oldest user-assistant pairs if exceeding max_turns.

        Args:
            thread_id: The thread identifier.
            role: 'user' or 'assistant'.
            content: Turn content text.
        """
        await self.get_or_create_thread(thread_id)

        row = await self._db.fetch_one(
            "SELECT state_blob FROM working_memory WHERE thread_id = ?",
            (thread_id,),
        )
        state = json.loads(row["state_blob"]) if row else {"turns": [], "next_index": 0}
        turns = state.get("turns", [])
        next_index = state.get("next_index", len(turns))

        turn = {
            "role": role,
            "content": content,
            "turn_index": next_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        turns.append(turn)
        next_index += 1

        # Evict oldest pairs if exceeding max_turns
        while len(turns) > self._max_turns:
            turns.pop(0)

        state["turns"] = turns
        state["next_index"] = next_index

        await self._db.execute(
            "UPDATE working_memory SET state_blob = ?, updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
            (json.dumps(state), thread_id),
        )

    async def format_for_prompt(self, thread_id: str) -> str:
        """Format turns as prompt-ready conversation history.

        Args:
            thread_id: The thread identifier.

        Returns:
            Formatted conversation history string.
        """
        turns = await self.get_turns(thread_id)
        if not turns:
            return ""

        parts: list[str] = []
        for turn in turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            parts.append(f"{role_label}: {turn.content}")

        return "\n".join(parts)

    async def clear(self, thread_id: str) -> None:
        """Clear working memory for a thread.

        Args:
            thread_id: The thread identifier.
        """
        state: dict = {"turns": [], "next_index": 0}
        await self._db.execute(
            "UPDATE working_memory SET state_blob = ?, updated_at = CURRENT_TIMESTAMP WHERE thread_id = ?",
            (json.dumps(state), thread_id),
        )
        logger.info("working_memory_cleared", thread_id=thread_id)
