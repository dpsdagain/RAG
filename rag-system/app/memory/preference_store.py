"""User preference store for persistent personalization.

Manages user preferences that get injected into every LLM prompt,
supporting activation, deactivation, and superseding of preferences.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.infrastructure.database import Database
from app.infrastructure.observability import get_logger

logger = get_logger("memory.preference_store")


class PreferenceStore:
    """Persistent user preference manager.

    Preferences influence LLM behavior by being injected into the
    system prompt. Supports categories like output_format, language,
    domain expertise, etc.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def add_preference(
        self,
        category: str,
        preference: str,
        supersedes_id: str | None = None,
    ) -> str:
        """Add a new preference, optionally superseding an existing one.

        Args:
            category: Category (e.g., 'output_format', 'language').
            preference: The preference text.
            supersedes_id: ID of a preference this supersedes (deactivates it).

        Returns:
            The new preference ID.
        """
        pref_id = str(uuid.uuid4())

        if supersedes_id:
            await self.deactivate_preference(supersedes_id)

        await self._db.execute(
            """INSERT INTO user_preferences (pref_id, category, preference, supersedes_id, is_active)
               VALUES (?, ?, ?, ?, 1)""",
            (pref_id, category, preference, supersedes_id),
        )
        logger.info(
            "preference_added",
            pref_id=pref_id,
            category=category,
            supersedes=supersedes_id,
        )
        return pref_id

    async def get_active_preferences(self) -> list[dict]:
        """Get all active preferences.

        Returns:
            List of preference dicts.
        """
        rows = await self._db.fetch_all(
            "SELECT * FROM user_preferences WHERE is_active = 1 ORDER BY category, created_at"
        )
        return [
            {
                "pref_id": row["pref_id"],
                "category": row["category"],
                "preference": row["preference"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def deactivate_preference(self, pref_id: str) -> None:
        """Deactivate a preference.

        Args:
            pref_id: The preference ID to deactivate.
        """
        await self._db.execute(
            "UPDATE user_preferences SET is_active = 0 WHERE pref_id = ?",
            (pref_id,),
        )
        logger.info("preference_deactivated", pref_id=pref_id)

    async def update_preference(
        self,
        pref_id: str,
        category: str,
        preference: str,
    ) -> str:
        """Update by creating a new preference and superseding the old one.

        Args:
            pref_id: ID of the preference to supersede.
            category: New category.
            preference: New preference text.

        Returns:
            The new preference ID.
        """
        return await self.add_preference(
            category=category,
            preference=preference,
            supersedes_id=pref_id,
        )

    async def format_for_prompt(self) -> str:
        """Format active preferences for prompt injection.

        Returns:
            Formatted preferences string, or empty string if none.
        """
        prefs = await self.get_active_preferences()
        if not prefs:
            return ""

        parts: list[str] = []
        for pref in prefs:
            parts.append(f"- [{pref['category']}]: {pref['preference']}")

        return "User Preferences:\n" + "\n".join(parts)
