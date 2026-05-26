"""Procedural rule engine for query-triggered behavior rules.

Matches incoming queries against regex trigger patterns and injects
matching rule instructions into the LLM prompt. Rules can be loaded
from YAML or added dynamically.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import yaml

from app.infrastructure.database import Database
from app.infrastructure.observability import get_logger

logger = get_logger("memory.procedural_rules")


class ProceduralRuleEngine:
    """Matches queries against procedural rules and injects instructions.

    Rules define trigger patterns (regex) and instruction text that
    gets prepended to the LLM prompt when a pattern matches. Useful
    for enforcing output formats, citation styles, and domain-specific
    behavior without modifying the core pipeline.
    """

    def __init__(self, db: Database, rules_file: str | Path | None = None) -> None:
        self._db = db
        self._rules_file = rules_file

    async def load_rules_from_yaml(self, yaml_path: str | Path) -> int:
        """Load rules from a YAML file into the database.

        Args:
            yaml_path: Path to the YAML rules file.

        Returns:
            Number of rules loaded.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning("rules_file_not_found", path=str(yaml_path))
            return 0

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        rules = data.get("rules", [])
        count = 0

        for rule_data in rules:
            trigger = rule_data.get("trigger_pattern", "")
            text = rule_data.get("rule_text", "")
            priority = rule_data.get("priority", 0)

            if not trigger or not text:
                continue

            rule_id = rule_data.get("rule_id", str(uuid.uuid4()))

            # Upsert: check if rule_id already exists
            existing = await self._db.fetch_one(
                "SELECT rule_id FROM procedural_rules WHERE rule_id = ?",
                (rule_id,),
            )
            if existing:
                await self._db.execute(
                    """UPDATE procedural_rules
                       SET trigger_pattern = ?, rule_text = ?, priority = ?, is_active = 1
                       WHERE rule_id = ?""",
                    (trigger, text, priority, rule_id),
                )
            else:
                await self._db.execute(
                    """INSERT INTO procedural_rules (rule_id, trigger_pattern, rule_text, priority, is_active)
                       VALUES (?, ?, ?, ?, 1)""",
                    (rule_id, trigger, text, priority),
                )
            count += 1

        logger.info("procedural_rules_loaded", count=count, source=str(yaml_path))
        return count

    async def match_rules(self, query: str) -> list[dict]:
        """Match a query against active rule trigger patterns.

        Args:
            query: The incoming user query.

        Returns:
            List of matching rule dicts sorted by priority (desc).
        """
        rows = await self._db.fetch_all(
            "SELECT * FROM procedural_rules WHERE is_active = 1 ORDER BY priority DESC"
        )

        matches: list[dict] = []
        for row in rows:
            pattern = row["trigger_pattern"]
            try:
                if re.search(pattern, query, re.IGNORECASE):
                    matches.append({
                        "rule_id": row["rule_id"],
                        "trigger_pattern": pattern,
                        "rule_text": row["rule_text"],
                        "priority": row["priority"],
                    })
            except re.error as e:
                logger.warning(
                    "invalid_rule_pattern",
                    rule_id=row["rule_id"],
                    pattern=pattern,
                    error=str(e),
                )

        return matches

    async def format_for_prompt(self, query: str) -> str:
        """Format matching rules for prompt injection.

        Args:
            query: The incoming user query.

        Returns:
            Formatted rules string, or empty string if no matches.
        """
        rules = await self.match_rules(query)
        if not rules:
            return ""

        parts: list[str] = []
        for rule in rules:
            parts.append(f"[Rule]: {rule['rule_text']}")

        return "Procedural Instructions:\n" + "\n".join(parts)

    async def add_rule(
        self,
        trigger_pattern: str,
        rule_text: str,
        priority: int = 0,
    ) -> str:
        """Add a new procedural rule.

        Args:
            trigger_pattern: Regex pattern for query matching.
            rule_text: Instruction text to inject on match.
            priority: Higher priority rules are matched first.

        Returns:
            The new rule ID.
        """
        rule_id = str(uuid.uuid4())

        # Validate regex
        try:
            re.compile(trigger_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        await self._db.execute(
            """INSERT INTO procedural_rules (rule_id, trigger_pattern, rule_text, priority, is_active)
               VALUES (?, ?, ?, ?, 1)""",
            (rule_id, trigger_pattern, rule_text, priority),
        )
        logger.info("procedural_rule_added", rule_id=rule_id, pattern=trigger_pattern)
        return rule_id

    async def get_all_rules(self) -> list[dict]:
        """Get all procedural rules.

        Returns:
            List of rule dicts.
        """
        rows = await self._db.fetch_all(
            "SELECT * FROM procedural_rules ORDER BY priority DESC, created_at"
        )
        return [
            {
                "rule_id": row["rule_id"],
                "trigger_pattern": row["trigger_pattern"],
                "rule_text": row["rule_text"],
                "priority": row["priority"],
                "is_active": bool(row["is_active"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    async def deactivate_rule(self, rule_id: str) -> None:
        """Deactivate a procedural rule.

        Args:
            rule_id: The rule ID to deactivate.
        """
        await self._db.execute(
            "UPDATE procedural_rules SET is_active = 0 WHERE rule_id = ?",
            (rule_id,),
        )
        logger.info("procedural_rule_deactivated", rule_id=rule_id)
