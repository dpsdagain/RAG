"""Query decomposition prompt template."""
from __future__ import annotations

TEMPLATE = """Break the following complex query into 1-3 simpler, self-contained sub-queries that together cover all aspects of the original question.

Return ONLY a JSON array of strings. Example: ["sub-query 1", "sub-query 2"]

Query: {query}"""


def build_messages(query: str) -> list[dict[str, str]]:
    """Build messages for query decomposition."""
    return [{"role": "user", "content": TEMPLATE.format(query=query)}]
