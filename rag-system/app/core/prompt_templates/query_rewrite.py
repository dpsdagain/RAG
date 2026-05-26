"""Query rewrite prompt template for INSUFFICIENT CRAG verdicts."""
from __future__ import annotations

TEMPLATE = """The following query did not retrieve sufficient context. Rewrite it to be more specific and targeted while preserving the original intent.

Original Query: {original_query}

Available Context (what was retrieved):
{retrieval_context}

Respond with ONLY the improved query, no explanation."""


def build_messages(query: str, context: str) -> list[dict[str, str]]:
    """Build messages for query rewriting."""
    return [
        {
            "role": "user",
            "content": TEMPLATE.format(
                original_query=query,
                retrieval_context=context,
            ),
        },
    ]
