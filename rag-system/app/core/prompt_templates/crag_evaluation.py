"""CRAG quality evaluation prompt template."""
from __future__ import annotations

TEMPLATE = """Evaluate whether the following retrieved chunks are sufficient to answer the user's query.

Query: {query}

Retrieved Chunks:
{top_15_chunks}

Respond with exactly one word:
- SUFFICIENT: The chunks contain clear, direct answers to the query.
- PARTIAL: The chunks contain some relevant information but may not fully answer the query.
- INSUFFICIENT: The chunks do not contain relevant information to answer the query."""


def build_messages(query: str, chunks_text: str) -> list[dict[str, str]]:
    """Build messages for CRAG quality evaluation."""
    return [
        {
            "role": "user",
            "content": TEMPLATE.format(query=query, top_15_chunks=chunks_text),
        },
    ]
