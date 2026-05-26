"""Faithfulness verification prompt template."""
from __future__ import annotations

TEMPLATE = """Compare the generated response against the source chunks. Identify any claims in the response that are NOT supported by the provided sources.

Source Chunks:
{retrieved_chunks}

Generated Response:
{generated_response}

If all claims are supported, respond with: ALL CLAIMS VERIFIED
Otherwise, list each unsupported claim on a separate line, prefixed with "- UNSUPPORTED: "."""


def build_messages(response: str, chunks_text: str) -> list[dict[str, str]]:
    """Build messages for faithfulness verification."""
    return [
        {
            "role": "user",
            "content": TEMPLATE.format(
                retrieved_chunks=chunks_text,
                generated_response=response,
            ),
        },
    ]
