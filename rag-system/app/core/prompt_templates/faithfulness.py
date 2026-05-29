"""Faithfulness verification prompt template.

Asks the LLM judge to enumerate EVERY factual claim in the response
and tag each as SUPPORTED or UNSUPPORTED against the source chunks.
This per-claim breakdown is what enables a 0-1 RAGAS-style score
(supported / total) instead of a binary pass/fail.
"""
from __future__ import annotations

TEMPLATE = """You are evaluating whether a generated response is grounded in the provided source chunks.

Your job: enumerate EVERY factual claim in the generated response and label each one.

For EACH claim, output exactly one line in this format:
  - SUPPORTED: <restate the claim in one short sentence>
  - UNSUPPORTED: <restate the claim in one short sentence>

A claim is SUPPORTED only if the source chunks contain explicit information that backs it.
A claim is UNSUPPORTED if it goes beyond what the sources say, contradicts them, or is fabricated.
Generic conversational filler ("I will help you", "let me explain") is NOT a claim — skip it.

If after enumeration EVERY claim is supported, you may instead reply with the single line:
  ALL CLAIMS VERIFIED

Source Chunks:
{retrieved_chunks}

Generated Response:
{generated_response}

Now list every claim, one per line:"""


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
