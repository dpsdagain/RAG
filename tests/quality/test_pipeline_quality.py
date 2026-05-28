"""End-to-end pipeline quality tests with a REAL LLM.

Skipped automatically when RAG_LLM__API_KEY isn't set. When active, each
test makes one real LLM call (CRAG and faithfulness are disabled to keep
cost predictable — those add 2 extra LLM calls per test).
"""
from __future__ import annotations

import pytest

from app.core.pipeline import RAGPipeline
from app.infrastructure.config import get_settings

from .golden_set import GOLDEN_SET


pytestmark = [
    pytest.mark.requires_db,
    pytest.mark.requires_embedder,
    pytest.mark.requires_llm,
    pytest.mark.slow,
]


@pytest.fixture
def quality_pipeline(ingested_corpus_db, real_embedder, real_llm_client):
    """Pipeline with extra cloud LLM calls disabled to keep tests cheap."""
    settings = get_settings()
    # Reduce cost: keep generation, skip the 3 helper LLM calls.
    settings.pipeline.crag_enabled = False
    settings.pipeline.decomposition_enabled = False
    settings.pipeline.faithfulness_check_enabled = False
    return RAGPipeline(
        db=ingested_corpus_db,
        embedder=real_embedder,
        llm=real_llm_client,
        settings=settings,
    )


def _normalize_for_match(text: str) -> str:
    """Normalize LLM output for substring matching.

    Real LLMs love to emit Unicode punctuation — non-breaking hyphens
    (U+2011), narrow no-break space (U+202F), em-dashes, smart quotes —
    that defeat naive substring tests. Map them to ASCII first.
    """
    replacements = {
        "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
        " ": " ", " ": " ", " ": " ",
        "“": '"', "”": '"', "‘": "'", "’": "'",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.lower()


@pytest.mark.parametrize("case", GOLDEN_SET, ids=[c["id"] for c in GOLDEN_SET])
async def test_answer_mentions_expected_facts(case, quality_pipeline):
    """The generated answer should reference at least one expected fact."""
    result = await quality_pipeline.execute(case["query"])
    answer = _normalize_for_match(result.response)

    matched = [sub for sub in case["must_generate"] if sub in answer]
    assert matched, (
        f"answer for {case['query']!r} contained none of {case['must_generate']!r}\n"
        f"answer was: {result.response[:400]!r}"
    )

    # Quality guards
    assert len(result.response) > 30, "answer suspiciously short"
    assert result.sources, "expected at least one source citation"


async def test_answer_cites_sources(quality_pipeline):
    """Successful generation must come back with at least one source."""
    result = await quality_pipeline.execute("What is the chunking strategy?")
    assert result.sources
    for src in result.sources:
        # SourceCitation invariants
        assert 0.0 <= src.score <= 1.0
        assert len(src.content_snippet) <= 300


async def test_cache_returns_same_response_on_repeat(quality_pipeline):
    """Repeating a query should return the cached response, not call the LLM again."""
    first = await quality_pipeline.execute("Which embedding model does the system use?")
    second = await quality_pipeline.execute("Which embedding model does the system use?")
    assert second.crag_verdict == "CACHED"
    assert second.response == first.response


async def test_streaming_yields_sources_then_tokens_then_done(quality_pipeline):
    """execute_stream must emit sources first, then tokens, then done."""
    events: list[dict] = []
    async for evt in quality_pipeline.execute_stream("How does hybrid search work?"):
        events.append(evt)

    event_names = [e["event"] for e in events]
    assert "token" in event_names
    assert "done" in event_names
    done_idx = event_names.index("done")
    first_token_idx = event_names.index("token")
    if "sources" in event_names:
        sources_idx = event_names.index("sources")
        # Sources must be the first non-trivial event, before any token.
        assert sources_idx < first_token_idx
    assert first_token_idx < done_idx
