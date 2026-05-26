"""CRAG (Corrective RAG) quality gate."""
from __future__ import annotations

from app.core.prompt_templates import crag_evaluation
from app.infrastructure.database import ChunkResult
from app.infrastructure.observability import get_logger, metrics
from app.models.llm_client import LLMClient

logger = get_logger("core.crag_gate")


class CRAGGate:
    """Evaluates retrieval quality and signals when retry is needed.

    Returns SUFFICIENT, PARTIAL, or INSUFFICIENT. On LLM failure,
    fails open with PARTIAL to avoid blocking the pipeline.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def evaluate(self, query: str, chunks: list[ChunkResult]) -> str:
        """Evaluate retrieval quality.

        Args:
            query: The user query.
            chunks: Retrieved and reranked chunks.

        Returns:
            'SUFFICIENT', 'PARTIAL', or 'INSUFFICIENT'.
        """
        if not chunks:
            return "INSUFFICIENT"

        chunks_text = "\n\n".join(
            f"[Chunk {i+1}]: {c.content[:500]}" for i, c in enumerate(chunks[:15])
        )

        messages = crag_evaluation.build_messages(query, chunks_text)

        try:
            response = await self._llm.generate(messages, temperature=0.0, max_tokens=20)
            verdict = response.strip().upper()

            # Normalize response
            if "SUFFICIENT" in verdict and "INSUFFICIENT" not in verdict:
                result = "SUFFICIENT"
            elif "INSUFFICIENT" in verdict:
                result = "INSUFFICIENT"
            else:
                result = "PARTIAL"

            metrics.increment(f"crag_{result.lower()}")
            logger.info("crag_evaluated", verdict=result, chunks=len(chunks))
            return result

        except Exception as e:
            logger.warning("crag_evaluation_failed", error=str(e))
            metrics.increment("crag_errors")
            return "PARTIAL"  # Fail open
