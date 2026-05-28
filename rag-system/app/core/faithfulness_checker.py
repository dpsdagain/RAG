"""Post-generation faithfulness checker."""
from __future__ import annotations

from app.core.prompt_templates import faithfulness as faith_prompt
from app.infrastructure.observability import get_logger, metrics
from app.models.llm_client import LLMClient

logger = get_logger("core.faithfulness_checker")


class FaithfulnessChecker:
    """Verifies that generated responses are grounded in source chunks.

    On LLM failure, returns a SKIPPED result to avoid blocking.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def check(self, response: str, chunks: list[dict]) -> dict:
        """Check response faithfulness against source chunks.

        Args:
            response: The generated response text.
            chunks: Enriched chunk dicts from parent context injector.

        Returns:
            Dict with keys: verified (bool), unsupported_claims (list[str]), raw_result (str).
        """
        if not chunks or not response:
            return {"verified": True, "unsupported_claims": [], "raw_result": "SKIPPED"}

        chunks_text = "\n\n".join(
            f"[Source {i+1}]: {c.get('content', '')[:500]}"
            for i, c in enumerate(chunks[:10])
        )

        messages = faith_prompt.build_messages(response, chunks_text)

        try:
            result = await self._llm.generate(messages, temperature=0.0, max_tokens=500)
            result_text = result.strip()

            if "ALL CLAIMS VERIFIED" in result_text.upper():
                metrics.increment("faithfulness_verified")
                logger.info("faithfulness_verified")
                return {
                    "verified": True,
                    "unsupported_claims": [],
                    "raw_result": result_text,
                }

            # Parse unsupported claims
            unsupported: list[str] = []
            for line in result_text.split("\n"):
                line = line.strip()
                if line.startswith("- UNSUPPORTED:") or line.startswith("UNSUPPORTED:"):
                    claim = line.replace("- UNSUPPORTED:", "").replace("UNSUPPORTED:", "").strip()
                    if claim:
                        unsupported.append(claim)

            metrics.increment("faithfulness_issues")
            logger.warning(
                "faithfulness_issues_found",
                unsupported_count=len(unsupported),
            )

            return {
                "verified": len(unsupported) == 0,
                "unsupported_claims": unsupported,
                "raw_result": result_text,
            }

        except Exception as e:
            # Fail CLOSED: a safety gate that errors out must not silently
            # claim the response is verified. Surface the failure so the
            # pipeline can warn the user instead of pretending all is well.
            logger.warning("faithfulness_check_failed", error=str(e))
            metrics.increment("faithfulness_errors")
            return {
                "verified": False,
                "unsupported_claims": [],
                "raw_result": "ERROR",
            }
