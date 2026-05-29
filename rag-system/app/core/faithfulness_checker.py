"""Post-generation faithfulness checker with RAGAS-style 0–1 scoring.

The check runs the LLM-as-judge on (response, sources) → returns:
  * verified         — bool, True only if no unsupported claims
  * unsupported_claims — list of claim strings the LLM flagged
  * score            — float in [0, 1], 1 = fully grounded, 0 = none of
                       the claims supported. RAGAS-style faithfulness
                       metric: (#supported_claims) / (#total_claims).
  * raw_result       — the raw LLM output for debugging
  * latency_ms       — how long the judge call took

Scores are also persisted to disk (one JSONL row per check) so drift can
be inspected over time without spinning up a separate eval pipeline.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.prompt_templates import faithfulness as faith_prompt
from app.infrastructure.observability import get_logger, metrics
from app.models.llm_client import LLMClient

logger = get_logger("core.faithfulness_checker")


# Where the per-check audit log lives. Each line is one JSON dict.
# Used for offline drift analysis ("did our faithfulness pass rate drop
# this week?") — separate from request_log because we want focused
# scoring data, not full request metadata.
_DEFAULT_LOG_PATH = Path("./data/faithfulness_log.jsonl")


# A "claim" sentinel marker the prompt asks the LLM to emit for each
# claim it identifies in the response. The judge prompt template tells
# the LLM to output lines like:
#     - SUPPORTED: <claim text>
#     - UNSUPPORTED: <claim text>
# We count supported + unsupported lines to derive a 0..1 score.
_SUPPORTED_RE = re.compile(r"^\s*[-*]?\s*SUPPORTED\s*:", re.MULTILINE | re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"^\s*[-*]?\s*UNSUPPORTED\s*:", re.MULTILINE | re.IGNORECASE)


class FaithfulnessChecker:
    """Verifies that generated responses are grounded in source chunks.

    Produces a RAGAS-style faithfulness score in [0, 1] in addition to
    the pass/fail flag. Fails CLOSED on LLM errors (returns score=0,
    verified=False) so a broken judge can never silently approve an
    unverified answer.
    """

    def __init__(
        self, llm: LLMClient, log_path: str | Path | None = None,
    ) -> None:
        self._llm = llm
        self._log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH

    async def check(self, response: str, chunks: list[dict]) -> dict:
        """Check response faithfulness against source chunks.

        Args:
            response: The generated response text.
            chunks: Enriched chunk dicts from parent context injector.

        Returns:
            Dict with: verified (bool), score (float in [0,1]),
            unsupported_claims (list[str]), supported_count (int),
            total_claims (int), raw_result (str), latency_ms (float).
        """
        if not chunks or not response:
            result = {
                "verified": True,
                "score": 1.0,
                "unsupported_claims": [],
                "supported_count": 0,
                "total_claims": 0,
                "raw_result": "SKIPPED",
                "latency_ms": 0.0,
            }
            self._append_log(response, result)
            return result

        chunks_text = "\n\n".join(
            f"[Source {i+1}]: {c.get('content', '')[:500]}"
            for i, c in enumerate(chunks[:10])
        )

        messages = faith_prompt.build_messages(response, chunks_text)

        t0 = time.perf_counter()
        try:
            raw = await self._llm.generate(
                messages, temperature=0.0, max_tokens=500
            )
            latency_ms = (time.perf_counter() - t0) * 1000
            result_text = raw.strip()

            # Fast-path: explicit "ALL CLAIMS VERIFIED" sentinel
            if "ALL CLAIMS VERIFIED" in result_text.upper():
                metrics.increment("faithfulness_verified")
                metrics.record_latency("faithfulness", latency_ms)
                result = {
                    "verified": True,
                    "score": 1.0,
                    "unsupported_claims": [],
                    "supported_count": self._count_supported(result_text),
                    "total_claims": self._count_supported(result_text),
                    "raw_result": result_text,
                    "latency_ms": round(latency_ms, 1),
                }
                self._append_log(response, result)
                logger.info("faithfulness_verified", score=1.0)
                return result

            # Parse the claim list. We count SUPPORTED + UNSUPPORTED lines
            # to derive a proper RAGAS-style score:
            #   score = supported / (supported + unsupported)
            unsupported = self._parse_unsupported(result_text)
            supported_count = self._count_supported(result_text)
            total = supported_count + len(unsupported)

            # If the LLM didn't emit any SUPPORTED/UNSUPPORTED tags but
            # the verdict isn't "ALL CLAIMS VERIFIED" either, we treat it
            # as "judge couldn't parse the response" — score conservatively
            # at 0.5 to flag the ambiguity rather than fabricate a number.
            if total == 0:
                score = 0.5
                verdict = "AMBIGUOUS"
            else:
                score = supported_count / total
                verdict = "ISSUES_FOUND" if unsupported else "VERIFIED"

            metrics.record_latency("faithfulness", latency_ms)
            if unsupported:
                metrics.increment("faithfulness_issues")
            else:
                metrics.increment("faithfulness_verified")

            result = {
                "verified": len(unsupported) == 0,
                "score": round(score, 4),
                "unsupported_claims": unsupported,
                "supported_count": supported_count,
                "total_claims": total,
                "raw_result": result_text,
                "latency_ms": round(latency_ms, 1),
            }
            self._append_log(response, result)
            logger.info(
                "faithfulness_scored",
                verdict=verdict,
                score=result["score"],
                supported=supported_count,
                total=total,
            )
            return result

        except Exception as e:
            # Fail CLOSED: a safety gate that errors must not silently
            # claim the response is verified.
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.warning("faithfulness_check_failed", error=str(e))
            metrics.increment("faithfulness_errors")
            result = {
                "verified": False,
                "score": 0.0,
                "unsupported_claims": [],
                "supported_count": 0,
                "total_claims": 0,
                "raw_result": "ERROR",
                "latency_ms": round(latency_ms, 1),
            }
            self._append_log(response, result)
            return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_supported(text: str) -> int:
        """Count SUPPORTED: lines in the judge's output."""
        return len(_SUPPORTED_RE.findall(text))

    @staticmethod
    def _parse_unsupported(text: str) -> list[str]:
        """Extract the unsupported claim texts from the judge's output."""
        unsupported: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Match "- UNSUPPORTED: ..." / "* UNSUPPORTED: ..." / "UNSUPPORTED: ..."
            if _UNSUPPORTED_RE.match(line):
                # Strip the prefix + leading bullet to get the claim text
                claim = re.sub(
                    r"^\s*[-*]?\s*UNSUPPORTED\s*:\s*", "",
                    stripped, flags=re.IGNORECASE,
                ).strip()
                if claim:
                    unsupported.append(claim)
        return unsupported

    def _append_log(self, response: str, result: dict) -> None:
        """Append one JSONL row per check for offline drift analysis.

        Best-effort: failures are logged but never propagated — a broken
        audit log must not block live requests.
        """
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "score": result.get("score"),
                "verified": result.get("verified"),
                "supported_count": result.get("supported_count"),
                "total_claims": result.get("total_claims"),
                "latency_ms": result.get("latency_ms"),
                "response_preview": response[:200],
                "raw_result_preview": (result.get("raw_result") or "")[:300],
            }
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("faithfulness_log_append_failed", error=str(e))
