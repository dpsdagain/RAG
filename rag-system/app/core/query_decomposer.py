"""Query decomposer for complex multi-topic queries."""
from __future__ import annotations

import json

from app.core.prompt_templates import decomposition
from app.infrastructure.observability import get_logger, metrics
from app.models.llm_client import LLMClient

logger = get_logger("core.query_decomposer")


class QueryDecomposer:
    """Breaks complex queries into 1-3 simpler sub-queries.

    On failure, gracefully returns the original query as a single-element list.
    """

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def decompose(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries.

        Args:
            query: The complex user query.

        Returns:
            List of 1-3 sub-query strings.
        """
        messages = decomposition.build_messages(query)

        try:
            response = await self._llm.generate(messages, temperature=0.0, max_tokens=300)

            # Parse JSON array from response
            # Handle responses that might have markdown code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

            sub_queries = json.loads(cleaned)

            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                # Cap at 3 sub-queries
                result = sub_queries[:3]
                metrics.increment("queries_decomposed")
                logger.info(
                    "query_decomposed",
                    original=query[:100],
                    sub_queries=len(result),
                )
                return result

            logger.warning("decomposition_invalid_format", response=response[:200])
            return [query]

        except json.JSONDecodeError:
            logger.warning("decomposition_parse_failed", response=response[:200] if 'response' in dir() else "no response")
            return [query]
        except Exception as e:
            logger.warning("decomposition_failed", error=str(e))
            return [query]
