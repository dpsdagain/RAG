"""Cloud PDF parser using LlamaParse API.

Fallback parser for complex/scanned PDFs when local parsers fail.
Requires a LlamaParse API key configured in settings.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.config import get_settings
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.pdf_cloud")

LLAMAPARSE_API_URL = "https://api.cloud.llamaindex.ai/api/parsing"


class ConfigurationError(Exception):
    """Raised when required configuration is missing."""
    pass


class PDFCloudParser(BaseParser):
    """Parse PDFs via LlamaParse cloud API.

    Uploads the PDF, polls for completion, and downloads the parsed
    Markdown output. Includes retry logic with exponential backoff.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or get_settings().llamaparse.api_key

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        """Parse a PDF via LlamaParse cloud API.

        Args:
            source: Path to the PDF file.

        Returns:
            ParsedDocument with Markdown content.

        Raises:
            ConfigurationError: If API key is not configured.
        """
        if not self._api_key:
            raise ConfigurationError(
                "LlamaParse API key is required. Set it in configs/config.yaml "
                "under llamaparse.api_key"
            )

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "accept": "application/json",
        }

        max_retries = 2
        backoff = 2.0

        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    # Upload file
                    with open(source, "rb") as f:
                        resp = await client.post(
                            f"{LLAMAPARSE_API_URL}/upload",
                            headers=headers,
                            files={"file": (source.name, f, "application/pdf")},
                            data={"result_type": "markdown"},
                        )
                        resp.raise_for_status()
                        job_data = resp.json()
                        job_id = job_data.get("id")

                    # Poll for completion
                    for _ in range(60):  # Max 5 minutes
                        await asyncio.sleep(5)
                        status_resp = await client.get(
                            f"{LLAMAPARSE_API_URL}/job/{job_id}",
                            headers=headers,
                        )
                        status_resp.raise_for_status()
                        status_data = status_resp.json()

                        if status_data.get("status") == "SUCCESS":
                            break
                        elif status_data.get("status") in ("ERROR", "FAILED"):
                            raise RuntimeError(
                                f"LlamaParse job failed: {status_data.get('error', 'Unknown error')}"
                            )

                    # Download result
                    result_resp = await client.get(
                        f"{LLAMAPARSE_API_URL}/job/{job_id}/result/markdown",
                        headers=headers,
                    )
                    result_resp.raise_for_status()
                    result_data = result_resp.json()
                    md_content = result_data.get("markdown", "")

                    logger.info(
                        "pdf_cloud_parsed",
                        source=str(source),
                        job_id=job_id,
                        content_length=len(md_content),
                    )

                    return ParsedDocument(
                        content=md_content,
                        title=source.stem.replace("_", " ").title(),
                        pages=[PageContent(page_number=1, text=md_content)],
                        metadata={
                            "parser": self.parser_name(),
                            "job_id": job_id,
                            "quality_score": 0.85,
                        },
                        source_type="pdf",
                    )

            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                if attempt < max_retries:
                    wait = backoff * (2 ** attempt)
                    logger.warning(
                        "pdf_cloud_retry",
                        attempt=attempt + 1,
                        error=str(e),
                        backoff=wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(f"LlamaParse failed after {max_retries + 1} attempts: {e}") from e

        raise RuntimeError("LlamaParse parsing failed unexpectedly")

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parser_name(self) -> str:
        return "llamaparse"
