"""Cloud VLM OCR parser using Gemini or OpenAI vision APIs.

Sends document page images to cloud vision-language models for
high-quality text extraction from complex or degraded scans.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import httpx

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.config import get_settings
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.ocr_cloud")

EXTRACTION_PROMPT = (
    "Extract all text from this document image. Preserve formatting, "
    "tables, and structure. Output in Markdown format."
)


class OCRCloudParser(BaseParser):
    """Parse documents using cloud Vision-Language Models.

    Supports Gemini Flash and GPT-4o-mini for high-quality OCR
    of complex or degraded document scans.
    """

    def __init__(self, provider: str | None = None, api_key: str | None = None) -> None:
        settings = get_settings()
        self._provider = provider or settings.cloud_vlm.provider
        self._api_key = api_key or settings.cloud_vlm.api_key

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        if not self._api_key:
            raise ValueError("Cloud VLM API key is required for OCR")

        try:
            import pymupdf
        except ImportError as e:
            raise ImportError("pymupdf is required for page rendering") from e

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        doc = pymupdf.open(str(source))
        pages: list[PageContent] = []
        all_text: list[str] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for page_num in range(len(doc)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                img_b64 = base64.b64encode(img_bytes).decode()

                try:
                    text = await self._call_vlm(client, img_b64)
                except Exception as e:
                    logger.warning("ocr_cloud_page_failed", page=page_num, error=str(e))
                    text = ""

                pages.append(PageContent(page_number=page_num + 1, text=text))
                all_text.append(text)

        doc.close()

        content = "\n\n".join(all_text)
        logger.info("ocr_cloud_parsed", source=str(source), pages=len(pages))

        return ParsedDocument(
            content=content,
            title=source.stem.replace("_", " ").title(),
            pages=pages,
            metadata={"parser": self.parser_name(), "quality_score": 0.9},
            source_type="pdf",
        )

    async def _call_vlm(self, client: httpx.AsyncClient, image_b64: str) -> str:
        """Call the VLM API with an image for text extraction."""
        if self._provider == "gemini":
            return await self._call_gemini(client, image_b64)
        elif self._provider == "openai":
            return await self._call_openai(client, image_b64)
        else:
            raise ValueError(f"Unsupported VLM provider: {self._provider}")

    async def _call_gemini(self, client: httpx.AsyncClient, image_b64: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self._api_key}"
        body = {
            "contents": [{
                "parts": [
                    {"text": EXTRACTION_PROMPT},
                    {"inline_data": {"mime_type": "image/png", "data": image_b64}},
                ],
            }],
        }
        resp = await client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            return "".join(p.get("text", "") for p in parts)
        return ""

    async def _call_openai(self, client: httpx.AsyncClient, image_b64: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}"}
        body = {
            "model": "gpt-4o-mini",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": EXTRACTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
            "max_tokens": 4096,
        }
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")

    def supported_extensions(self) -> list[str]:
        return [".pdf", ".png", ".jpg", ".jpeg"]

    def parser_name(self) -> str:
        return f"ocr_cloud_{self._provider}"
