"""Web crawler parser using Crawl4AI for URL ingestion.

Extracts clean Markdown content from web pages, including
JavaScript-rendered content when possible.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.web_crawl4ai")

_crawl4ai_available = True
try:
    from crawl4ai import AsyncWebCrawler  # type: ignore[import-untyped]
except ImportError:
    _crawl4ai_available = False


class WebCrawl4AIParser(BaseParser):
    """Parse web pages using Crawl4AI.

    Crawls a URL, extracts clean Markdown content with metadata.
    Optional dependency — raises ImportError if not installed.
    """

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        if not _crawl4ai_available:
            raise ImportError(
                "crawl4ai is not installed. Install with: pip install crawl4ai"
            )

        url = str(source)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc

        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url)

                md_content = result.markdown if hasattr(result, 'markdown') else ""
                title = result.metadata.get("title", "") if hasattr(result, 'metadata') and result.metadata else ""

                if not md_content and hasattr(result, 'cleaned_html'):
                    md_content = result.cleaned_html or ""

                logger.info(
                    "web_crawled",
                    url=url,
                    domain=domain,
                    content_length=len(md_content),
                )

                return ParsedDocument(
                    content=md_content,
                    title=title or domain,
                    pages=[PageContent(page_number=1, text=md_content)],
                    metadata={
                        "parser": self.parser_name(),
                        "url": url,
                        "domain": domain,
                        "crawled_at": datetime.now(timezone.utc).isoformat(),
                    },
                    source_type="web",
                )

        except Exception as e:
            logger.error("web_crawl_failed", url=url, error=str(e))
            raise RuntimeError(f"Web crawling failed for {url}: {e}") from e

    def supported_extensions(self) -> list[str]:
        return []  # URLs, not file extensions

    def parser_name(self) -> str:
        return "crawl4ai"
