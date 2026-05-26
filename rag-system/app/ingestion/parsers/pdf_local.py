"""Local PDF parser using pymupdf4llm for Markdown extraction.

Primary parser for text-extractable PDFs. Falls back to alternative
parsers when quality_score indicates poor extraction (scanned docs).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.pdf_local")


class PDFLocalParser(BaseParser):
    """Parse PDFs to Markdown using pymupdf4llm.

    Preserves headers, bold text, lists, and reading order.
    Reports a quality_score in metadata to signal when fallback
    parsers should be attempted.
    """

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        """Parse a PDF file using pymupdf4llm.

        Args:
            source: Path to the PDF file.

        Returns:
            ParsedDocument with Markdown content and per-page breakdown.
        """
        try:
            import pymupdf4llm
            import pymupdf
        except ImportError as e:
            raise ImportError(
                "pymupdf4llm is required. Install with: pip install pymupdf4llm"
            ) from e

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        # Extract full Markdown
        md_content = pymupdf4llm.to_markdown(str(source))

        # Extract per-page content
        doc = pymupdf.open(str(source))
        pages: list[PageContent] = []
        total_chars = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            total_chars += len(text)

            # Extract tables (simple detection via text blocks)
            tables: list[str] = []
            # Detect markdown tables in the output
            page_md = pymupdf4llm.to_markdown(str(source), pages=[page_num])
            table_blocks = re.findall(
                r'(\|.+\|(?:\n\|.+\|)+)', page_md, re.MULTILINE
            )
            tables.extend(table_blocks)

            pages.append(PageContent(
                page_number=page_num + 1,
                text=text,
                tables=tables,
            ))

        doc.close()

        # Quality assessment
        num_pages = max(len(pages), 1)
        avg_chars = total_chars / num_pages
        quality_score = min(1.0, avg_chars / 100.0)  # < 100 chars/page = low quality

        # Extract title from first H1 or PDF metadata
        title = self._extract_title(md_content, source)

        logger.info(
            "pdf_local_parsed",
            source=str(source),
            pages=num_pages,
            total_chars=total_chars,
            quality_score=round(quality_score, 2),
        )

        return ParsedDocument(
            content=md_content,
            title=title,
            pages=pages,
            metadata={
                "quality_score": quality_score,
                "parser": self.parser_name(),
                "page_count": num_pages,
                "total_chars": total_chars,
            },
            source_type="pdf",
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parser_name(self) -> str:
        return "pymupdf4llm"

    @staticmethod
    def _extract_title(content: str, source: Path) -> str:
        """Extract title from Markdown H1 or fall back to filename."""
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        return source.stem.replace("_", " ").replace("-", " ").title()
