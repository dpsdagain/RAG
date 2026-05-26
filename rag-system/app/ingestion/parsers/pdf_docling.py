"""Docling-based PDF parser for complex document layouts.

Uses the Granite-Docling-258M model for structure-aware parsing of
multi-column, table-heavy, and complex PDF layouts on CPU.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.pdf_docling")

_docling_available = True
try:
    from docling.document_converter import DocumentConverter  # type: ignore[import-untyped]
except ImportError:
    _docling_available = False


class PDFDoclingParser(BaseParser):
    """Parse complex PDFs using Docling with Granite-Docling-258M.

    Handles multi-column layouts, embedded tables, and complex document
    structures. Optional dependency — raises clear error if not installed.
    """

    def __init__(self) -> None:
        self._converter: Any = None

    def load_model(self) -> None:
        """Load the Docling converter model into memory."""
        if not _docling_available:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling\n"
                "This is an optional dependency for complex PDF parsing."
            )
        if self._converter is None:
            self._converter = DocumentConverter()
            logger.info("docling_model_loaded")

    def unload_model(self) -> None:
        """Unload the model to free RAM."""
        self._converter = None
        logger.info("docling_model_unloaded")

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        """Parse a PDF file using Docling.

        Args:
            source: Path to the PDF file.

        Returns:
            ParsedDocument with structured Markdown content.
        """
        if not _docling_available:
            raise ImportError(
                "Docling is not installed. Install with: pip install docling"
            )

        self.load_model()
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"PDF not found: {source}")

        try:
            result = self._converter.convert(str(source))
            md_content = result.document.export_to_markdown()

            pages: list[PageContent] = []
            # Docling provides page-level content in some modes
            if hasattr(result.document, 'pages'):
                for i, page in enumerate(result.document.pages):
                    pages.append(PageContent(
                        page_number=i + 1,
                        text=str(page) if page else "",
                    ))

            title = None
            if hasattr(result.document, 'title') and result.document.title:
                title = str(result.document.title)
            else:
                title = source.stem.replace("_", " ").replace("-", " ").title()

            logger.info("pdf_docling_parsed", source=str(source), content_length=len(md_content))

            return ParsedDocument(
                content=md_content,
                title=title,
                pages=pages,
                metadata={
                    "parser": self.parser_name(),
                    "quality_score": 0.9,
                },
                source_type="pdf",
            )
        except Exception as e:
            logger.error("pdf_docling_failed", source=str(source), error=str(e))
            raise RuntimeError(f"Docling parsing failed: {e}") from e

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parser_name(self) -> str:
        return "docling"
