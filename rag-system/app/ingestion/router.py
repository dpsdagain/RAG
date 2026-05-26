"""Ingestion router: routes files to the correct parser based on type.

Implements a multi-tier fallback strategy for PDFs and gracefully
handles missing optional parsers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("ingestion.router")

# Extension → file type mapping
EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".py": "code", ".js": "code", ".ts": "code", ".tsx": "code",
    ".go": "code", ".rs": "code", ".java": "code",
    ".c": "code", ".cpp": "code", ".h": "code", ".hpp": "code",
    ".md": "markdown", ".txt": "markdown", ".csv": "markdown",
    ".rst": "markdown",
    ".docx": "office", ".pptx": "office", ".xlsx": "office",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tiff": "image",
    ".html": "web", ".htm": "web",
}


class IngestionRouter:
    """Routes files to the correct parser with multi-tier fallback for PDFs.

    PDF routing logic:
    1. Try pdf_local (pymupdf4llm) first
    2. If quality_score < 0.5, try pdf_docling
    3. If still poor, try pdf_cloud (LlamaParse)

    Missing optional parsers are handled gracefully with warnings.
    """

    def detect_file_type(self, file_path: Path) -> str:
        """Detect file type from extension.

        Args:
            file_path: Path to the file.

        Returns:
            File type string: 'pdf', 'code', 'web', 'markdown', 'image', 'office'.
        """
        ext = file_path.suffix.lower()
        return EXTENSION_MAP.get(ext, "unknown")

    async def parse_file(self, file_path: Path) -> ParsedDocument:
        """Parse a file using the appropriate parser with fallback.

        Args:
            file_path: Path to the file to parse.

        Returns:
            ParsedDocument from the best available parser.

        Raises:
            ValueError: If file type is unsupported.
        """
        file_type = self.detect_file_type(file_path)

        if file_type == "pdf":
            return await self._parse_pdf(file_path)
        elif file_type == "code":
            return await self._parse_code(file_path)
        elif file_type == "markdown":
            return await self._parse_markdown(file_path)
        elif file_type == "image":
            return await self._parse_image(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type} ({file_path.suffix})")

    async def parse_url(self, url: str) -> ParsedDocument:
        """Parse a URL using the web crawler.

        Args:
            url: URL to crawl and parse.

        Returns:
            ParsedDocument with web content.
        """
        try:
            from app.ingestion.parsers.web_crawl4ai import WebCrawl4AIParser
            parser = WebCrawl4AIParser()
            return await parser.parse(url)
        except ImportError:
            raise ImportError(
                "Web parsing requires crawl4ai. Install with: pip install crawl4ai"
            )

    async def _parse_pdf(self, file_path: Path) -> ParsedDocument:
        """Multi-tier PDF parsing with quality-based fallback."""
        # Tier 1: Local parser (pymupdf4llm)
        try:
            from app.ingestion.parsers.pdf_local import PDFLocalParser
            parser = PDFLocalParser()
            result = await parser.parse(file_path)
            quality = result.metadata.get("quality_score", 1.0)

            if quality >= 0.5:
                logger.info("pdf_parsed_tier1", quality=quality)
                return result

            logger.info("pdf_low_quality_tier1", quality=quality, msg="Trying Docling")
        except Exception as e:
            logger.warning("pdf_tier1_failed", error=str(e))

        # Tier 2: Docling
        try:
            from app.ingestion.parsers.pdf_docling import PDFDoclingParser
            parser2 = PDFDoclingParser()
            result = await parser2.parse(file_path)
            logger.info("pdf_parsed_tier2")
            return result
        except ImportError:
            logger.info("docling_not_installed", msg="Skipping tier 2")
        except Exception as e:
            logger.warning("pdf_tier2_failed", error=str(e))

        # Tier 3: Cloud (LlamaParse)
        try:
            from app.ingestion.parsers.pdf_cloud import PDFCloudParser
            parser3 = PDFCloudParser()
            result = await parser3.parse(file_path)
            logger.info("pdf_parsed_tier3")
            return result
        except Exception as e:
            logger.warning("pdf_tier3_failed", error=str(e))

        # Final fallback: return tier 1 result if we got one
        from app.ingestion.parsers.pdf_local import PDFLocalParser
        return await PDFLocalParser().parse(file_path)

    async def _parse_code(self, file_path: Path) -> ParsedDocument:
        """Parse source code files."""
        try:
            from app.ingestion.parsers.code_treesitter import CodeTreeSitterParser
            parser = CodeTreeSitterParser()
            return await parser.parse(file_path)
        except ImportError:
            logger.warning("tree_sitter_not_available", msg="Reading as raw text")
            return self._read_as_markdown(file_path, source_type="code")

    async def _parse_markdown(self, file_path: Path) -> ParsedDocument:
        """Parse plain text / Markdown files."""
        return self._read_as_markdown(file_path, source_type="markdown")

    async def _parse_image(self, file_path: Path) -> ParsedDocument:
        """Parse image files via OCR."""
        # Try PaddleOCR
        try:
            from app.ingestion.parsers.ocr_paddle import OCRPaddleParser
            return await OCRPaddleParser().parse(file_path)
        except ImportError:
            pass

        # Try Cloud VLM
        try:
            from app.ingestion.parsers.ocr_cloud import OCRCloudParser
            return await OCRCloudParser().parse(file_path)
        except Exception as e:
            raise ValueError(f"No OCR parser available for {file_path}: {e}")

    @staticmethod
    def _read_as_markdown(file_path: Path, source_type: str = "markdown") -> ParsedDocument:
        """Read a file as plain text/Markdown."""
        from app.ingestion.parsers.base import PageContent
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(
            content=content,
            title=file_path.stem.replace("_", " ").replace("-", " ").title(),
            pages=[PageContent(page_number=1, text=content)],
            metadata={"parser": "raw"},
            source_type=source_type,
        )
