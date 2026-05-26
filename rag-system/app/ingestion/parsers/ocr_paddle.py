"""PaddleOCR-based parser for scanned documents.

Optional dependency for OCR processing of clean scanned PDFs.
Renders PDF pages to images and processes them through PaddleOCR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.ocr_paddle")

_paddleocr_available = True
try:
    from paddleocr import PaddleOCR  # type: ignore[import-untyped]
except ImportError:
    _paddleocr_available = False


class OCRPaddleParser(BaseParser):
    """Parse scanned documents using PaddleOCR.

    Renders PDF pages to images using pymupdf, then processes each
    image through PaddleOCR for text extraction. Optional dependency.
    """

    def __init__(self) -> None:
        self._ocr: Any = None

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        if not _paddleocr_available:
            raise ImportError(
                "PaddleOCR is not installed. Install with: pip install paddleocr"
            )

        try:
            import pymupdf
        except ImportError as e:
            raise ImportError("pymupdf is required for PDF page rendering") from e

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"File not found: {source}")

        # Initialize OCR engine
        if self._ocr is None:
            self._ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

        doc = pymupdf.open(str(source))
        pages: list[PageContent] = []
        all_text: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render page to image at 200 DPI
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")

            # Save to temp file for PaddleOCR
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                tmp_path = tmp.name

            try:
                result = self._ocr.ocr(tmp_path, cls=True)
                page_text = ""
                if result and result[0]:
                    lines = [line[1][0] for line in result[0] if line[1]]
                    page_text = "\n".join(lines)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            pages.append(PageContent(
                page_number=page_num + 1,
                text=page_text,
            ))
            all_text.append(page_text)

        doc.close()

        content = "\n\n".join(all_text)
        logger.info(
            "ocr_paddle_parsed",
            source=str(source),
            pages=len(pages),
            content_length=len(content),
        )

        return ParsedDocument(
            content=content,
            title=source.stem.replace("_", " ").title(),
            pages=pages,
            metadata={"parser": self.parser_name(), "quality_score": 0.7},
            source_type="pdf",
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf", ".png", ".jpg", ".jpeg", ".tiff"]

    def parser_name(self) -> str:
        return "paddleocr"
