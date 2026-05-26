"""Base parser interface and common data classes for document parsing.

All parser implementations must extend BaseParser and return ParsedDocument
objects. This ensures consistent data flow through the ingestion pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PageContent:
    """Content extracted from a single page of a document."""
    page_number: int
    text: str
    tables: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Result of parsing a document through any parser.

    Attributes:
        content: Full extracted text (Markdown preferred).
        title: Document title extracted from content or metadata.
        pages: Per-page content breakdown.
        metadata: Parser-specific metadata (e.g., quality_score).
        source_type: Type of document: 'pdf', 'code', 'web', etc.
    """
    content: str
    title: str | None = None
    pages: list[PageContent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_type: str = "unknown"


class BaseParser(ABC):
    """Abstract base class for all document parsers.

    Subclasses must implement:
    - parse(): Async document parsing
    - supported_extensions(): List of supported file extensions
    - parser_name(): Human-readable parser name
    """

    @abstractmethod
    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        """Parse a document source and return structured content.

        Args:
            source: File path or URL to parse.
            **kwargs: Parser-specific arguments.

        Returns:
            ParsedDocument with extracted content.

        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the file format is unsupported.
            RuntimeError: If parsing fails.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this parser handles.

        Returns:
            List of extensions like ['.pdf'], ['.py', '.js'], etc.
        """
        ...

    @abstractmethod
    def parser_name(self) -> str:
        """Return the human-readable parser name.

        Returns:
            Parser name string (e.g., 'pymupdf4llm', 'docling').
        """
        ...
