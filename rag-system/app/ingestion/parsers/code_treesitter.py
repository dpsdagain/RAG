"""Tree-sitter based code parser for AST-aware code extraction.

Parses source code files into their AST and extracts function definitions,
class definitions, and top-level declarations as structured blocks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ingestion.parsers.base import BaseParser, PageContent, ParsedDocument
from app.infrastructure.observability import get_logger

logger = get_logger("parsers.code_treesitter")

_tree_sitter_available = True
try:
    import tree_sitter  # type: ignore[import-untyped]
except ImportError:
    _tree_sitter_available = False

# Extension → (language_name, tree-sitter module name)
LANGUAGE_MAP: dict[str, tuple[str, str]] = {
    ".py": ("python", "tree_sitter_python"),
    ".js": ("javascript", "tree_sitter_javascript"),
    ".ts": ("typescript", "tree_sitter_typescript"),
    ".tsx": ("typescript", "tree_sitter_typescript"),
    ".go": ("go", "tree_sitter_go"),
    ".rs": ("rust", "tree_sitter_rust"),
    ".java": ("java", "tree_sitter_java"),
    ".c": ("c", "tree_sitter_c"),
    ".cpp": ("cpp", "tree_sitter_cpp"),
    ".h": ("c", "tree_sitter_c"),
    ".hpp": ("cpp", "tree_sitter_cpp"),
}

# Node types to extract per language
EXTRACT_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "class_declaration", "arrow_function", "method_definition"},
    "typescript": {"function_declaration", "class_declaration", "arrow_function", "method_definition"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item"},
    "java": {"method_declaration", "class_declaration", "interface_declaration"},
    "c": {"function_definition", "struct_specifier"},
    "cpp": {"function_definition", "class_specifier", "struct_specifier"},
}


class CodeTreeSitterParser(BaseParser):
    """Parse source code into AST-structured blocks using tree-sitter.

    Each function/class/method becomes a separate 'page' in the
    ParsedDocument, enabling per-function chunking downstream.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Any] = {}

    def _get_parser(self, ext: str) -> tuple[Any, str] | None:
        """Get or create a tree-sitter parser for the given extension."""
        if not _tree_sitter_available:
            return None

        if ext not in LANGUAGE_MAP:
            return None

        lang_name, module_name = LANGUAGE_MAP[ext]
        if lang_name in self._parsers:
            return self._parsers[lang_name], lang_name

        try:
            lang_module = __import__(module_name)
            if hasattr(lang_module, 'language'):
                lang_fn = lang_module.language()
            else:
                lang_fn = lang_module.Language.build_library  # type: ignore
                return None  # Old API, skip

            language = tree_sitter.Language(lang_fn)
            parser = tree_sitter.Parser(language)
            self._parsers[lang_name] = parser
            return parser, lang_name
        except (ImportError, Exception) as e:
            logger.warning("tree_sitter_lang_failed", language=lang_name, error=str(e))
            return None

    async def parse(self, source: str | Path, **kwargs: Any) -> ParsedDocument:
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        content = source.read_text(encoding="utf-8", errors="replace")
        ext = source.suffix.lower()

        if not _tree_sitter_available:
            logger.warning("tree_sitter_not_available", msg="Falling back to raw content")
            return self._fallback_parse(content, source)

        result = self._get_parser(ext)
        if result is None:
            return self._fallback_parse(content, source)

        parser, lang_name = result
        tree = parser.parse(bytes(content, "utf-8"))
        extract_types = EXTRACT_TYPES.get(lang_name, set())

        blocks: list[dict[str, Any]] = []
        self._extract_nodes(tree.root_node, content, extract_types, blocks, lang_name)

        pages: list[PageContent] = []
        for i, block in enumerate(blocks):
            pages.append(PageContent(
                page_number=i + 1,
                text=block["content"],
            ))

        logger.info(
            "code_parsed",
            source=str(source),
            language=lang_name,
            blocks_extracted=len(blocks),
        )

        return ParsedDocument(
            content=content,
            title=source.name,
            pages=pages,
            metadata={
                "parser": self.parser_name(),
                "language": lang_name,
                "blocks": blocks,
                "file_path": str(source),
            },
            source_type="code",
        )

    def _extract_nodes(
        self,
        node: Any,
        source: str,
        types: set[str],
        blocks: list[dict],
        language: str,
    ) -> None:
        """Recursively extract matching AST nodes."""
        if node.type in types:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            node_content = source[node.start_byte:node.end_byte]

            # Extract name
            name = "unnamed"
            for child in node.children:
                if child.type in ("identifier", "name", "property_identifier"):
                    name = source[child.start_byte:child.end_byte]
                    break

            # Determine type
            node_type = "function"
            if "class" in node.type:
                node_type = "class"
            elif "method" in node.type:
                node_type = "method"
            elif "struct" in node.type or "interface" in node.type or "enum" in node.type:
                node_type = "type"

            # Extract docstring (first string child for Python)
            docstring = ""
            if language == "python" and node.type in ("function_definition", "class_definition"):
                for child in node.children:
                    if child.type == "block":
                        for stmt in child.children:
                            if stmt.type == "expression_statement":
                                for expr in stmt.children:
                                    if expr.type == "string":
                                        docstring = source[expr.start_byte:expr.end_byte]
                                        break
                            break

            blocks.append({
                "name": name,
                "type": node_type,
                "start_line": start_line,
                "end_line": end_line,
                "content": node_content,
                "docstring": docstring,
                "language": language,
            })

        # Recurse into children
        for child in node.children:
            self._extract_nodes(child, source, types, blocks, language)

    @staticmethod
    def _fallback_parse(content: str, source: Path) -> ParsedDocument:
        """Fallback when tree-sitter is not available."""
        return ParsedDocument(
            content=content,
            title=source.name,
            pages=[PageContent(page_number=1, text=content)],
            metadata={"parser": "raw", "language": source.suffix},
            source_type="code",
        )

    def supported_extensions(self) -> list[str]:
        return list(LANGUAGE_MAP.keys())

    def parser_name(self) -> str:
        return "tree_sitter"
