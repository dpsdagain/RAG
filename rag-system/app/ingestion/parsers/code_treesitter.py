"""Tree-sitter based code parser for AST-aware code extraction.

Parses source code files into their AST and extracts function definitions,
class definitions, and top-level declarations as structured blocks. For
each block we also extract:
  * calls_functions  — names of every function/method called from inside
  * references_constants — every ALL_CAPS identifier referenced

These two metadata streams power the call-graph + constants FTS5 index,
which enables propagation queries ("where is OPENROUTER_API_KEY used?",
"trace how chunks flow through hybrid_search") to hit a direct symbol
index instead of relying on fuzzy embedding match.

Per-language correctness: extraction walks tree-sitter nodes per
language, so it works for Python, JS/TS, Go, Rust, Java, C, C++.
"""
from __future__ import annotations

import re
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

# Tree-sitter node types that represent a function/method call,
# per language. Walking the AST and collecting these gives us a clean
# call graph that works without regex heuristics.
CALL_NODE_TYPES: dict[str, set[str]] = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "go": {"call_expression"},
    "rust": {"call_expression", "macro_invocation"},
    "java": {"method_invocation"},
    "c": {"call_expression"},
    "cpp": {"call_expression"},
}

# Identifiers we consider "constants" for the references_constants index.
# ALL_CAPS, at least 4 chars, may contain digits/underscores. Tight enough
# to avoid noise (e.g. single-letter loop vars).
_CONSTANT_RE = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")


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

            # Walk THIS block's subtree to collect calls + constants.
            # We don't recurse into nested function definitions for
            # call extraction — they get their own block entries.
            calls, constants = self._extract_calls_and_constants(
                node, source, language
            )

            blocks.append({
                "name": name,
                "type": node_type,
                "start_line": start_line,
                "end_line": end_line,
                "content": node_content,
                "docstring": docstring,
                "language": language,
                "calls_functions": calls,
                "references_constants": constants,
            })

        # Recurse into children
        for child in node.children:
            self._extract_nodes(child, source, types, blocks, language)

    @classmethod
    def _extract_calls_and_constants(
        cls, root_node: Any, source: str, language: str,
    ) -> tuple[list[str], list[str]]:
        """Walk a subtree, collect every call name + ALL_CAPS identifier.

        Returns (sorted_unique_calls, sorted_unique_constants).
        Both lists are deduplicated; empty lists when none found or
        when the language has no known call-node-type mapping.
        """
        call_types = CALL_NODE_TYPES.get(language, set())
        calls: set[str] = set()
        constants: set[str] = set()

        def _walk(n: Any) -> None:
            try:
                ntype = n.type
            except Exception:
                return

            # Call expression: take the FIRST identifier-like child as the
            # function name. This handles `foo()`, `obj.foo()`, `Type::foo()`.
            if ntype in call_types:
                fn_name = cls._first_identifier(n, source)
                if fn_name:
                    calls.add(fn_name)

            # Standalone identifier: check if it's ALL_CAPS constant.
            if ntype in ("identifier", "name", "property_identifier"):
                text = source[n.start_byte:n.end_byte]
                if _CONSTANT_RE.match(text):
                    constants.add(text)

            for child in n.children:
                _walk(child)

        _walk(root_node)
        return sorted(calls), sorted(constants)

    @staticmethod
    def _first_identifier(node: Any, source: str) -> str | None:
        """Return the function name from a call_expression node.

        Walks ONLY the function expression (first child of the call),
        not the arguments, so we don't accidentally pick up identifiers
        from inside the call's parameter list.

        For `obj.method()` → "method"; for `Type::foo()` → "foo".
        Returns the last identifier in the chain — that's the actual
        function name.
        """
        if not node.children:
            return None
        fn_expr = node.children[0]  # function expression, NOT args

        found: list[str] = []

        def _walk(n: Any) -> None:
            try:
                if n.type in ("identifier", "name", "property_identifier"):
                    found.append(source[n.start_byte:n.end_byte])
                for child in n.children:
                    _walk(child)
            except Exception:
                pass

        _walk(fn_expr)
        return found[-1] if found else None

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
