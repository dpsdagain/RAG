"""Code-specific chunker for tree-sitter parsed source files.

Each function, class, or method becomes its own chunk. The imports
section becomes a parent chunk linked to all function chunks in the file.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import Any

from app.infrastructure.database import ChunkRecord
from app.infrastructure.observability import get_logger
from app.ingestion.parsers.base import ParsedDocument
from app.models.embedder import Embedder

logger = get_logger("chunking.code_chunker")


class CodeChunker:
    """Specialized chunker for code parsed by tree-sitter.

    Creates one chunk per function/class/method, with an imports chunk
    serving as the parent for all blocks in the file.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def chunk(self, parsed_doc: ParsedDocument, doc_id: str) -> list[ChunkRecord]:
        """Chunk code into function/class-level segments.

        Args:
            parsed_doc: ParsedDocument from CodeTreeSitterParser.
            doc_id: Document ID.

        Returns:
            List of ChunkRecord objects.
        """
        blocks: list[dict[str, Any]] = parsed_doc.metadata.get("blocks", [])
        language = parsed_doc.metadata.get("language", "unknown")
        file_path = parsed_doc.metadata.get("file_path", "")

        if not blocks:
            # No AST blocks — fall back to single chunk
            if parsed_doc.content:
                embedding = self._embedder.embed(parsed_doc.content)
                return [ChunkRecord(
                    chunk_id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    content=parsed_doc.content,
                    embedding=embedding,
                    chunk_index=0,
                    section_title=parsed_doc.title,
                    token_count=self._embedder.count_tokens(parsed_doc.content),
                    content_hash=hashlib.sha256(parsed_doc.content.encode()).hexdigest(),
                    metadata_json=f'{{"language": "{language}", "file_path": "{file_path}"}}',
                )]
            return []

        records: list[ChunkRecord] = []

        # Extract imports section (everything before first block)
        first_start = blocks[0].get("start_line", 1) if blocks else 1
        lines = parsed_doc.content.split("\n")
        import_lines = lines[:max(0, first_start - 1)]
        import_text = "\n".join(import_lines).strip()

        # Create imports parent chunk
        parent_chunk_id: str | None = None
        if import_text:
            parent_chunk_id = str(uuid.uuid4())
            import_embedding = self._embedder.embed(import_text)
            records.append(ChunkRecord(
                chunk_id=parent_chunk_id,
                doc_id=doc_id,
                content=import_text,
                embedding=import_embedding,
                chunk_index=0,
                section_title=f"imports ({parsed_doc.title})",
                token_count=self._embedder.count_tokens(import_text),
                content_hash=hashlib.sha256(import_text.encode()).hexdigest(),
                metadata_json=f'{{"type": "imports", "language": "{language}", "file_path": "{file_path}"}}',
            ))

        # Create one chunk per code block
        texts = [b["content"] for b in blocks]
        embeddings = self._embedder.embed_batch(texts) if texts else []

        for i, (block, embedding) in enumerate(zip(blocks, embeddings)):
            chunk_id = str(uuid.uuid4())
            content = block["content"]
            metadata = {
                "type": block.get("type", "function"),
                "name": block.get("name", "unnamed"),
                "language": language,
                "file_path": file_path,
                "start_line": block.get("start_line"),
                "end_line": block.get("end_line"),
            }
            if block.get("docstring"):
                metadata["docstring"] = block["docstring"]

            records.append(ChunkRecord(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                embedding=embedding,
                chunk_index=i + 1,
                parent_chunk_id=parent_chunk_id,
                section_title=f"{block.get('type', 'fn')}:{block.get('name', 'unnamed')}",
                token_count=self._embedder.count_tokens(content),
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                metadata_json=str(metadata).replace("'", '"'),
            ))

        logger.info(
            "code_chunked",
            doc_id=doc_id,
            blocks=len(blocks),
            chunks=len(records),
            language=language,
        )

        return records
