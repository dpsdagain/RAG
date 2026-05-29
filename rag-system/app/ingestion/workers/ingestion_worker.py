"""Background ingestion worker running as a threading.Thread.

Monitors a watch directory for new files, processes them through
the parsing → chunking → embedding → storage pipeline, and manages
CPU contention with the query pipeline via semaphore.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.infrastructure.config import Settings
from app.infrastructure.database import Database, DocumentRecord
from app.infrastructure.observability import get_logger, metrics
from app.ingestion.chunking.code_chunker import CodeChunker
from app.ingestion.chunking.semantic_chunker import SemanticChunker
from app.ingestion.router import IngestionRouter
from app.models.embedder import Embedder
from app.models.llm_client import LLMClient

logger = get_logger("ingestion.worker")


@dataclass
class IngestResult:
    """Result of an ingestion operation."""
    doc_id: str
    filename: str
    status: str  # 'success', 'skipped', 'failed'
    chunks_created: int = 0
    error_message: str | None = None


class IngestionWorker(threading.Thread):
    """Background worker thread for file ingestion.

    Monitors a watch directory for new files, deduplicates by file hash,
    routes to the appropriate parser, chunks the content, embeds the
    chunks, and stores everything in the database.

    Uses a shared threading.Semaphore to prevent CPU contention between
    embedding operations and real-time query processing.
    """

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        settings: Settings,
        cpu_semaphore: threading.Semaphore | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        super().__init__(daemon=True, name="IngestionWorker")
        self._db = db
        self._embedder = embedder
        self._settings = settings
        # Optional LLM used only at ingest for the Contextual Retrieval
        # header (one short call per doc). If None, headers are skipped
        # and ingestion behaves as before.
        self._llm = llm
        self._router = IngestionRouter()
        self._chunker = SemanticChunker(
            embedder=embedder,
            max_chunk_tokens=settings.chunking.max_chunk_tokens,
            min_chunk_tokens=settings.chunking.min_chunk_tokens,
            semantic_threshold=settings.chunking.semantic_threshold,
        )
        self._code_chunker = CodeChunker(embedder)
        self._cpu_semaphore = cpu_semaphore or threading.Semaphore(1)
        self._stop_event = threading.Event()
        self._watch_dir = Path(settings.ingestion.watch_directory) if settings.ingestion.watch_directory else None
        self._poll_interval = 10.0  # seconds
        self._processed_hashes: set[str] = set()

    def run(self) -> None:
        """Main loop: watch directory for new files."""
        if not self._watch_dir:
            logger.info("ingestion_worker_no_watch_dir", msg="Watch directory not configured")
            return

        self._watch_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ingestion_worker_started", watch_dir=str(self._watch_dir))

        while not self._stop_event.is_set():
            try:
                self._scan_directory()
            except Exception as e:
                logger.error("ingestion_scan_error", error=str(e))

            self._stop_event.wait(self._poll_interval)

        logger.info("ingestion_worker_stopped")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()

    def _scan_directory(self) -> None:
        """Scan watch directory for new files."""
        if not self._watch_dir or not self._watch_dir.exists():
            return

        for file_path in self._watch_dir.iterdir():
            if file_path.is_file() and not file_path.name.startswith("."):
                # Check file size
                size_mb = file_path.stat().st_size / (1024 * 1024)
                if size_mb > self._settings.ingestion.max_file_size_mb:
                    logger.warning(
                        "file_too_large",
                        file=str(file_path),
                        size_mb=round(size_mb, 1),
                    )
                    continue

                file_hash = self._compute_file_hash(file_path)
                if file_hash not in self._processed_hashes:
                    # Run async ingest in a new event loop
                    loop = asyncio.new_event_loop()
                    try:
                        result = loop.run_until_complete(
                            self.ingest_file(file_path)
                        )
                        if result.status != "failed":
                            self._processed_hashes.add(file_hash)
                    finally:
                        loop.close()

    async def ingest_file(self, file_path: Path) -> IngestResult:
        """Ingest a single file through the full pipeline.

        Args:
            file_path: Path to the file to ingest.

        Returns:
            IngestResult with status and chunk count.
        """
        t0 = time.perf_counter()
        filename = file_path.name
        doc_id = str(uuid.uuid4())

        try:
            # Step 1: Compute file hash for dedup
            file_hash = self._compute_file_hash(file_path)
            existing = await self._db.get_document_by_hash(file_hash)
            if existing:
                logger.info("ingestion_skipped_duplicate", file=filename, existing_id=existing.doc_id)
                return IngestResult(
                    doc_id=existing.doc_id,
                    filename=filename,
                    status="skipped",
                )

            # Step 2: Parse
            parsed = await self._router.parse_file(file_path)

            # Step 2b: Contextual header (Anthropic Contextual Retrieval).
            # One LLM call per file → 1-sentence summary. The header gets
            # prepended to every chunk's content before embedding so terse
            # chunks (e.g. `RETRIEVER_K = 12`) carry their file's context
            # into the embedding space. Best documented ~60% retrieval
            # failure reduction in Anthropic's paper.
            contextual_header = await self._make_contextual_header(parsed, filename)

            # Step 3: Chunk (with CPU semaphore for embedding)
            self._cpu_semaphore.acquire()
            try:
                if parsed.source_type == "code":
                    chunks = self._code_chunker.chunk(
                        parsed, doc_id, contextual_header=contextual_header
                    )
                else:
                    chunks = self._chunker.chunk(
                        parsed, doc_id, contextual_header=contextual_header
                    )
            finally:
                self._cpu_semaphore.release()

            if not chunks:
                logger.warning("ingestion_no_chunks", file=filename)
                return IngestResult(
                    doc_id=doc_id, filename=filename,
                    status="failed", error_message="No chunks generated",
                )

            # Step 4: Store document + chunks
            content_hash = hashlib.sha256(parsed.content.encode()).hexdigest()
            doc_record = DocumentRecord(
                doc_id=doc_id,
                source_uri=str(file_path.absolute()),
                source_type=parsed.source_type,
                file_hash=file_hash,
                content_hash=content_hash,
                title=parsed.title,
                parser_used=parsed.metadata.get("parser", "unknown"),
                total_chunks=0,  # Updated by insert_chunks_batch
                status="active",
            )
            await self._db.insert_document(doc_record)
            await self._db.insert_chunks_batch(chunks)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.record_latency("ingestion", elapsed_ms)
            metrics.increment("documents_ingested")

            logger.info(
                "file_ingested",
                doc_id=doc_id,
                file=filename,
                chunks=len(chunks),
                latency_ms=round(elapsed_ms, 1),
                parser=parsed.metadata.get("parser"),
            )

            return IngestResult(
                doc_id=doc_id,
                filename=filename,
                status="success",
                chunks_created=len(chunks),
            )

        except Exception as e:
            logger.error("ingestion_failed", file=filename, error=str(e))
            metrics.increment("ingestion_failures")

            # Log to failed_ingests table
            try:
                await self._db.log_failed_ingest(
                    failure_id=str(uuid.uuid4()),
                    source_uri=str(file_path),
                    error_message=str(e),
                    parser_attempted="auto",
                )
            except Exception:
                pass

            return IngestResult(
                doc_id=doc_id,
                filename=filename,
                status="failed",
                error_message=str(e),
            )

    async def ingest_url(self, url: str) -> IngestResult:
        """Ingest content from a URL.

        Args:
            url: URL to crawl and ingest.

        Returns:
            IngestResult with status and chunk count.
        """
        t0 = time.perf_counter()
        doc_id = str(uuid.uuid4())

        try:
            # Parse URL
            parsed = await self._router.parse_url(url)

            # Dedup by content hash
            content_hash = hashlib.sha256(parsed.content.encode()).hexdigest()

            # Contextual header (same as file ingest path).
            contextual_header = await self._make_contextual_header(parsed, url)

            # Chunk
            self._cpu_semaphore.acquire()
            try:
                chunks = self._chunker.chunk(
                    parsed, doc_id, contextual_header=contextual_header
                )
            finally:
                self._cpu_semaphore.release()

            if not chunks:
                return IngestResult(
                    doc_id=doc_id, filename=url,
                    status="failed", error_message="No chunks generated from URL",
                )

            # Store
            doc_record = DocumentRecord(
                doc_id=doc_id,
                source_uri=url,
                source_type="web",
                file_hash=content_hash,
                content_hash=content_hash,
                title=parsed.title,
                parser_used=parsed.metadata.get("parser", "crawl4ai"),
                status="active",
            )
            await self._db.insert_document(doc_record)
            await self._db.insert_chunks_batch(chunks)

            elapsed_ms = (time.perf_counter() - t0) * 1000
            metrics.record_latency("ingestion_url", elapsed_ms)

            logger.info(
                "url_ingested",
                doc_id=doc_id,
                url=url,
                chunks=len(chunks),
                latency_ms=round(elapsed_ms, 1),
            )

            return IngestResult(
                doc_id=doc_id,
                filename=url,
                status="success",
                chunks_created=len(chunks),
            )

        except Exception as e:
            logger.error("url_ingestion_failed", url=url, error=str(e))
            return IngestResult(
                doc_id=doc_id, filename=url,
                status="failed", error_message=str(e),
            )

    async def _make_contextual_header(self, parsed, filename: str) -> str:
        """Generate a 1-sentence contextual header for the parsed document.

        Returns a short summary like
        *"This file (config.py) holds the retrieval/cache tuning constants."*
        that gets prepended to every chunk before embedding. Improves
        retrieval precision on terse chunks (constant definitions, short
        utility functions) that have weak standalone signal.

        On any failure (no LLM configured, API error, empty input) returns
        an empty string — ingestion continues without the header, falling
        back to pre-Contextual-Retrieval behavior.
        """
        if self._llm is None or not parsed.content:
            return ""

        # Send only the first ~3000 chars to keep the call cheap and
        # predictable. The summary is about the doc as a WHOLE, not
        # per-chunk, so the head usually contains enough signal.
        snippet = parsed.content[:3000]
        title_hint = parsed.title or filename
        prompt = (
            f"In ONE sentence (max 25 words), describe what this document is "
            f"about. Be specific: name the topic, the file's apparent role, "
            f"or its main subject matter. Do NOT preface with 'This document' "
            f"or 'The file' — just state the topic directly.\n\n"
            f"FILE: {title_hint}\n\n"
            f"CONTENT (truncated):\n{snippet}"
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self._llm.generate(
                messages, temperature=0.0, max_tokens=80
            )
            # Strip surrounding whitespace and clip to 200 chars defensively.
            header = response.strip().replace("\n", " ")[:200]
            if header:
                metrics.increment("contextual_headers_generated")
                logger.info(
                    "contextual_header",
                    file=filename,
                    header_preview=header[:80],
                )
            return header
        except Exception as e:
            logger.warning("contextual_header_failed", file=filename, error=str(e))
            metrics.increment("contextual_header_errors")
            return ""

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                sha256.update(block)
        return sha256.hexdigest()
