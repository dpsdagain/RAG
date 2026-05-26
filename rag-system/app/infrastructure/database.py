"""Database layer: SQLite + sqlite-vec + FTS5.

Manages all persistent storage including documents, chunks (with vector
embeddings), full-text search index, conversations, memory, preferences,
procedural rules, and request logging.

Uses asyncio.to_thread() for non-blocking SQLite access with a
threading.Lock for write serialization.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from app.infrastructure.observability import get_logger, metrics

logger = get_logger("database")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentRecord:
    """Represents a document in the documents table."""
    doc_id: str
    source_uri: str
    source_type: str  # 'pdf', 'code', 'web', 'markdown', 'image'
    file_hash: str
    content_hash: str | None = None
    title: str | None = None
    parser_used: str | None = None
    total_chunks: int = 0
    status: str = "active"  # 'active', 'reindexing', 'deleted'


@dataclass
class ChunkRecord:
    """Represents a chunk in the chunks table."""
    chunk_id: str
    doc_id: str
    content: str
    embedding: list[float]
    chunk_index: int
    parent_chunk_id: str | None = None
    section_title: str | None = None
    source_page: int | None = None
    token_count: int = 0
    content_hash: str = ""
    event_time: str | None = None
    metadata_json: str | None = None


@dataclass
class ChunkResult:
    """Represents a chunk returned from search operations."""
    chunk_id: str
    doc_id: str
    content: str
    score: float
    chunk_index: int = 0
    parent_chunk_id: str | None = None
    section_title: str | None = None
    source_page: int | None = None
    token_count: int = 0
    source_uri: str = ""
    source_type: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_f32(vec: list[float]) -> bytes:
    """Serialize a float list to a compact F32 binary blob for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_f32(blob: bytes, dim: int = 384) -> list[float]:
    """Deserialize a F32 binary blob back to a float list."""
    return list(struct.unpack(f"{dim}f", blob))


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------

class Database:
    """Async-friendly SQLite database manager with sqlite-vec and FTS5 support.

    All write operations are serialized through a threading.Lock.
    All blocking SQLite calls are dispatched via asyncio.to_thread().
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = Lock()
        self._conn: sqlite3.Connection | None = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create the SQLite connection with required extensions."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            # Load sqlite-vec extension
            self._conn.enable_load_extension(True)
            try:
                import sqlite_vec
                sqlite_vec.load(self._conn)
            except (ImportError, Exception) as e:
                logger.warning("sqlite-vec not available, vector search disabled", error=str(e))
            self._conn.enable_load_extension(False)
            # Set critical pragmas
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    async def initialize(self) -> None:
        """Create all tables and indexes. Idempotent."""
        await asyncio.to_thread(self._initialize_sync)
        logger.info("database_initialized", db_path=str(self._db_path))

    def _initialize_sync(self) -> None:
        """Synchronous table creation."""
        conn = self._get_connection()
        with self._write_lock:
            # Documents
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id          TEXT PRIMARY KEY,
                    source_uri      TEXT NOT NULL,
                    source_type     TEXT NOT NULL,
                    file_hash       TEXT NOT NULL,
                    content_hash    TEXT,
                    title           TEXT,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parser_used     TEXT,
                    total_chunks    INTEGER DEFAULT 0,
                    status          TEXT DEFAULT 'active'
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_uri)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")

            # Chunks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id        TEXT PRIMARY KEY,
                    doc_id          TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                    content         TEXT NOT NULL,
                    chunk_index     INTEGER NOT NULL,
                    parent_chunk_id TEXT,
                    section_title   TEXT,
                    source_page     INTEGER,
                    token_count     INTEGER NOT NULL,
                    content_hash    TEXT NOT NULL,
                    event_time      TIMESTAMP,
                    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata_json   TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_chunk_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_ingested ON chunks(ingested_at)")

            # sqlite-vec virtual table for vector search
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                        chunk_id TEXT PRIMARY KEY,
                        embedding float[384]
                    )
                """)
            except sqlite3.OperationalError as e:
                logger.warning("Could not create vec0 virtual table", error=str(e))

            # FTS5 full-text search index
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                        content,
                        section_title,
                        content='chunks',
                        content_rowid='rowid',
                        tokenize='porter unicode61'
                    )
                """)
                # Sync triggers
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, content, section_title)
                        VALUES (NEW.rowid, NEW.content, NEW.section_title);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
                        VALUES ('delete', OLD.rowid, OLD.content, OLD.section_title);
                    END
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, content, section_title)
                        VALUES ('delete', OLD.rowid, OLD.content, OLD.section_title);
                        INSERT INTO chunks_fts(rowid, content, section_title)
                        VALUES (NEW.rowid, NEW.content, NEW.section_title);
                    END
                """)
            except sqlite3.OperationalError as e:
                logger.warning("Could not create FTS5 table", error=str(e))

            # Conversations (episodic memory)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    started_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    summary         TEXT,
                    turn_count      INTEGER DEFAULT 0
                )
            """)
            # Conversation vector search table
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS conversations_vec USING vec0(
                        conversation_id TEXT PRIMARY KEY,
                        summary_embedding float[384]
                    )
                """)
            except sqlite3.OperationalError:
                pass

            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id         TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                    role            TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    retrieved_chunk_ids TEXT,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    turn_index      INTEGER NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON conversation_turns(conversation_id)")

            # Working memory
            conn.execute("""
                CREATE TABLE IF NOT EXISTS working_memory (
                    thread_id       TEXT PRIMARY KEY,
                    state_blob      TEXT NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # User preferences
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    pref_id         TEXT PRIMARY KEY,
                    category        TEXT NOT NULL,
                    preference      TEXT NOT NULL,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    supersedes_id   TEXT,
                    is_active       BOOLEAN DEFAULT 1
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prefs_active ON user_preferences(is_active)")

            # Procedural rules
            conn.execute("""
                CREATE TABLE IF NOT EXISTS procedural_rules (
                    rule_id         TEXT PRIMARY KEY,
                    trigger_pattern TEXT NOT NULL,
                    rule_text       TEXT NOT NULL,
                    priority        INTEGER DEFAULT 0,
                    is_active       BOOLEAN DEFAULT 1,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Failed ingests
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_ingests (
                    failure_id      TEXT PRIMARY KEY,
                    source_uri      TEXT NOT NULL,
                    error_message   TEXT NOT NULL,
                    parser_attempted TEXT,
                    failed_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    retry_count     INTEGER DEFAULT 0,
                    resolved        BOOLEAN DEFAULT 0
                )
            """)

            # Request log
            conn.execute("""
                CREATE TABLE IF NOT EXISTS request_log (
                    request_id      TEXT PRIMARY KEY,
                    query           TEXT NOT NULL,
                    sub_queries     TEXT,
                    crag_verdict    TEXT,
                    faithfulness    TEXT,
                    retrieved_chunks TEXT,
                    response_length INTEGER,
                    total_latency_ms INTEGER,
                    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.commit()

    # ------------------------------------------------------------------
    # Generic query helpers
    # ------------------------------------------------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute a write SQL statement."""
        def _run() -> None:
            conn = self._get_connection()
            with self._write_lock:
                conn.execute(sql, params)
                conn.commit()
        await asyncio.to_thread(_run)

    async def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Fetch all rows for a read query."""
        def _run() -> list[sqlite3.Row]:
            conn = self._get_connection()
            return conn.execute(sql, params).fetchall()
        return await asyncio.to_thread(_run)

    async def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        """Fetch a single row for a read query."""
        def _run() -> sqlite3.Row | None:
            conn = self._get_connection()
            return conn.execute(sql, params).fetchone()
        return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Document operations
    # ------------------------------------------------------------------

    async def insert_document(self, doc: DocumentRecord) -> str:
        """Insert a document record. Returns doc_id."""
        def _run() -> str:
            conn = self._get_connection()
            with self._write_lock:
                conn.execute(
                    """INSERT INTO documents (doc_id, source_uri, source_type, file_hash, content_hash, title, parser_used, total_chunks, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (doc.doc_id, doc.source_uri, doc.source_type, doc.file_hash,
                     doc.content_hash, doc.title, doc.parser_used, doc.total_chunks, doc.status),
                )
                conn.commit()
            return doc.doc_id
        return await asyncio.to_thread(_run)

    async def get_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """Look up a document by file hash for deduplication."""
        row = await self.fetch_one(
            "SELECT * FROM documents WHERE file_hash = ? AND status = 'active'",
            (file_hash,),
        )
        if row is None:
            return None
        return DocumentRecord(
            doc_id=row["doc_id"], source_uri=row["source_uri"],
            source_type=row["source_type"], file_hash=row["file_hash"],
            content_hash=row["content_hash"], title=row["title"],
            parser_used=row["parser_used"], total_chunks=row["total_chunks"],
            status=row["status"],
        )

    async def delete_document(self, doc_id: str) -> None:
        """Delete a document and all its chunks (cascading)."""
        def _run() -> None:
            conn = self._get_connection()
            with self._write_lock:
                # Delete from vec table
                try:
                    chunk_ids = [
                        r[0] for r in conn.execute(
                            "SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)
                        ).fetchall()
                    ]
                    for cid in chunk_ids:
                        conn.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
                except sqlite3.OperationalError:
                    pass
                # Cascade deletes chunks → triggers delete FTS5 entries
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                conn.commit()
        await asyncio.to_thread(_run)
        logger.info("document_deleted", doc_id=doc_id)

    # ------------------------------------------------------------------
    # Chunk operations
    # ------------------------------------------------------------------

    async def insert_chunk(self, chunk: ChunkRecord) -> str:
        """Insert a single chunk with its embedding. Returns chunk_id."""
        def _run() -> str:
            conn = self._get_connection()
            with self._write_lock:
                conn.execute(
                    """INSERT INTO chunks (chunk_id, doc_id, content, chunk_index, parent_chunk_id,
                       section_title, source_page, token_count, content_hash, event_time, metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (chunk.chunk_id, chunk.doc_id, chunk.content, chunk.chunk_index,
                     chunk.parent_chunk_id, chunk.section_title, chunk.source_page,
                     chunk.token_count, chunk.content_hash, chunk.event_time, chunk.metadata_json),
                )
                # Insert into vec table
                try:
                    conn.execute(
                        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                        (chunk.chunk_id, _serialize_f32(chunk.embedding)),
                    )
                except sqlite3.OperationalError:
                    pass
                conn.commit()
            return chunk.chunk_id
        return await asyncio.to_thread(_run)

    async def insert_chunks_batch(self, chunks: list[ChunkRecord]) -> list[str]:
        """Insert multiple chunks in a single transaction. Returns list of chunk_ids."""
        if not chunks:
            return []

        def _run() -> list[str]:
            conn = self._get_connection()
            ids: list[str] = []
            with self._write_lock:
                for chunk in chunks:
                    conn.execute(
                        """INSERT INTO chunks (chunk_id, doc_id, content, chunk_index, parent_chunk_id,
                           section_title, source_page, token_count, content_hash, event_time, metadata_json)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (chunk.chunk_id, chunk.doc_id, chunk.content, chunk.chunk_index,
                         chunk.parent_chunk_id, chunk.section_title, chunk.source_page,
                         chunk.token_count, chunk.content_hash, chunk.event_time, chunk.metadata_json),
                    )
                    try:
                        conn.execute(
                            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                            (chunk.chunk_id, _serialize_f32(chunk.embedding)),
                        )
                    except sqlite3.OperationalError:
                        pass
                    ids.append(chunk.chunk_id)
                # Update document chunk count
                if chunks:
                    doc_id = chunks[0].doc_id
                    conn.execute(
                        "UPDATE documents SET total_chunks = total_chunks + ? WHERE doc_id = ?",
                        (len(chunks), doc_id),
                    )
                conn.commit()
            return ids

        return await asyncio.to_thread(_run)

    async def get_chunk_by_id(self, chunk_id: str) -> ChunkRecord | None:
        """Fetch a single chunk by ID."""
        row = await self.fetch_one("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,))
        if row is None:
            return None
        # Fetch embedding from vec table
        embedding: list[float] = []
        vec_row = await self.fetch_one(
            "SELECT embedding FROM chunks_vec WHERE chunk_id = ?", (chunk_id,)
        )
        if vec_row is not None:
            embedding = _deserialize_f32(vec_row["embedding"])
        return ChunkRecord(
            chunk_id=row["chunk_id"], doc_id=row["doc_id"],
            content=row["content"], embedding=embedding,
            chunk_index=row["chunk_index"],
            parent_chunk_id=row["parent_chunk_id"],
            section_title=row["section_title"],
            source_page=row["source_page"],
            token_count=row["token_count"],
            content_hash=row["content_hash"],
            event_time=row["event_time"],
            metadata_json=row["metadata_json"],
        )

    async def get_chunks_by_doc_id(self, doc_id: str) -> list[ChunkRecord]:
        """Fetch all chunks for a document, ordered by chunk_index."""
        rows = await self.fetch_all(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index",
            (doc_id,),
        )
        results: list[ChunkRecord] = []
        for row in rows:
            results.append(ChunkRecord(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                content=row["content"], embedding=[],
                chunk_index=row["chunk_index"],
                parent_chunk_id=row["parent_chunk_id"],
                section_title=row["section_title"],
                source_page=row["source_page"],
                token_count=row["token_count"],
                content_hash=row["content_hash"],
                event_time=row["event_time"],
                metadata_json=row["metadata_json"],
            ))
        return results

    async def get_parent_chunk(self, chunk_id: str) -> ChunkRecord | None:
        """Fetch the parent chunk for a given chunk."""
        row = await self.fetch_one(
            "SELECT parent_chunk_id FROM chunks WHERE chunk_id = ?", (chunk_id,)
        )
        if row is None or row["parent_chunk_id"] is None:
            return None
        return await self.get_chunk_by_id(row["parent_chunk_id"])

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    async def vector_search(self, embedding: list[float], top_k: int = 50) -> list[ChunkResult]:
        """Dense vector search using sqlite-vec cosine distance."""
        def _run() -> list[ChunkResult]:
            conn = self._get_connection()
            blob = _serialize_f32(embedding)
            try:
                rows = conn.execute(
                    """SELECT cv.chunk_id, cv.distance,
                              c.doc_id, c.content, c.chunk_index, c.parent_chunk_id,
                              c.section_title, c.source_page, c.token_count,
                              d.source_uri, d.source_type
                       FROM chunks_vec cv
                       JOIN chunks c ON c.chunk_id = cv.chunk_id
                       JOIN documents d ON d.doc_id = c.doc_id
                       WHERE d.status = 'active'
                       AND cv.embedding MATCH ?
                       ORDER BY cv.distance
                       LIMIT ?""",
                    (blob, top_k),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.error("vector_search_failed", error=str(e))
                return []

            results: list[ChunkResult] = []
            for row in rows:
                # sqlite-vec returns distance; convert to similarity score
                score = 1.0 - float(row["distance"])
                results.append(ChunkResult(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                    content=row["content"], score=score,
                    chunk_index=row["chunk_index"],
                    parent_chunk_id=row["parent_chunk_id"],
                    section_title=row["section_title"],
                    source_page=row["source_page"],
                    token_count=row["token_count"],
                    source_uri=row["source_uri"],
                    source_type=row["source_type"],
                ))
            return results

        with metrics.timer("vector_search"):
            return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    async def bm25_search(self, query: str, top_k: int = 50) -> list[ChunkResult]:
        """BM25 full-text search using FTS5."""
        def _run() -> list[ChunkResult]:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    """SELECT c.chunk_id, c.doc_id, c.content, c.chunk_index,
                              c.parent_chunk_id, c.section_title, c.source_page, c.token_count,
                              d.source_uri, d.source_type,
                              rank AS bm25_score
                       FROM chunks_fts fts
                       JOIN chunks c ON c.rowid = fts.rowid
                       JOIN documents d ON d.doc_id = c.doc_id
                       WHERE chunks_fts MATCH ?
                       AND d.status = 'active'
                       ORDER BY rank
                       LIMIT ?""",
                    (query, top_k),
                ).fetchall()
            except sqlite3.OperationalError as e:
                logger.error("bm25_search_failed", error=str(e))
                return []

            results: list[ChunkResult] = []
            for row in rows:
                # FTS5 rank is negative (lower = better); normalize to positive score
                score = -float(row["bm25_score"]) if row["bm25_score"] else 0.0
                results.append(ChunkResult(
                    chunk_id=row["chunk_id"], doc_id=row["doc_id"],
                    content=row["content"], score=score,
                    chunk_index=row["chunk_index"],
                    parent_chunk_id=row["parent_chunk_id"],
                    section_title=row["section_title"],
                    source_page=row["source_page"],
                    token_count=row["token_count"],
                    source_uri=row["source_uri"],
                    source_type=row["source_type"],
                ))
            return results

        with metrics.timer("bm25_search"):
            return await asyncio.to_thread(_run)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    async def log_request(
        self,
        request_id: str,
        query: str,
        sub_queries: list[str] | None = None,
        crag_verdict: str | None = None,
        faithfulness: str | None = None,
        retrieved_chunks: list[str] | None = None,
        response_length: int = 0,
        total_latency_ms: int = 0,
    ) -> None:
        """Log a request to the request_log table."""
        await self.execute(
            """INSERT INTO request_log (request_id, query, sub_queries, crag_verdict,
               faithfulness, retrieved_chunks, response_length, total_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                request_id, query,
                json.dumps(sub_queries) if sub_queries else None,
                crag_verdict, faithfulness,
                json.dumps(retrieved_chunks) if retrieved_chunks else None,
                response_length, total_latency_ms,
            ),
        )

    async def log_failed_ingest(
        self,
        failure_id: str,
        source_uri: str,
        error_message: str,
        parser_attempted: str | None = None,
    ) -> None:
        """Log a failed ingestion attempt."""
        await self.execute(
            """INSERT INTO failed_ingests (failure_id, source_uri, error_message, parser_attempted)
               VALUES (?, ?, ?, ?)""",
            (failure_id, source_uri, error_message, parser_attempted),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("database_closed")
