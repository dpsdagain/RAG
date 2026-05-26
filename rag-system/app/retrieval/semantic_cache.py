"""In-memory semantic cache for RAG responses.

Caches query embeddings and their responses. Lookups find cached entries
whose query embedding has cosine similarity above a threshold, avoiding
redundant retrieval and generation for semantically equivalent queries.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from app.infrastructure.observability import get_logger, metrics

logger = get_logger("retrieval.semantic_cache")


@dataclass
class CacheEntry:
    """A cached response with its query embedding and metadata."""
    query_embedding: list[float]
    response: str
    sources: list[dict[str, Any]]
    timestamp: float
    hit_count: int = 0
    conversation_id: str = ""


class SemanticCache:
    """Thread-safe in-memory semantic cache with LRU eviction and TTL.

    Stores query embeddings and their generated responses. On lookup,
    computes cosine similarity between the new query and all cached
    entries, returning the cached response if similarity exceeds the
    threshold and the entry has not expired.
    """

    def __init__(
        self,
        max_entries: int = 500,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
    ) -> None:
        """Initialize the semantic cache.

        Args:
            max_entries: Maximum number of cached entries.
            similarity_threshold: Minimum cosine similarity for a cache hit.
            ttl_seconds: Time-to-live for cache entries in seconds.
        """
        self._max_entries = max_entries
        self._threshold = similarity_threshold
        self._ttl = ttl_seconds
        self._entries: list[CacheEntry] = []
        self._lock = Lock()
        self._total_lookups = 0
        self._total_hits = 0

    def lookup(self, query_embedding: list[float]) -> CacheEntry | None:
        """Check cache for a semantically similar query.

        Args:
            query_embedding: Embedding of the incoming query.

        Returns:
            CacheEntry if a match is found, None otherwise.
        """
        with self._lock:
            self._total_lookups += 1
            now = time.time()

            if not self._entries:
                metrics.increment("cache_misses")
                return None

            query_vec = np.array(query_embedding, dtype=np.float32)
            best_entry: CacheEntry | None = None
            best_sim = 0.0

            for entry in self._entries:
                # Check TTL
                if now - entry.timestamp > self._ttl:
                    continue

                sim = self.cosine_similarity_np(query_vec, np.array(entry.query_embedding, dtype=np.float32))
                if sim > best_sim:
                    best_sim = sim
                    best_entry = entry

            if best_entry is not None and best_sim >= self._threshold:
                best_entry.hit_count += 1
                self._total_hits += 1
                metrics.increment("cache_hits")
                logger.debug(
                    "cache_hit",
                    similarity=round(best_sim, 4),
                    hit_count=best_entry.hit_count,
                )
                return best_entry

            metrics.increment("cache_misses")
            return None

    def store(
        self,
        query_embedding: list[float],
        response: str,
        sources: list[dict[str, Any]],
        conversation_id: str = "",
    ) -> None:
        """Store a new cache entry, evicting LRU if necessary.

        Args:
            query_embedding: Embedding of the query.
            response: Generated response text.
            sources: Serialized source citations.
            conversation_id: Associated conversation ID.
        """
        with self._lock:
            entry = CacheEntry(
                query_embedding=query_embedding,
                response=response,
                sources=sources,
                timestamp=time.time(),
                conversation_id=conversation_id,
            )

            # Evict expired entries
            now = time.time()
            self._entries = [e for e in self._entries if now - e.timestamp <= self._ttl]

            # Evict LRU if at capacity
            if len(self._entries) >= self._max_entries:
                # Sort by timestamp (oldest first), remove oldest
                self._entries.sort(key=lambda e: e.timestamp)
                self._entries.pop(0)

            self._entries.append(entry)
            logger.debug("cache_stored", entry_count=len(self._entries))

    def invalidate_all(self) -> None:
        """Clear the entire cache."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            logger.info("cache_invalidated", cleared=count)

    def get_stats(self) -> dict[str, Any]:
        """Return cache statistics.

        Returns:
            Dict with hit_rate, entry_count, total_lookups, total_hits.
        """
        with self._lock:
            hit_rate = (
                self._total_hits / self._total_lookups
                if self._total_lookups > 0
                else 0.0
            )
            now = time.time()
            ages = [now - e.timestamp for e in self._entries]
            avg_age = sum(ages) / len(ages) if ages else 0.0

            return {
                "hit_rate": round(hit_rate, 4),
                "entry_count": len(self._entries),
                "total_lookups": self._total_lookups,
                "total_hits": self._total_hits,
                "avg_age_seconds": round(avg_age, 1),
            }

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector.
            b: Second vector.

        Returns:
            Cosine similarity in [-1, 1].
        """
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        return SemanticCache.cosine_similarity_np(a_arr, b_arr)

    @staticmethod
    def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two numpy vectors."""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(dot / (norm_a * norm_b))
