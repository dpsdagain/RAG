"""Observability: structured logging and metrics collection.

Provides structured JSON logging via structlog and a lightweight
in-process metrics collector for latency tracking and counters.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from threading import Lock
from typing import Any, AsyncIterator, Iterator

import structlog


# ---------------------------------------------------------------------------
# Structlog configuration
# ---------------------------------------------------------------------------

_configured = False


def _configure_structlog(log_level: str = "INFO", json_output: bool = True) -> None:
    """Configure structlog processors and output format."""
    global _configured
    if _configured:
        return

    import logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named structured logger.

    Args:
        name: Logger name, typically the module name.

    Returns:
        A bound structlog logger instance.
    """
    if not _configured:
        log_level = os.environ.get("RAG_LOG_LEVEL", "INFO")
        json_mode = os.environ.get("RAG_LOG_JSON", "false").lower() != "false"
        _configure_structlog(log_level=log_level, json_output=json_mode)

    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class MetricsCollector:
    """In-process metrics collector for latencies and counters.

    Thread-safe. Stores raw latency samples and counter values.
    Provides aggregated statistics via get_stats().

    This is intentionally lightweight — no external dependencies,
    no Prometheus, no StatsD. For a single-user system, this is sufficient.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._start_time = time.monotonic()
        self._max_samples = 10000  # Keep last N samples per metric

    def record_latency(self, operation: str, ms: float) -> None:
        """Record a latency measurement in milliseconds.

        Args:
            operation: Name of the operation (e.g., 'retrieval', 'rerank').
            ms: Latency in milliseconds.
        """
        with self._lock:
            samples = self._latencies[operation]
            samples.append(ms)
            # Circular buffer: discard oldest when exceeding max
            if len(samples) > self._max_samples:
                self._latencies[operation] = samples[-self._max_samples:]

    def increment(self, counter: str, value: int = 1) -> None:
        """Increment a counter.

        Args:
            counter: Counter name (e.g., 'cache_hits', 'queries_total').
            value: Amount to increment by.
        """
        with self._lock:
            self._counters[counter] += value

    def get_counter(self, counter: str) -> int:
        """Get current value of a counter."""
        with self._lock:
            return self._counters.get(counter, 0)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated statistics for all metrics.

        Returns:
            Dict containing latency percentiles per operation,
            counter values, RSS memory, and uptime.
        """
        with self._lock:
            stats: dict[str, Any] = {
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
                "rss_mb": self.get_rss_mb(),
                "counters": dict(self._counters),
                "latencies": {},
            }

            for operation, samples in self._latencies.items():
                if not samples:
                    continue
                sorted_samples = sorted(samples)
                n = len(sorted_samples)
                stats["latencies"][operation] = {
                    "count": n,
                    "avg_ms": round(sum(sorted_samples) / n, 2),
                    "p50_ms": round(sorted_samples[n // 2], 2),
                    "p95_ms": round(sorted_samples[int(n * 0.95)], 2) if n >= 20 else None,
                    "p99_ms": round(sorted_samples[int(n * 0.99)], 2) if n >= 100 else None,
                    "min_ms": round(sorted_samples[0], 2),
                    "max_ms": round(sorted_samples[-1], 2),
                }

            return stats

    @staticmethod
    def get_rss_mb() -> float:
        """Get current process RSS (Resident Set Size) in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 1)
        except ImportError:
            # Fallback for Windows without psutil
            try:
                import ctypes
                import ctypes.wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                ctypes.windll.psapi.GetProcessMemoryInfo(  # type: ignore[attr-defined]
                    ctypes.windll.kernel32.GetCurrentProcess(),  # type: ignore[attr-defined]
                    ctypes.byref(counters),
                    counters.cb,
                )
                return round(counters.WorkingSetSize / (1024 * 1024), 1)
            except Exception:
                return 0.0

    @contextmanager
    def timer(self, operation: str) -> Iterator[dict[str, float]]:
        """Synchronous context manager for timing operations.

        Usage:
            with metrics.timer('my_operation') as t:
                do_something()
            # t['elapsed_ms'] is available after the block
        """
        result: dict[str, float] = {"elapsed_ms": 0.0}
        start = time.perf_counter()
        try:
            yield result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result["elapsed_ms"] = round(elapsed_ms, 2)
            self.record_latency(operation, elapsed_ms)

    @asynccontextmanager
    async def async_timer(self, operation: str) -> AsyncIterator[dict[str, float]]:
        """Async context manager for timing operations.

        Usage:
            async with metrics.async_timer('my_operation') as t:
                await do_something_async()
        """
        result: dict[str, float] = {"elapsed_ms": 0.0}
        start = time.perf_counter()
        try:
            yield result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            result["elapsed_ms"] = round(elapsed_ms, 2)
            self.record_latency(operation, elapsed_ms)

    def reset(self) -> None:
        """Reset all metrics. Useful for testing."""
        with self._lock:
            self._latencies.clear()
            self._counters.clear()
            self._start_time = time.monotonic()


# Singleton instance
metrics = MetricsCollector()
