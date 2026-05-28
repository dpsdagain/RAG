"""Configuration management for the RAG system.

Loads settings from YAML config file with env var overrides (RAG_ prefix).
Uses pydantic-settings for validation and type coercion.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Nested config groups
# ---------------------------------------------------------------------------

class DatabaseConfig(BaseModel):
    """Database storage settings."""
    db_path: str = Field(default="./data/rag.db", description="Path to SQLite database file")


class EmbeddingConfig(BaseModel):
    """Embedding model settings."""
    model_path: str = Field(default="BAAI/bge-small-en-v1.5", description="HuggingFace model ID or local path")
    dim: int = Field(default=384, description="Embedding vector dimensions")
    onnx_num_threads: int = Field(default=4, description="ONNX Runtime CPU threads")
    batch_size: int = Field(default=64, description="Sentences per ONNX inference batch")


class LLMConfig(BaseModel):
    """Cloud LLM provider settings."""
    provider: str = Field(default="ollama", description="LLM provider: ollama, openai, anthropic")
    base_url: str = Field(default="http://localhost:11434", description="Provider API base URL")
    model: str = Field(default="llama3.2", description="Model name/ID")
    api_key: str = Field(default="", description="API key (empty for Ollama)")
    temperature: float = Field(default=0.1, description="Generation temperature")
    max_tokens: int = Field(default=2048, description="Max tokens for generation")
    timeout_seconds: float = Field(default=60.0, description="Request timeout in seconds")


class LlamaParseConfig(BaseModel):
    """LlamaParse cloud parsing settings."""
    api_key: str = Field(default="", description="LlamaParse API key")


class CloudVLMConfig(BaseModel):
    """Cloud Vision-Language Model settings."""
    provider: str = Field(default="gemini", description="VLM provider: gemini, openai")
    api_key: str = Field(default="", description="VLM API key")


class RetrievalConfig(BaseModel):
    """Retrieval pipeline settings."""
    dense_top_k: int = Field(default=50, description="Dense search top-K results")
    sparse_top_k: int = Field(default=50, description="BM25 sparse search top-K results")
    rrf_k: int = Field(default=60, description="RRF smoothing constant")
    rrf_top_n: int = Field(default=100, description="Results after RRF fusion")
    rerank_top_k: int = Field(default=15, description="Final results after reranking")
    context_budget_tokens: int = Field(default=12000, description="Max tokens for LLM context")
    rerank_model: str = Field(default="ms-marco-MiniLM-L-12-v2", description="FlashRank model name")


class PipelineConfig(BaseModel):
    """RAG pipeline feature toggles."""
    # CRAG defaults OFF — costs a cloud LLM call per query and usually returns
    # SUFFICIENT anyway. Faithfulness is the higher-value safety gate.
    crag_enabled: bool = Field(default=False, description="Enable CRAG quality gate")
    decomposition_enabled: bool = Field(default=True, description="Enable query decomposition")
    faithfulness_check_enabled: bool = Field(default=True, description="Enable post-gen faithfulness check")


class MemoryConfig(BaseModel):
    """Memory engine settings."""
    working_memory_max_turns: int = Field(default=10, description="Max conversation turns in working memory")
    episodic_memory_top_k: int = Field(default=3, description="Top-K past conversations to recall")


class CacheConfig(BaseModel):
    """Semantic cache settings."""
    max_entries: int = Field(default=500, description="Max cache entries")
    # Lowered from 0.95 — see configs/config.yaml for the why.
    similarity_threshold: float = Field(default=0.85, description="Min cosine similarity for cache hit")
    ttl_seconds: int = Field(default=3600, description="Cache entry TTL in seconds")


class ChunkingConfig(BaseModel):
    """Chunking strategy settings."""
    max_chunk_tokens: int = Field(default=512, description="Max tokens per chunk")
    min_chunk_tokens: int = Field(default=50, description="Min tokens per chunk (merge if smaller)")
    semantic_threshold: float = Field(default=0.3, description="Inter-sentence similarity threshold for splitting")


class ServerConfig(BaseModel):
    """HTTP server settings."""
    host: str = Field(default="0.0.0.0", description="Bind host")
    port: int = Field(default=8000, description="Bind port")
    api_key: str = Field(default="change-me-in-production", description="Static API key for auth")


class ObservabilityConfig(BaseModel):
    """Logging and metrics settings."""
    log_level: str = Field(default="INFO", description="Log level: DEBUG, INFO, WARNING, ERROR")
    log_file: str = Field(default="./logs/rag.log", description="Log file path")


class IngestionConfig(BaseModel):
    """Ingestion worker settings."""
    watch_directory: str = Field(default="", description="Directory to watch for new files (empty = disabled)")
    batch_size: int = Field(default=32, description="Chunks per embedding batch")
    max_file_size_mb: int = Field(default=100, description="Max file size for ingestion in MB")


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------

class Settings(BaseModel):
    """Root configuration for the RAG system.

    Loaded from configs/config.yaml with env var overrides using RAG_ prefix.
    Example: RAG_SERVER__PORT=9000 overrides server.port.
    """
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    llamaparse: LlamaParseConfig = Field(default_factory=LlamaParseConfig)
    cloud_vlm: CloudVLMConfig = Field(default_factory=CloudVLMConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge overrides into base dict."""
    result = base.copy()
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply RAG_ prefixed env vars as config overrides.

    Format: RAG_SECTION__KEY=value → data[section][key] = value
    Double underscore (__) separates nesting levels.
    """
    prefix = "RAG_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        parts = env_key[len(prefix):].lower().split("__")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Attempt type coercion for common types
        final_key = parts[-1]
        if env_value.lower() in ("true", "false"):
            current[final_key] = env_value.lower() == "true"
        elif env_value.isdigit():
            current[final_key] = int(env_value)
        else:
            try:
                current[final_key] = float(env_value)
            except ValueError:
                current[final_key] = env_value
    return data


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML file with env var overrides.

    Args:
        config_path: Path to YAML config file. If None, searches for
                     configs/config.yaml relative to the project root.

    Returns:
        Validated Settings instance.
    """
    # Load .env so RAG_* secrets reach os.environ before override merge.
    # Existing env vars take precedence; .env only fills gaps.
    try:
        from dotenv import load_dotenv
        for env_path in (
            Path(__file__).resolve().parent.parent.parent / ".env",
            Path.cwd() / ".env",
        ):
            if env_path.exists():
                load_dotenv(env_path, override=False)
                break
    except ImportError:
        pass

    data: dict[str, Any] = {}

    if config_path is None:
        # Search relative to this file's location
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml",
            Path.cwd() / "configs" / "config.yaml",
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
                if isinstance(raw, dict):
                    data = raw

    data = _apply_env_overrides(data)
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton Settings instance.

    Uses lru_cache to ensure only one Settings object exists.
    Call get_settings.cache_clear() to force reload.
    """
    return load_settings()
