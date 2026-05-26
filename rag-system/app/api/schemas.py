"""Pydantic v2 schemas for all API request/response models.

Defines all data contracts used by the FastAPI routes, ensuring strict
type validation and automatic OpenAPI documentation generation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for POST /v1/chat/completions."""
    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    conversation_id: str | None = Field(default=None, description="Thread ID for conversation continuity")
    stream: bool = Field(default=True, description="Enable SSE streaming response")


class SourceCitation(BaseModel):
    """A single source citation from retrieved chunks."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content_snippet: str = Field(..., max_length=300, description="Relevant excerpt from the chunk")
    source_uri: str = Field(..., description="Original document URI")
    source_type: str = Field(..., description="Document type: pdf, code, web, etc.")
    page: int | None = Field(default=None, description="Page number in original document")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class ChatResponse(BaseModel):
    """Response body for non-streaming chat requests."""
    response: str = Field(..., description="Generated response text")
    sources: list[SourceCitation] = Field(default_factory=list, description="Source citations")
    conversation_id: str = Field(..., description="Thread ID for this conversation")
    crag_verdict: str | None = Field(default=None, description="CRAG quality gate result")
    faithfulness_result: str | None = Field(default=None, description="Faithfulness check result")
    latency_ms: float = Field(..., description="Total pipeline latency in milliseconds")
    query_type: str = Field(default="simple_retrieval", description="Classified query type")


class ChatStreamEvent(BaseModel):
    """Server-Sent Event for streaming chat responses."""
    event: str = Field(
        ...,
        description="Event type: 'token', 'sources', 'metadata', 'done', 'error'",
    )
    data: str | dict[str, Any] = Field(
        ...,
        description="Event payload: text chunk for 'token', structured data for others",
    )


# ---------------------------------------------------------------------------
# Ingestion API
# ---------------------------------------------------------------------------

class IngestFileResponse(BaseModel):
    """Response for POST /v1/ingest/file."""
    doc_id: str = Field(..., description="Assigned document ID")
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Ingestion status: success, skipped, failed")
    chunks_created: int = Field(default=0, description="Number of chunks generated")
    error_message: str | None = Field(default=None, description="Error message if failed")


class IngestURLRequest(BaseModel):
    """Request body for POST /v1/ingest/url."""
    url: str = Field(..., description="URL to crawl and ingest")


class IngestURLResponse(BaseModel):
    """Response for POST /v1/ingest/url."""
    doc_id: str = Field(..., description="Assigned document ID")
    url: str = Field(..., description="Crawled URL")
    status: str = Field(..., description="Ingestion status: success, skipped, failed")
    chunks_created: int = Field(default=0, description="Number of chunks generated")
    error_message: str | None = Field(default=None, description="Error message if failed")


# ---------------------------------------------------------------------------
# Memory API
# ---------------------------------------------------------------------------

class PreferenceCreate(BaseModel):
    """Request body for creating/updating a user preference."""
    category: str = Field(..., min_length=1, description="Category: output_format, language, domain, etc.")
    preference: str = Field(..., min_length=1, description="Preference text")
    supersedes_id: str | None = Field(default=None, description="ID of preference this supersedes")


class PreferenceResponse(BaseModel):
    """Response for a single user preference."""
    pref_id: str = Field(..., description="Preference ID")
    category: str = Field(..., description="Preference category")
    preference: str = Field(..., description="Preference text")
    is_active: bool = Field(..., description="Whether preference is active")
    created_at: str = Field(..., description="ISO timestamp of creation")


class ConversationSummary(BaseModel):
    """Summary of a past conversation."""
    conversation_id: str = Field(..., description="Conversation ID")
    summary: str = Field(default="", description="LLM-generated summary")
    started_at: str = Field(..., description="ISO timestamp")
    turn_count: int = Field(default=0, description="Number of turns")


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    """System statistics response."""
    total_documents: int = Field(..., description="Total documents ingested")
    total_chunks: int = Field(..., description="Total chunks stored")
    rss_mb: float = Field(..., description="Current process RSS in MB")
    cache_hit_rate: float = Field(default=0.0, description="Semantic cache hit rate")
    avg_latency_ms: float = Field(default=0.0, description="Average query latency in ms")
    uptime_seconds: float = Field(..., description="Process uptime in seconds")
    latency_stats: dict[str, Any] = Field(default_factory=dict, description="Per-operation latency stats")
    counters: dict[str, int] = Field(default_factory=dict, description="Event counters")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status: healthy, degraded, unhealthy")
    version: str = Field(default="0.1.0", description="Application version")
    db_connected: bool = Field(..., description="Database connectivity status")
    embedder_loaded: bool = Field(default=False, description="Embedding model loaded")
    llm_reachable: bool = Field(default=False, description="LLM provider reachable")


class FailedIngest(BaseModel):
    """A failed ingestion record."""
    failure_id: str
    source_uri: str
    error_message: str
    parser_attempted: str | None
    failed_at: str
    retry_count: int
    resolved: bool


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type or code")
    detail: str | None = Field(default=None, description="Detailed error message")
