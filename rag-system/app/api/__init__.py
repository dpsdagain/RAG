"""API layer – FastAPI route schemas and endpoint definitions."""

from __future__ import annotations

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    SourceCitation,
    IngestFileResponse,
    IngestURLRequest,
    IngestURLResponse,
    PreferenceCreate,
    PreferenceResponse,
    ConversationSummary,
    StatsResponse,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    "ChatRequest",
    "ChatResponse",
    "ChatStreamEvent",
    "SourceCitation",
    "IngestFileResponse",
    "IngestURLRequest",
    "IngestURLResponse",
    "PreferenceCreate",
    "PreferenceResponse",
    "ConversationSummary",
    "StatsResponse",
    "HealthResponse",
    "ErrorResponse",
]
