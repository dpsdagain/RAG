"""API route modules."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes.chat import router as chat_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.memory import router as memory_router
from app.api.routes.admin import router as admin_router

all_routers: list[APIRouter] = [
    chat_router,
    ingest_router,
    memory_router,
    admin_router,
]

__all__ = ["all_routers"]
