"""Memory API routes: preferences and conversation history."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.schemas import (
    PreferenceCreate, PreferenceResponse, ConversationSummary, ErrorResponse,
)

router = APIRouter(prefix="/v1/memory", tags=["Memory"])


async def _get_preferences(request: Request):
    return request.app.state.pipeline.preferences


async def _get_episodic(request: Request):
    return request.app.state.pipeline.episodic_memory


@router.get("/preferences", response_model=list[PreferenceResponse], summary="List preferences")
async def list_preferences(store=Depends(_get_preferences)):
    """List all active user preferences."""
    prefs = await store.get_active_preferences()
    return [PreferenceResponse(**p) for p in prefs]


@router.put("/preferences", response_model=PreferenceResponse, summary="Add/update preference")
async def add_preference(body: PreferenceCreate, store=Depends(_get_preferences)):
    """Add or update a user preference."""
    pref_id = await store.add_preference(
        category=body.category,
        preference=body.preference,
        supersedes_id=body.supersedes_id,
    )
    return PreferenceResponse(
        pref_id=pref_id,
        category=body.category,
        preference=body.preference,
        is_active=True,
        created_at="now",
    )


@router.delete("/preferences/{pref_id}", summary="Deactivate preference")
async def deactivate_preference(pref_id: str, store=Depends(_get_preferences)):
    """Deactivate a user preference."""
    await store.deactivate_preference(pref_id)
    return {"status": "deactivated", "pref_id": pref_id}


@router.get("/conversations", response_model=list[ConversationSummary], summary="List conversations")
async def list_conversations(limit: int = 50, episodic=Depends(_get_episodic)):
    """List past conversations with summaries."""
    convos = await episodic.get_all_conversations(limit=limit)
    return [ConversationSummary(**c) for c in convos]
