"""Chat API route: POST /v1/chat/completions."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, ErrorResponse

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


async def _get_pipeline(request: Request):
    return request.app.state.pipeline


def _format_sse(event_name: str, data) -> str:
    """Encode an event as a proper SSE frame (named event + JSON data)."""
    payload = json.dumps(data) if not isinstance(data, str) else json.dumps(data)
    return f"event: {event_name}\ndata: {payload}\n\n"


@router.post(
    "/completions",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Generate a RAG response",
)
async def chat_completions(
    body: ChatRequest,
    pipeline=Depends(_get_pipeline),
):
    """Execute the RAG pipeline for a user query.

    Supports both streaming (SSE) and non-streaming responses.
    """
    if body.stream:
        async def event_stream():
            try:
                async for evt in pipeline.execute_stream(
                    query=body.query,
                    conversation_id=body.conversation_id,
                ):
                    yield _format_sse(
                        evt.get("event", "message"),
                        evt.get("data"),
                    )
            except Exception as e:
                yield _format_sse("error", str(e))

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await pipeline.execute(
        query=body.query,
        conversation_id=body.conversation_id,
        stream=False,
    )
    return result
