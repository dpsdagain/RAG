"""Chat API route: POST /v1/chat/completions."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest, ChatResponse, ChatStreamEvent, ErrorResponse

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


async def _get_pipeline(request: Request):
    return request.app.state.pipeline


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
                # For now, execute non-streaming and wrap as SSE
                result = await pipeline.execute(
                    query=body.query,
                    conversation_id=body.conversation_id,
                    stream=False,
                )
                # Emit tokens
                for i in range(0, len(result.response), 20):
                    chunk = result.response[i:i + 20]
                    event = ChatStreamEvent(event="token", data=chunk)
                    yield f"data: {event.model_dump_json()}\n\n"

                # Emit sources
                sources_event = ChatStreamEvent(
                    event="sources",
                    data={"sources": [s.model_dump() for s in result.sources]},
                )
                yield f"data: {sources_event.model_dump_json()}\n\n"

                # Emit done
                done_event = ChatStreamEvent(
                    event="done",
                    data={
                        "conversation_id": result.conversation_id,
                        "crag_verdict": result.crag_verdict,
                        "latency_ms": result.latency_ms,
                    },
                )
                yield f"data: {done_event.model_dump_json()}\n\n"

            except Exception as e:
                error_event = ChatStreamEvent(event="error", data=str(e))
                yield f"data: {error_event.model_dump_json()}\n\n"

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
