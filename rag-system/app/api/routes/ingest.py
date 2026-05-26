"""Ingestion API routes: file upload and URL ingestion."""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Request, UploadFile, File

from app.api.schemas import IngestFileResponse, IngestURLRequest, IngestURLResponse, ErrorResponse

router = APIRouter(prefix="/v1/ingest", tags=["Ingestion"])


async def _get_worker(request: Request):
    return request.app.state.ingestion_worker


@router.post(
    "/file",
    response_model=IngestFileResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ingest a file",
)
async def ingest_file(
    file: UploadFile = File(...),
    worker=Depends(_get_worker),
):
    """Upload and ingest a document file."""
    # Save uploaded file to temp location
    suffix = Path(file.filename or "upload").suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = await worker.ingest_file(tmp_path)
        return IngestFileResponse(
            doc_id=result.doc_id,
            filename=file.filename or "unknown",
            status=result.status,
            chunks_created=result.chunks_created,
            error_message=result.error_message,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post(
    "/url",
    response_model=IngestURLResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Ingest from URL",
)
async def ingest_url(
    body: IngestURLRequest,
    worker=Depends(_get_worker),
):
    """Crawl and ingest content from a URL."""
    result = await worker.ingest_url(body.url)
    return IngestURLResponse(
        doc_id=result.doc_id,
        url=body.url,
        status=result.status,
        chunks_created=result.chunks_created,
        error_message=result.error_message,
    )
