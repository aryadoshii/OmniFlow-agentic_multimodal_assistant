"""Ingestion API route supporting text and multiple file uploads."""

import logging
from typing import Annotated
from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import BaseModel, Field
from omniflow.exceptions import InvalidInputError
from omniflow.models.document import NormalizedDocument
from omniflow.services.ingestion import IngestionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


class IngestionResponse(BaseModel):
    """Standardized response schema returned by the ingestion endpoint."""

    documents: list[NormalizedDocument] = Field(
        ...,
        description="List of normalized documents extracted from the inputs.",
    )
    total_count: int = Field(
        ...,
        description="Total number of documents successfully processed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings aggregated across all processed inputs.",
    )


def get_ingestion_service() -> IngestionService:
    """Dependency provider for IngestionService."""
    return IngestionService()


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_content(
    text: Annotated[str | None, Form(description="Optional plain text input.")] = None,
    files: Annotated[
        list[UploadFile] | None,
        File(description="Optional list of file uploads (PDF, image, audio)."),
    ] = None,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    """Ingests text and/or multiple files, returning standardized NormalizedDocuments."""
    has_text = bool(text and text.strip())
    has_files = bool(files and len(files) > 0)

    if not has_text and not has_files:
        raise InvalidInputError(
            "No input provided. Please provide either text or one or more file uploads."
        )

    file_payloads: list[tuple[bytes, str, str | None]] = []

    if files:
        for file in files:
            filename = file.filename or "upload"
            content = await file.read()
            declared_mime = file.content_type
            file_payloads.append((content, filename, declared_mime))

    normalized_docs = service.ingest_inputs(
        text=text if has_text else None,
        files=file_payloads if file_payloads else None,
    )

    all_warnings: list[str] = []
    for doc in normalized_docs:
        all_warnings.extend(doc.warnings)

    return IngestionResponse(
        documents=normalized_docs,
        total_count=len(normalized_docs),
        warnings=all_warnings,
    )
