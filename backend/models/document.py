"""Normalized document representation for multimodal inputs in OmniFlow."""

import uuid
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Conceptual input modality types supported by OmniFlow."""

    TEXT = "text"
    IMAGE = "image"
    PDF = "pdf"
    AUDIO = "audio"


class ExtractionMethod(str, Enum):
    """Method utilized by processors to extract text/information."""

    DIRECT_INPUT = "direct_input"
    NATIVE_TEXT = "native_text"
    OCR = "ocr"
    MIXED = "mixed"
    SPEECH_TO_TEXT = "speech_to_text"


class NormalizedDocument(BaseModel):
    """Standardized representation of an ingested document or input.

    Represents the output of multimodal extraction without storing heavy raw binaries.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the normalized document.",
    )
    filename: str = Field(
        ...,
        description="Original name of the uploaded or referenced file.",
    )
    source_type: SourceType = Field(
        ...,
        description="Categorical modality of the source input.",
    )
    mime_type: str = Field(
        ...,
        description="MIME type of the source file (e.g. application/pdf, image/png).",
    )
    content: str = Field(
        default="",
        description="Extracted, normalized text or transcript from the input.",
    )
    extraction_method: ExtractionMethod = Field(
        ...,
        description="Method used to extract content (native, OCR, STT, direct).",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score if provided by the underlying engine.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary safe metadata (page numbers, audio duration, etc.).",
    )
    detected_urls: list[str] = Field(
        default_factory=list,
        description="Detected URLs (e.g. YouTube links) extracted from document text.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during extraction or processing.",
    )
