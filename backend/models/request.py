"""User request and input domain models."""

from typing import Any
from pydantic import BaseModel, Field


class UploadedInput(BaseModel):
    """Metadata representing an uploaded file input before extraction."""

    filename: str = Field(..., description="Original filename of the upload.")
    mime_type: str = Field(..., description="MIME type detected for the file.")
    size_bytes: int = Field(..., ge=0, description="File size in bytes.")


class UserRequest(BaseModel):
    """Standardized user request schema."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="The user's query or instruction to the assistant.",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional session or conversation identifier.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional client metadata or contextual flags.",
    )
