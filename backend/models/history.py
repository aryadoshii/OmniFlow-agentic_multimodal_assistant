"""Pydantic schemas for the conversation history API (frontend sidebar).

These mirror history_store.py's SQLite rows at the API boundary. Storage
concerns (row shape, SQL) stay in history_store.py; this module only
defines what the API actually accepts/returns.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """One row in the sidebar's history list."""

    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    attachment_count: int


class MessageOut(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class AttachmentOut(BaseModel):
    id: str
    filename: str
    mime_type: str
    created_at: str


class ConversationDetail(BaseModel):
    """Full conversation, returned when restoring history."""

    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageOut]
    attachments: list[AttachmentOut]


class CreateConversationRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class AttachmentIn(BaseModel):
    filename: str
    mime_type: str


class SaveTurnRequest(BaseModel):
    """One request/response exchange to persist against an existing
    conversation -- the user's query, the assistant's resulting answer and
    metadata, and any attachment metadata. Mirrors the fields of
    OmniFlowResponse (see backend/models/response.py) that are actually
    worth restoring later; nothing here re-enters the agent workflow."""

    query: str = Field(..., min_length=1)
    status: str
    answer: str | None = None
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    execution_trace: dict | None = None
    normalized_documents: list[dict] = Field(default_factory=list)
    evidence: list[dict] = Field(
        default_factory=list,
        description="EvidenceReference entries (see backend/models/evidence.py), stored "
        "opaquely for restoring a past conversation's Sources section.",
    )
    cross_source_analysis: dict | None = Field(
        default=None,
        description="CrossSourceAnalysis (see backend/agents/cross_source.py), stored "
        "opaquely for restoring a past conversation's cross-source analysis section.",
    )
    attachments: list[AttachmentIn] = Field(default_factory=list)
