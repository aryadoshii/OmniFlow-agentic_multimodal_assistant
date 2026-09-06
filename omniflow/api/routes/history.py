"""Conversation history API -- backs the frontend's sidebar.

Purely additive: this never touches /query's agent workflow, RAG, or tool
execution. It only persists/retrieves conversation metadata, message
content, and attachment metadata (never uploaded bytes) via
history_store.py's SQLite-backed functions.
"""

import json
import logging
import uuid
from fastapi import APIRouter

from omniflow.exceptions import ConversationNotFoundError
from omniflow.models.history import (
    AttachmentOut,
    ConversationDetail,
    ConversationSummary,
    CreateConversationRequest,
    MessageOut,
    SaveTurnRequest,
)
from omniflow.services import history_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["History"])


@router.post("", response_model=ConversationSummary, status_code=201)
async def create_conversation(payload: CreateConversationRequest) -> ConversationSummary:
    """Creates a new, empty conversation (called once per new chat, on its first turn)."""
    conversation_id = str(uuid.uuid4())
    row = history_store.create_conversation(conversation_id, payload.title)
    return ConversationSummary(**row, message_count=0, attachment_count=0)


@router.get("", response_model=list[ConversationSummary])
async def list_conversations() -> list[ConversationSummary]:
    """Lists all conversations, most recently updated first (sidebar history)."""
    return [ConversationSummary(**row) for row in history_store.list_conversations()]


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(conversation_id: str) -> ConversationDetail:
    """Fetches one conversation's full messages/attachments (restoring history)."""
    row = history_store.get_conversation(conversation_id)
    if row is None:
        raise ConversationNotFoundError(
            f"No conversation found with id '{conversation_id}'.",
            details={"conversation_id": conversation_id},
        )
    return ConversationDetail(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=[MessageOut(**m) for m in row["messages"]],
        attachments=[AttachmentOut(**a) for a in row["attachments"]],
    )


@router.post("/{conversation_id}/turns", response_model=ConversationSummary)
async def save_turn(conversation_id: str, payload: SaveTurnRequest) -> ConversationSummary:
    """Persists one user query + the resulting assistant response as a turn.

    The assistant's restorable fields (answer/status/warnings/errors/
    execution_trace/normalized_documents) are stored as a single JSON blob
    in the assistant message's `content` -- this is display persistence
    only, never re-parsed by the agent workflow.
    """
    assistant_content = json.dumps(
        {
            "status": payload.status,
            "answer": payload.answer,
            "warnings": payload.warnings,
            "errors": payload.errors,
            "execution_trace": payload.execution_trace,
            "normalized_documents": payload.normalized_documents,
        }
    )
    attachments = [
        (str(uuid.uuid4()), attachment.filename, attachment.mime_type)
        for attachment in payload.attachments
    ]

    row = history_store.add_turn(
        conversation_id=conversation_id,
        user_message_id=str(uuid.uuid4()),
        assistant_message_id=str(uuid.uuid4()),
        query=payload.query,
        assistant_content=assistant_content,
        attachments=attachments,
    )
    if row is None:
        raise ConversationNotFoundError(
            f"No conversation found with id '{conversation_id}'.",
            details={"conversation_id": conversation_id},
        )

    updated = history_store.get_conversation(conversation_id)
    return ConversationSummary(
        id=row["id"],
        title=row["title"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        message_count=len(updated["messages"]) if updated else 0,
        attachment_count=len(updated["attachments"]) if updated else 0,
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str) -> None:
    """Permanently deletes a conversation and its messages/attachments."""
    deleted = history_store.delete_conversation(conversation_id)
    if not deleted:
        raise ConversationNotFoundError(
            f"No conversation found with id '{conversation_id}'.",
            details={"conversation_id": conversation_id},
        )
