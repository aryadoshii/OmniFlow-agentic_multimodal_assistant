"""SQLite-backed conversation history store.

A small, dependency-free persistence layer for the frontend's conversation
sidebar: it stores conversation metadata, the user/assistant messages
exchanged, and attachment *metadata* only (filename/mime_type -- never the
uploaded bytes themselves, which stay ephemeral per the app's existing
upload-handling design; see temp_manager.py). This is display/history
persistence only -- it never feeds back into /query's agent workflow, RAG,
or any tool.

Uses the stdlib sqlite3 module directly (no ORM): the schema is three small
tables and every query here is a single, simple statement, so an ORM would
add a dependency without buying anything. A new connection is opened per
call rather than held open across requests -- this app has no meaningful
concurrent write load (a single local user's browser), and this avoids any
cross-request/cross-thread connection-sharing concerns entirely.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from backend.config import get_settings

# Anchored to the project root (this file's parent's parent's parent),
# independent of process cwd -- same reasoning as config.py's
# _PROJECT_ROOT_ENV_FILE. Settings.history_db_path (default
# "database/omniflow_history.db") is resolved against this root unless it
# is already absolute, so a deployment can override it (e.g. HISTORY_DB_PATH
# pointing at a mounted disk) without touching this module. Not a committed
# artifact: it holds a given machine's local conversation history.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_db_path() -> Path:
    configured = Path(get_settings().history_db_path)
    return configured if configured.is_absolute() else _PROJECT_ROOT / configured


DB_PATH = _resolve_db_path()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Creates the history tables if they don't already exist.

    Safe to call on every startup (see main.py's lifespan) -- CREATE TABLE
    IF NOT EXISTS is idempotent and this performs no destructive migration.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attachments (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments(conversation_id)"
        )


def create_conversation(conversation_id: str, title: str) -> dict:
    now = _now()
    with _connection() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title, now, now),
        )
    return {"id": conversation_id, "title": title, "created_at": now, "updated_at": now}


def list_conversations() -> list[dict]:
    with _connection() as conn:
        rows = conn.execute(
            """
            SELECT
                c.id, c.title, c.created_at, c.updated_at,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                (SELECT COUNT(*) FROM attachments a WHERE a.conversation_id = c.id) AS attachment_count
            FROM conversations c
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connection() as conn:
        conv_row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if conv_row is None:
            return None

        message_rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC",
            (conversation_id,),
        ).fetchall()
        attachment_rows = conn.execute(
            "SELECT id, filename, mime_type, created_at FROM attachments "
            "WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC",
            (conversation_id,),
        ).fetchall()

    return {
        **dict(conv_row),
        "messages": [dict(row) for row in message_rows],
        "attachments": [dict(row) for row in attachment_rows],
    }


def delete_conversation(conversation_id: str) -> bool:
    with _connection() as conn:
        cursor = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0


def add_turn(
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    query: str,
    assistant_content: str,
    attachments: list[tuple[str, str, str]],
) -> dict | None:
    """Appends one user message + one assistant message (+ any attachment
    metadata) to a conversation, and bumps its updated_at. Returns the
    updated conversation summary, or None if the conversation doesn't exist.

    Args:
        attachments: (attachment_id, filename, mime_type) triples.
        assistant_content: Pre-serialized (JSON) string capturing the
                            assistant response fields worth restoring later
                            (answer, status, warnings, errors, execution
                            trace, processed documents) -- this module
                            stores it opaquely and has no opinion on its
                            shape; that belongs to the API layer.
    """
    now = _now()
    with _connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if exists is None:
            return None

        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
            (user_message_id, conversation_id, query, now),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
            (assistant_message_id, conversation_id, assistant_content, now),
        )
        for attachment_id, filename, mime_type in attachments:
            conn.execute(
                "INSERT INTO attachments (id, conversation_id, filename, mime_type, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (attachment_id, conversation_id, filename, mime_type, now),
            )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id)
        )
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
    return dict(row)
