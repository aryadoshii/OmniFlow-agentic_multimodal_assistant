"""Integration tests for the conversation history API (SQLite-backed sidebar).

Each test gets its own temporary SQLite file (history_store.DB_PATH is
monkeypatched before the app starts) so these tests never touch a real
local omniflow_history.db and never interfere with each other.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import create_app
from backend.services import history_store


@pytest.fixture
def history_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(history_store, "DB_PATH", tmp_path / "test_history.db")
    test_settings = Settings(_env_file=None, APP_ENV="test", LOG_LEVEL="WARNING")
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as client:
        yield client


class TestCreateConversation:
    def test_create_returns_new_conversation(self, history_client: TestClient) -> None:
        response = history_client.post("/conversations", json={"title": "Summarize this PDF"})
        assert response.status_code == 201
        data = response.json()
        assert data["title"] == "Summarize this PDF"
        assert data["message_count"] == 0
        assert data["attachment_count"] == 0
        assert data["id"]

    def test_blank_title_rejected(self, history_client: TestClient) -> None:
        response = history_client.post("/conversations", json={"title": ""})
        assert response.status_code == 422


class TestListConversations:
    def test_empty_history_returns_empty_list(self, history_client: TestClient) -> None:
        response = history_client.get("/conversations")
        assert response.status_code == 200
        assert response.json() == []

    def test_created_conversations_are_listed_most_recently_updated_first(
        self, history_client: TestClient
    ) -> None:
        first = history_client.post("/conversations", json={"title": "First"}).json()
        history_client.post("/conversations", json={"title": "Second"}).json()

        # Touch "First" so it becomes the most recently updated.
        history_client.post(
            f"/conversations/{first['id']}/turns",
            json={"query": "hello", "status": "completed", "answer": "hi"},
        )

        listed = history_client.get("/conversations").json()
        assert [c["title"] for c in listed] == ["First", "Second"]


class TestSaveTurnAndRestore:
    def test_save_turn_persists_query_and_response(self, history_client: TestClient) -> None:
        conv = history_client.post("/conversations", json={"title": "Summarize"}).json()

        response = history_client.post(
            f"/conversations/{conv['id']}/turns",
            json={
                "query": "Summarize this PDF in 3 bullets.",
                "status": "completed",
                "answer": "- a\n- b\n- c",
                "warnings": ["low confidence OCR"],
                "errors": [],
                "execution_trace": {"session_id": None, "total_duration_ms": 12.3, "steps": []},
                "normalized_documents": [{"id": "doc1", "filename": "report.pdf"}],
                "attachments": [{"filename": "report.pdf", "mime_type": "application/pdf"}],
            },
        )
        assert response.status_code == 200
        summary = response.json()
        assert summary["message_count"] == 2
        assert summary["attachment_count"] == 1

        detail = history_client.get(f"/conversations/{conv['id']}").json()
        assert len(detail["messages"]) == 2
        user_msg, assistant_msg = detail["messages"]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Summarize this PDF in 3 bullets."
        assert assistant_msg["role"] == "assistant"
        assert "- a\\n- b\\n- c" in assistant_msg["content"]
        assert detail["attachments"][0]["filename"] == "report.pdf"

    def test_save_turn_against_missing_conversation_returns_404(
        self, history_client: TestClient
    ) -> None:
        response = history_client.post(
            "/conversations/does-not-exist/turns",
            json={"query": "hi", "status": "completed"},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"

    def test_get_missing_conversation_returns_404(self, history_client: TestClient) -> None:
        response = history_client.get("/conversations/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "CONVERSATION_NOT_FOUND"


class TestDeleteConversation:
    def test_delete_removes_conversation_and_its_messages(self, history_client: TestClient) -> None:
        conv = history_client.post("/conversations", json={"title": "Temp"}).json()
        history_client.post(
            f"/conversations/{conv['id']}/turns",
            json={"query": "hi", "status": "completed", "answer": "hello"},
        )

        delete_response = history_client.delete(f"/conversations/{conv['id']}")
        assert delete_response.status_code == 204

        assert history_client.get(f"/conversations/{conv['id']}").status_code == 404
        assert conv["id"] not in [c["id"] for c in history_client.get("/conversations").json()]

    def test_delete_missing_conversation_returns_404(self, history_client: TestClient) -> None:
        response = history_client.delete("/conversations/does-not-exist")
        assert response.status_code == 404


class TestPersistenceAcrossConnections:
    def test_history_survives_a_fresh_store_connection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates a backend restart: a fresh init_db() call against the
        same file must not lose or corrupt existing data."""
        db_path = tmp_path / "restart_test.db"
        monkeypatch.setattr(history_store, "DB_PATH", db_path)
        history_store.init_db()
        history_store.create_conversation("conv-1", "Persisted conversation")

        # Simulate the app restarting: init_db() runs again against the same file.
        history_store.init_db()

        conversations = history_store.list_conversations()
        assert len(conversations) == 1
        assert conversations[0]["title"] == "Persisted conversation"
