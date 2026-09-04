"""Tests for health check and root endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    """Verifies that GET /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "OmniFlow"
    assert "version" in data
    assert data["environment"] == "test"


def test_root_endpoint(client: TestClient) -> None:
    """Verifies that GET / returns 200 OK with API discovery metadata."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert "OmniFlow" in data["message"]
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"
