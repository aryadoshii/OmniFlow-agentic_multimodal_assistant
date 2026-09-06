"""Pytest fixtures for OmniFlow test suite."""

import pytest
from fastapi.testclient import TestClient
from backend.config import Settings, get_settings
from backend.main import create_app


@pytest.fixture
def test_settings() -> Settings:
    """Fixture providing isolated test settings.

    _env_file=None disables dotenv loading for this instance so a real
    local .env (e.g. a developer's own GEMINI_API_KEY) never leaks into
    test settings -- tests that rely on GEMINI_API_KEY being unset must
    stay hermetic regardless of what exists on disk outside the repo.
    """
    return Settings(
        _env_file=None,
        APP_ENV="test",
        LOG_LEVEL="WARNING",
        HOST="127.0.0.1",
        PORT=8000,
        MAX_UPLOAD_SIZE_MB=10,
    )


@pytest.fixture
def client(test_settings: Settings) -> TestClient:
    """Fixture providing a FastAPI TestClient with test settings injected."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    with TestClient(app) as test_client:
        yield test_client
