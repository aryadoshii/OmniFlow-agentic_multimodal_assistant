"""Tests for centralized configuration management."""

import pytest
from pydantic import ValidationError
from omniflow.config import Settings


def test_default_settings() -> None:
    """Verifies that default settings load with expected baseline values."""
    settings = Settings()
    assert settings.app_name == "OmniFlow"
    assert settings.app_env in ["development", "staging", "production", "test"]
    assert settings.max_upload_size_mb == 25
    assert settings.llm_model == "gemini-2.5-flash"
    assert settings.port == 8000


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that environment variables correctly override default settings."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MAX_UPLOAD_SIZE_MB", "50")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-pro")
    monkeypatch.setenv("PORT", "9000")

    settings = Settings()
    assert settings.app_env == "production"
    assert settings.is_production is True
    assert settings.is_development is False
    assert settings.log_level == "DEBUG"
    assert settings.max_upload_size_mb == 50
    assert settings.llm_model == "gemini-2.5-pro"
    assert settings.port == 9000


def test_upload_size_validation() -> None:
    """Verifies that upload size outside allowed bounds raises ValidationError."""
    with pytest.raises(ValidationError):
        Settings(MAX_UPLOAD_SIZE_MB=0)

    with pytest.raises(ValidationError):
        Settings(MAX_UPLOAD_SIZE_MB=101)


def test_secret_key_masking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the Gemini API key is stored as SecretStr and not leaked in string representations."""
    test_key = "AIzaSyFakeTestKeyForValidation12345"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)

    settings = Settings()
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == test_key
    assert test_key not in str(settings.gemini_api_key)
    assert test_key not in repr(settings.gemini_api_key)
