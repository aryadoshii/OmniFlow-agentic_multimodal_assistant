"""Tests for centralized configuration management."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from omniflow import config
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


def test_env_file_is_anchored_to_project_root_not_process_cwd() -> None:
    """A relative env_file (e.g. ".env") resolves against the process's
    current working directory, so `uvicorn omniflow.main:app --reload`
    silently finds no .env (pydantic-settings does not error on a missing
    file) whenever launched from anywhere other than the exact repo root
    -- surfacing as a false "GEMINI_API_KEY is not configured" even when a
    real .env with a real key exists at the project root. The configured
    env_file must instead be an absolute path anchored to the project
    root, independent of cwd.
    """
    project_root = Path(config.__file__).resolve().parent.parent
    assert Settings.model_config["env_file"] == project_root / ".env"
    assert Path(Settings.model_config["env_file"]).is_absolute()


def test_settings_loads_regardless_of_process_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Simulates uvicorn being launched from a directory other than the
    repo root. A real OS environment variable (as Docker/Render always
    supply, and as a correctly-anchored .env load would also produce)
    must still be picked up -- this would have appeared to work even with
    the old relative-path bug, but guards against a regression that
    breaks environment-variable loading entirely while "fixing" the path.
    """
    other_cwd = tmp_path / "somewhere" / "else"
    other_cwd.mkdir(parents=True)
    monkeypatch.chdir(other_cwd)
    monkeypatch.setenv("GEMINI_API_KEY", "test-non-secret-placeholder-key")

    settings = Settings()
    assert settings.gemini_api_key is not None
    assert settings.gemini_api_key.get_secret_value() == "test-non-secret-placeholder-key"
