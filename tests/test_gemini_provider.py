"""Tests for GeminiProvider (Phase 4.1).

All tests mock the google-genai SDK boundary (Client, generate_content, and
the SDK's own error classes) -- no network access or real API key is ever
used. Real (lightweight, importable without network) google.genai.errors
classes are used to build realistic exception fixtures, since constructing
them performs no I/O.
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from omniflow.config import Settings
from omniflow.exceptions import ConfigurationError, ExternalProviderError
from omniflow.providers.gemini_provider import GeminiProvider


def _settings(api_key: str | None = "fake-test-key", model: str = "gemini-2.5-flash") -> Settings:
    # _env_file=None: a real local .env (e.g. a developer's own
    # GEMINI_API_KEY) must never leak into this "missing key" fixture and
    # falsely satisfy the configuration check.
    kwargs = {"LLM_MODEL": model, "_env_file": None}
    if api_key is not None:
        kwargs["GEMINI_API_KEY"] = api_key
    return Settings(**kwargs)


def _mock_client_returning(response: MagicMock) -> MagicMock:
    """Builds a mock replacing genai.Client() so .models.generate_content() returns a fixed value."""
    instance = MagicMock()
    instance.models.generate_content.return_value = response
    return MagicMock(return_value=instance)


def _mock_client_raising(exc: BaseException) -> MagicMock:
    """Builds a mock replacing genai.Client() so .models.generate_content() raises."""
    instance = MagicMock()
    instance.models.generate_content.side_effect = exc
    return MagicMock(return_value=instance)


class _SampleIntent(BaseModel):
    intent: str
    confidence: float


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_api_key_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError) as exc_info:
            GeminiProvider(settings=_settings(api_key=None))
        assert exc_info.value.error_code == "CONFIGURATION_ERROR"
        assert exc_info.value.details["setting"] == "GEMINI_API_KEY"

    def test_construction_does_not_call_generate_content(self) -> None:
        """Constructing the provider must call genai.Client() (confirmed
        lightweight/side-effect-free) but never generate_content()."""
        mock_cls = _mock_client_returning(MagicMock(text="unused"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            GeminiProvider(settings=_settings())
        mock_cls.return_value.models.generate_content.assert_not_called()

    def test_missing_sdk_raises_configuration_error(self) -> None:
        with patch("omniflow.providers.gemini_provider.genai", None):
            with pytest.raises(ConfigurationError, match="not installed"):
                GeminiProvider(settings=_settings())

    def test_module_import_makes_no_client_calls(self) -> None:
        """Importing the module must never construct a client or call the SDK."""
        mock_cls = MagicMock()
        with patch.dict(sys.modules):
            sys.modules.pop("omniflow.providers.gemini_provider", None)
            with patch("google.genai.Client", mock_cls):
                importlib.import_module("omniflow.providers.gemini_provider")
        mock_cls.assert_not_called()


# ---------------------------------------------------------------------------
# Configuration-driven behavior
# ---------------------------------------------------------------------------


class TestConfigurationDriven:
    def test_configured_model_name_used_in_request(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="hi"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings(model="gemini-1.5-pro"))
            provider.generate("hello")

        call_kwargs = mock_cls.return_value.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-1.5-pro"

    def test_api_key_passed_to_client_not_hardcoded(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="hi"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            GeminiProvider(settings=_settings(api_key="my-secret-key-value"))

        client_kwargs = mock_cls.call_args[1]
        assert client_kwargs["api_key"] == "my-secret-key-value"

    def test_timeout_setting_passed_to_http_options(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="hi"))
        settings = Settings(GEMINI_API_KEY="fake", LLM_MODEL="gemini-2.5-flash", GEMINI_TIMEOUT_SECONDS=5)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            GeminiProvider(settings=settings)

        http_options = mock_cls.call_args[1]["http_options"]
        assert http_options.timeout == 5000  # seconds -> milliseconds

    def test_sdk_retries_disabled(self) -> None:
        """Confirms the provider disables the SDK's own default retry
        behavior, so a 429 surfaces immediately rather than after ~5 silent
        retries -- required for predictable free-tier-safe behavior."""
        mock_cls = _mock_client_returning(MagicMock(text="hi"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            GeminiProvider(settings=_settings())

        http_options = mock_cls.call_args[1]["http_options"]
        assert http_options.retry_options.attempts == 1


# ---------------------------------------------------------------------------
# Successful generation
# ---------------------------------------------------------------------------


class TestSuccessfulGeneration:
    def test_plain_text_generation(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="Paris is the capital of France."))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            result = provider.generate("What is the capital of France?")

        assert result == "Paris is the capital of France."

    def test_plain_text_generation_passes_system_instruction(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="ok"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            provider.generate("hello", system_instruction="Be concise.")

        call_kwargs = mock_cls.return_value.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "Be concise."

    def test_structured_generation_returns_validated_model(self) -> None:
        mock_cls = _mock_client_returning(
            MagicMock(text='{"intent": "search", "confidence": 0.87}')
        )
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            result = provider.generate_structured("classify this", _SampleIntent)

        assert isinstance(result, _SampleIntent)
        assert result.intent == "search"
        assert result.confidence == 0.87

    def test_structured_generation_sets_response_schema_config(self) -> None:
        mock_cls = _mock_client_returning(
            MagicMock(text='{"intent": "search", "confidence": 0.5}')
        )
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            provider.generate_structured("classify this", _SampleIntent)

        call_kwargs = mock_cls.return_value.models.generate_content.call_args[1]
        assert call_kwargs["config"].response_schema is _SampleIntent
        assert call_kwargs["config"].response_mime_type == "application/json"


# ---------------------------------------------------------------------------
# Malformed structured output
# ---------------------------------------------------------------------------


class TestMalformedStructuredOutput:
    def test_invalid_json_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text="not valid json at all"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate_structured("classify this", _SampleIntent)

        assert exc_info.value.details["reason"] == "invalid_structured_output"

    def test_json_missing_required_fields_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text='{"unrelated_field": 1}'))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate_structured("classify this", _SampleIntent)

        assert exc_info.value.details["reason"] == "invalid_structured_output"

    def test_empty_response_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_returning(MagicMock(text=""))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate_structured("classify this", _SampleIntent)

        assert exc_info.value.details["reason"] == "invalid_structured_output"


# ---------------------------------------------------------------------------
# Provider/network failure mapping
# ---------------------------------------------------------------------------


class TestFailureMapping:
    def test_authentication_error_401_raises_configuration_error(self) -> None:
        from google.genai import errors

        exc = errors.ClientError(
            code=401, response_json={"error": {"message": "bad key", "status": "UNAUTHENTICATED"}}
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ConfigurationError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "authentication_failed"

    def test_permission_denied_403_raises_configuration_error(self) -> None:
        from google.genai import errors

        exc = errors.ClientError(
            code=403, response_json={"error": {"message": "denied", "status": "PERMISSION_DENIED"}}
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ConfigurationError):
                provider.generate("hello")

    def test_invalid_model_404_raises_configuration_error(self) -> None:
        from google.genai import errors

        exc = errors.ClientError(
            code=404, response_json={"error": {"message": "not found", "status": "NOT_FOUND"}}
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings(model="not-a-real-model"))
            with pytest.raises(ConfigurationError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "invalid_model"

    def test_rate_limit_429_raises_external_provider_error(self) -> None:
        from google.genai import errors

        exc = errors.ClientError(
            code=429,
            response_json={"error": {"message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}},
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.error_code == "EXTERNAL_PROVIDER_ERROR"
        assert exc_info.value.details["reason"] == "rate_limited"
        assert exc_info.value.status_code == 502

    def test_service_unavailable_503_raises_external_provider_error(self) -> None:
        from google.genai import errors

        exc = errors.ServerError(
            code=503, response_json={"error": {"message": "unavailable", "status": "UNAVAILABLE"}}
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "service_unavailable"

    def test_timeout_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_raising(httpx.TimeoutException("request timed out"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "timeout"

    def test_network_error_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_raising(httpx.ConnectError("connection refused"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "network_error"

    def test_unexpected_exception_raises_external_provider_error(self) -> None:
        mock_cls = _mock_client_raising(RuntimeError("something broke"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
            with pytest.raises(ExternalProviderError) as exc_info:
                provider.generate("hello")
        assert exc_info.value.details["reason"] == "unexpected_error"
        assert exc_info.value.details["error_type"] == "RuntimeError"

    def test_error_details_never_contain_api_key(self) -> None:
        from google.genai import errors

        exc = errors.ClientError(
            code=401, response_json={"error": {"message": "bad key", "status": "UNAUTHENTICATED"}}
        )
        mock_cls = _mock_client_raising(exc)
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings(api_key="super-secret-value-xyz"))
            with pytest.raises(ConfigurationError) as exc_info:
                provider.generate("hello")
        assert "super-secret-value-xyz" not in str(exc_info.value.details)
        assert "super-secret-value-xyz" not in exc_info.value.message


# ---------------------------------------------------------------------------
# BaseLLMProvider contract
# ---------------------------------------------------------------------------


class TestBaseLLMProviderContract:
    def test_gemini_provider_is_a_base_llm_provider(self) -> None:
        from omniflow.providers.base import BaseLLMProvider

        mock_cls = _mock_client_returning(MagicMock(text="hi"))
        with patch("omniflow.providers.gemini_provider.genai.Client", mock_cls):
            provider = GeminiProvider(settings=_settings())
        assert isinstance(provider, BaseLLMProvider)

    def test_no_langgraph_or_orchestration_imports(self) -> None:
        """The module may discuss LangGraph in prose (docstrings describing
        the future architecture) but must never actually import it or any
        orchestration/tool/RAG component."""
        import omniflow.providers.gemini_provider as provider_module

        with open(provider_module.__file__, encoding="utf-8") as f:
            lines = f.readlines()
        import_lines = [
            line.lower() for line in lines
            if line.strip().startswith("import ") or line.strip().startswith("from ")
        ]
        forbidden_modules = [
            "langgraph",
            "omniflow.models.state",
            "omniflow.tools",
            "omniflow.rag",
            "omniflow.graph",
        ]
        for line in import_lines:
            for forbidden in forbidden_modules:
                assert forbidden not in line, f"Forbidden import '{forbidden}' found: {line!r}"
