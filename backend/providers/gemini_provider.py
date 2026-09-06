"""Gemini LLM provider -- the sole application boundary for Gemini API communication.

Wraps the official ``google-genai`` SDK behind the BaseLLMProvider interface.
No other application component should import ``google.genai`` directly --
LangGraph nodes (Phase 4.2+) call BaseLLMProvider, never the SDK.

This module contains no planner, prompt-template, tool-selection, or RAG
logic -- its only responsibility is LLM communication: constructing the
client, sending a generation request, validating structured output, and
translating SDK/network failures into OmniFlow domain exceptions.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from backend.config import Settings, get_settings
from backend.exceptions import ConfigurationError, ExternalProviderError
from backend.providers.base import BaseLLMProvider, ResponseModelT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level optional import so the app still starts without the package
# installed, and so tests can patch this module's own attributes without any
# network access. Importing this module performs no network call either way.
# ---------------------------------------------------------------------------
try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - exercised via a dedicated test
    genai = None  # type: ignore[assignment]
    genai_errors = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]

# google-genai's own default retry policy retries up to 5 times (with
# backoff) on 429/5xx by default. That would silently delay and obscure the
# rate-limit signal Phase 4's future agent execution layer needs to react to,
# and risks burning through free-tier quota on retries this provider has no
# business making. Retries are explicitly disabled here (1 = no retries);
# retry/backoff policy belongs to that future layer, not this one.
_NO_RETRY_ATTEMPTS = 1


class GeminiProvider(BaseLLMProvider):
    """Concrete BaseLLMProvider backed by Google's Gemini API (google-genai SDK).

    Configuration (model name, API key, timeout) comes entirely from the
    centralized ``Settings`` system -- nothing is hardcoded here. Constructing
    this provider performs no network call (the underlying ``genai.Client``
    is a lightweight, side-effect-free object); the first actual API request
    happens only when ``generate()``/``generate_structured()`` is invoked.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        if genai is None:
            raise ConfigurationError(
                "google-genai is not installed. Install it with: pip install google-genai",
                details={"dependency": "google-genai"},
            )

        api_key = self._settings.gemini_api_key
        if api_key is None:
            raise ConfigurationError(
                "GEMINI_API_KEY is not configured.",
                details={"setting": "GEMINI_API_KEY"},
            )

        self._model_name = self._settings.llm_model
        timeout_ms = int(self._settings.gemini_timeout_seconds * 1000)

        self._client = genai.Client(
            api_key=api_key.get_secret_value(),
            http_options=genai_types.HttpOptions(
                timeout=timeout_ms,
                retry_options=genai_types.HttpRetryOptions(attempts=_NO_RETRY_ATTEMPTS),
            ),
        )

    # ------------------------------------------------------------------
    # Public API (BaseLLMProvider contract)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        """Generates a plain-text completion.

        Raises:
            ConfigurationError: If the configured model name is invalid, or
                                 Gemini rejects the configured credentials.
            ExternalProviderError: If the request fails due to rate limiting,
                                    service unavailability, timeout, network
                                    failure, or an unexpected provider error.
        """
        response = self._request(prompt, system_instruction=system_instruction)
        return response.text or ""

    def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModelT],
        system_instruction: str | None = None,
    ) -> ResponseModelT:
        """Generates a completion and validates it into ``response_model``.

        The raw response text is explicitly re-validated via
        ``response_model.model_validate_json()`` before being returned --
        Gemini's own schema-constrained JSON output is a strong hint, not a
        substitute for this application's own validation.

        Raises:
            ConfigurationError: If the configured model name is invalid, or
                                 Gemini rejects the configured credentials.
            ExternalProviderError: If the request fails due to rate limiting,
                                    service unavailability, timeout, network
                                    failure, malformed/unvalidatable structured
                                    output, or an unexpected provider error.
        """
        response = self._request(
            prompt,
            system_instruction=system_instruction,
            response_schema=response_model,
        )
        raw_text = response.text
        if not raw_text:
            raise ExternalProviderError(
                "Gemini returned an empty response for a structured request.",
                details={"reason": "invalid_structured_output", "model": self._model_name},
            )

        try:
            return response_model.model_validate_json(raw_text)
        except ValidationError as exc:
            logger.error(
                "Gemini structured output failed validation against %s.",
                response_model.__name__,
            )
            raise ExternalProviderError(
                f"Gemini's response could not be validated into {response_model.__name__}.",
                details={
                    "reason": "invalid_structured_output",
                    "model": self._model_name,
                    "response_model": response_model.__name__,
                },
            ) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        prompt: str,
        system_instruction: str | None = None,
        response_schema: type[BaseModel] | None = None,
    ) -> Any:
        """Sends one generation request and translates any failure.

        Never logs the prompt, system instruction, or model output -- only
        safe operational metadata (model name, output type, duration, and
        success/failure).
        """
        config_kwargs: dict[str, Any] = {}
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        config = genai_types.GenerateContentConfig(**config_kwargs) if config_kwargs else None
        output_type = "structured" if response_schema is not None else "text"

        start = time.perf_counter()
        logger.info(
            "Gemini request starting (model=%s, output_type=%s).",
            self._model_name,
            output_type,
        )
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=config,
            )
        except genai_errors.APIError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._raise_for_api_error(exc, duration_ms)
        except httpx.TimeoutException as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Gemini request timed out after %.1fms (model=%s).",
                duration_ms,
                self._model_name,
            )
            raise ExternalProviderError(
                "Gemini request timed out.",
                details={"reason": "timeout", "model": self._model_name},
            ) from exc
        except httpx.HTTPError as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Gemini request failed due to a network error after %.1fms (model=%s): %s",
                duration_ms,
                self._model_name,
                type(exc).__name__,
            )
            raise ExternalProviderError(
                "Gemini request failed due to a network error.",
                details={"reason": "network_error", "model": self._model_name},
            ) from exc
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Gemini request failed unexpectedly after %.1fms (model=%s): %s",
                duration_ms,
                self._model_name,
                type(exc).__name__,
            )
            raise ExternalProviderError(
                f"Unexpected Gemini provider failure: {type(exc).__name__}.",
                details={
                    "reason": "unexpected_error",
                    "error_type": type(exc).__name__,
                    "model": self._model_name,
                },
            ) from exc

        duration_ms = (time.perf_counter() - start) * 1000.0
        logger.info(
            "Gemini request succeeded (model=%s, output_type=%s, duration_ms=%.1f).",
            self._model_name,
            output_type,
            duration_ms,
        )
        return response

    def _raise_for_api_error(self, exc: Exception, duration_ms: float) -> None:
        """Maps a google-genai APIError to the appropriate OmniFlow exception.

        Never reraises the SDK exception type directly, and never includes
        the prompt or response body in the raised exception's details.
        """
        code = getattr(exc, "code", None)
        status = getattr(exc, "status", None)
        logger.error(
            "Gemini API error after %.1fms (model=%s): code=%s status=%s",
            duration_ms,
            self._model_name,
            code,
            status,
        )

        if code in (401, 403):
            raise ConfigurationError(
                "Gemini rejected the configured credentials.",
                details={"reason": "authentication_failed", "gemini_status_code": str(code)},
            ) from exc
        if code == 404:
            raise ConfigurationError(
                f"Gemini model '{self._model_name}' was not found or is invalid.",
                details={
                    "reason": "invalid_model",
                    "model": self._model_name,
                    "gemini_status_code": str(code),
                },
            ) from exc
        if code == 429:
            raise ExternalProviderError(
                "Gemini rate limit or quota exceeded.",
                details={"reason": "rate_limited", "gemini_status_code": str(code)},
            ) from exc
        if isinstance(code, int) and code >= 500:
            raise ExternalProviderError(
                "Gemini service is currently unavailable.",
                details={"reason": "service_unavailable", "gemini_status_code": str(code)},
            ) from exc

        raise ExternalProviderError(
            f"Gemini API error: {status or type(exc).__name__}.",
            details={"reason": "api_error", "gemini_status_code": str(code)},
        ) from exc
