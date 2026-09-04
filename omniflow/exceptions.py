"""Application exception hierarchy for OmniFlow."""

from typing import Any


class OmniFlowException(Exception):
    """Base exception for all OmniFlow application errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}


class InvalidInputError(OmniFlowException):
    """Raised when user input or payload parameters are malformed or invalid."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="INVALID_INPUT",
            status_code=400,
            details=details,
        )


class UnsupportedFileError(OmniFlowException):
    """Raised when an uploaded file type or MIME type is not supported."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_FILE_TYPE",
            status_code=415,
            details=details,
        )


class ProcessingFailureError(OmniFlowException):
    """Raised when parsing, extraction, or processing of an input fails."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="PROCESSING_FAILURE",
            status_code=422,
            details=details,
        )


class ExternalProviderError(OmniFlowException):
    """Raised when an external API (such as Gemini) fails, times out, or errors."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="EXTERNAL_PROVIDER_ERROR",
            status_code=502,
            details=details,
        )


class OrchestrationError(OmniFlowException):
    """Raised when agent execution, planning, or graph state transitions fail."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="ORCHESTRATION_ERROR",
            status_code=500,
            details=details,
        )


class ConfigurationError(OmniFlowException):
    """Raised when critical configuration or credentials are missing or invalid."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )
