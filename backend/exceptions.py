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


class OCRProcessingError(ProcessingFailureError):
    """Raised when OCR extraction fails due to binary issues or corrupt images."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "OCR_PROCESSING_ERROR"


class TranscriptionError(ProcessingFailureError):
    """Raised when speech-to-text audio transcription fails or backend is unavailable."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "TRANSCRIPTION_ERROR"


class EmbeddingGenerationError(ProcessingFailureError):
    """Raised when the local embedding backend is unavailable or encoding fails."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "EMBEDDING_GENERATION_ERROR"


class TranscriptUnavailableError(ProcessingFailureError):
    """Raised when a video/source is valid but no transcript could be retrieved for it.

    Distinct from ExternalProviderError: this means the provider was reachable
    and answered, but the requested content has no transcript (disabled,
    missing, unplayable, age-restricted, or the requested translation
    language is unavailable) -- not a network/provider outage.
    """

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "TRANSCRIPT_UNAVAILABLE"


class UploadValidationError(InvalidInputError):
    """Raised when an uploaded file fails validation checks (size, extension, empty)."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "UPLOAD_VALIDATION_ERROR"


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


class ToolExecutionError(OmniFlowException):
    """Raised when a deterministic tool fails to execute or returns an invalid result."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="TOOL_EXECUTION_ERROR",
            status_code=500,
            details=details,
        )


class ToolNotFoundError(OmniFlowException):
    """Raised when a requested tool name is not present in the tool registry."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="TOOL_NOT_FOUND",
            status_code=404,
            details=details,
        )


class ToolAlreadyRegisteredError(OmniFlowException):
    """Raised when attempting to register a tool name that is already registered."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="TOOL_ALREADY_REGISTERED",
            status_code=500,
            details=details,
        )


class ConversationNotFoundError(OmniFlowException):
    """Raised when a requested conversation id does not exist in history."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message=message,
            error_code="CONVERSATION_NOT_FOUND",
            status_code=404,
            details=details,
        )


class RAGRetrievalError(ProcessingFailureError):
    """Raised when vector store indexing or semantic retrieval fails."""

    def __init__(
        self, message: str, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message=message, details=details)
        self.error_code = "RAG_RETRIEVAL_ERROR"
