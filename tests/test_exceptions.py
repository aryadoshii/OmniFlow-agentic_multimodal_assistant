"""Tests for exception hierarchy and API error handling."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.error_handlers import register_error_handlers
from backend.exceptions import (
    ConfigurationError,
    ExternalProviderError,
    InvalidInputError,
    OmniFlowException,
    OrchestrationError,
    ProcessingFailureError,
    UnsupportedFileError,
)


def test_exception_hierarchy_attributes() -> None:
    """Verifies that all domain exceptions define appropriate status codes and error codes."""
    test_cases = [
        (InvalidInputError("bad input"), 400, "INVALID_INPUT"),
        (UnsupportedFileError("bad ext"), 415, "UNSUPPORTED_FILE_TYPE"),
        (ProcessingFailureError("corrupt"), 422, "PROCESSING_FAILURE"),
        (ExternalProviderError("timeout"), 502, "EXTERNAL_PROVIDER_ERROR"),
        (OrchestrationError("bad state"), 500, "ORCHESTRATION_ERROR"),
        (ConfigurationError("missing key"), 500, "CONFIGURATION_ERROR"),
    ]

    for exc, expected_status, expected_code in test_cases:
        assert isinstance(exc, OmniFlowException)
        assert exc.status_code == expected_status
        assert exc.error_code == expected_code
        assert exc.message is not None


def test_api_error_handler_domain_exception() -> None:
    """Verifies that throwing an OmniFlowException produces structured JSON without stack trace."""
    test_app = FastAPI()
    register_error_handlers(test_app)

    @test_app.get("/trigger-invalid-input")
    def trigger_invalid() -> None:
        raise InvalidInputError("Prompt cannot be empty.", details={"field": "prompt"})

    @test_app.get("/trigger-unsupported-file")
    def trigger_unsupported() -> None:
        raise UnsupportedFileError("MIME type image/tiff is not supported.")

    client = TestClient(test_app, raise_server_exceptions=False)

    # Test 400 InvalidInputError
    resp_400 = client.get("/trigger-invalid-input")
    assert resp_400.status_code == 400
    data_400 = resp_400.json()
    assert "error" in data_400
    assert data_400["error"]["code"] == "INVALID_INPUT"
    assert data_400["error"]["message"] == "Prompt cannot be empty."
    assert data_400["error"]["details"] == {"field": "prompt"}

    # Test 415 UnsupportedFileError
    resp_415 = client.get("/trigger-unsupported-file")
    assert resp_415.status_code == 415
    data_415 = resp_415.json()
    assert "error" in data_415
    assert data_415["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_api_error_handler_unhandled_exception() -> None:
    """Verifies that unexpected server exceptions return a sanitized 500 without stack trace."""
    test_app = FastAPI()
    register_error_handlers(test_app)

    @test_app.get("/trigger-crash")
    def trigger_crash() -> None:
        raise RuntimeError("Secret DB connection string or internal trace")

    client = TestClient(test_app, raise_server_exceptions=False)
    resp = client.get("/trigger-crash")

    assert resp.status_code == 500
    data = resp.json()
    assert "error" in data
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    # Ensure internal exception message is NOT leaked to client
    assert "Secret DB connection string" not in data["error"]["message"]
    assert "An unexpected error occurred" in data["error"]["message"]
