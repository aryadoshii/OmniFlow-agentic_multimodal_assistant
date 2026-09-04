"""Integration tests for IngestionService and POST /ingest API endpoint."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from omniflow.api.routes.ingest import get_ingestion_service
from omniflow.config import Settings
from omniflow.exceptions import UnsupportedFileError, UploadValidationError
from omniflow.models.document import SourceType
from omniflow.processors.image_processor import ImageProcessor
from omniflow.processors.pdf_processor import PDFProcessor
from omniflow.processors.text_processor import TextProcessor
from omniflow.services.ingestion import IngestionService
from omniflow.services.ocr_service import OCRService
from tests.test_processors import (
    create_sample_image_bytes,
    create_sample_pdf_bytes,
)


def get_mock_ingestion_service() -> IngestionService:
    """Provides an IngestionService with isolated OCR to keep tests deterministic and fast."""
    mock_ocr = MagicMock(spec=OCRService)
    mock_ocr.extract_text_with_confidence.return_value = (
        "Mock OCR Extracted Text",
        0.92,
        {"word_count": 4},
    )

    processors = [
        TextProcessor(),
        PDFProcessor(ocr_service=mock_ocr),
        ImageProcessor(ocr_service=mock_ocr),
    ]
    return IngestionService(processors=processors)


# ============================================================================
# IngestionService Unit Tests
# ============================================================================


def test_ingestion_service_text_and_files() -> None:
    service = get_mock_ingestion_service()
    pdf_bytes = create_sample_pdf_bytes(["Important research findings on agents."])

    docs = service.ingest_inputs(
        text="Instruction prompt from user.",
        files=[(pdf_bytes, "paper.pdf", "application/pdf")],
    )

    assert len(docs) == 2
    assert docs[0].source_type == SourceType.TEXT
    assert docs[0].content == "Instruction prompt from user."
    assert docs[1].source_type == SourceType.PDF
    assert "Important research findings" in docs[1].content


def test_ingestion_service_unsupported_file() -> None:
    service = get_mock_ingestion_service()
    with pytest.raises(UnsupportedFileError):
        service.process_file(b"MZ...", "program.exe", "application/x-msdownload")


def test_ingestion_service_empty_file() -> None:
    service = get_mock_ingestion_service()
    with pytest.raises(UploadValidationError):
        service.process_file(b"", "empty.txt", "text/plain")


def test_ingestion_service_oversized_file() -> None:
    service = get_mock_ingestion_service()
    service.settings = Settings(MAX_UPLOAD_SIZE_MB=1)
    huge_bytes = b"x" * (2 * 1024 * 1024)  # 2 MB

    with pytest.raises(UploadValidationError):
        service.process_file(huge_bytes, "large.txt", "text/plain")


# ============================================================================
# API Endpoint Integration Tests (POST /ingest)
# ============================================================================


def test_api_ingest_text_only(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_ingestion_service] = get_mock_ingestion_service

    response = client.post(
        "/ingest",
        data={"text": "Hello OmniFlow assistant!"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert data["documents"][0]["source_type"] == "text"
    assert data["documents"][0]["content"] == "Hello OmniFlow assistant!"
    assert data["documents"][0]["extraction_method"] == "direct_input"


def test_api_ingest_single_file_pdf(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_ingestion_service] = get_mock_ingestion_service

    pdf_content = create_sample_pdf_bytes(["Autonomous multimodal architecture."])

    response = client.post(
        "/ingest",
        files=[("files", ("test.pdf", pdf_content, "application/pdf"))],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 1
    assert data["documents"][0]["filename"] == "test.pdf"
    assert data["documents"][0]["source_type"] == "pdf"
    assert "Autonomous multimodal architecture." in data["documents"][0]["content"]


def test_api_ingest_multiple_files_and_text(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_ingestion_service] = get_mock_ingestion_service

    pdf_content = create_sample_pdf_bytes(["PDF report section."])
    img_content = create_sample_image_bytes("PNG")

    response = client.post(
        "/ingest",
        data={"text": "Synthesize this data."},
        files=[
            ("files", ("report.pdf", pdf_content, "application/pdf")),
            ("files", ("diagram.png", img_content, "image/png")),
        ],
    )

    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 3
    # Preserves order: text first, then files in order
    assert data["documents"][0]["source_type"] == "text"
    assert data["documents"][1]["source_type"] == "pdf"
    assert data["documents"][2]["source_type"] == "image"
    assert data["documents"][2]["content"] == "Mock OCR Extracted Text"


def test_api_ingest_empty_request_error(client: TestClient) -> None:
    response = client.post("/ingest")
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["code"] == "INVALID_INPUT"


def test_api_ingest_unsupported_file_error(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_ingestion_service] = get_mock_ingestion_service

    response = client.post(
        "/ingest",
        files=[("files", ("script.sh", b"#!/bin/bash\necho 1", "text/x-sh"))],
    )

    assert response.status_code == 415
    data = response.json()
    assert data["error"]["code"] == "UNSUPPORTED_FILE_TYPE"


def test_api_ingest_corrupt_pdf_error(client: TestClient) -> None:
    app = client.app
    app.dependency_overrides[get_ingestion_service] = get_mock_ingestion_service

    response = client.post(
        "/ingest",
        files=[("files", ("corrupted.pdf", b"garbage bytes not pdf", "application/pdf"))],
    )

    assert response.status_code == 422
    data = response.json()
    assert data["error"]["code"] == "PROCESSING_FAILURE"
