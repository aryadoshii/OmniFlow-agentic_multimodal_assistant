"""Tests for domain and Pydantic models."""

import pytest
from pydantic import ValidationError

from omniflow.models import (
    AgentState,
    ExecutionTrace,
    ExtractionMethod,
    NormalizedDocument,
    OmniFlowResponse,
    SourceType,
    ToolExecutionTrace,
    UploadedInput,
    UserRequest,
)


def test_normalized_document_defaults() -> None:
    """Verifies NormalizedDocument creates valid IDs and defaults."""
    doc = NormalizedDocument(
        filename="notes.pdf",
        source_type=SourceType.PDF,
        mime_type="application/pdf",
        content="Meeting notes text",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
    )
    assert doc.id is not None
    assert len(doc.id) == 36  # UUID4 length
    assert doc.filename == "notes.pdf"
    assert doc.source_type == SourceType.PDF
    assert doc.content == "Meeting notes text"
    assert doc.extraction_method == ExtractionMethod.NATIVE_TEXT
    assert doc.confidence is None
    assert doc.detected_urls == []
    assert doc.warnings == []
    assert doc.metadata == {}


def test_normalized_document_confidence_bounds() -> None:
    """Verifies confidence score must be within [0.0, 1.0]."""
    with pytest.raises(ValidationError):
        NormalizedDocument(
            filename="image.png",
            source_type=SourceType.IMAGE,
            mime_type="image/png",
            content="OCR output",
            extraction_method=ExtractionMethod.OCR,
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        NormalizedDocument(
            filename="image.png",
            source_type=SourceType.IMAGE,
            mime_type="image/png",
            content="OCR output",
            extraction_method=ExtractionMethod.OCR,
            confidence=-0.1,
        )


def test_user_request_validation() -> None:
    """Verifies UserRequest enforces non-empty prompt."""
    req = UserRequest(prompt="Summarize this document")
    assert req.prompt == "Summarize this document"
    assert req.session_id is None

    with pytest.raises(ValidationError):
        UserRequest(prompt="")


def test_safe_execution_trace() -> None:
    """Verifies ExecutionTrace accepts safe metadata and serializes cleanly."""
    step = ToolExecutionTrace(
        step_name="detect_youtube_urls",
        tool_name="youtube_url_detector",
        status="success",
        duration_ms=12.4,
        details={"url_count": "1"},
    )
    trace = ExecutionTrace(
        session_id="session-123",
        total_duration_ms=45.6,
        steps=[step],
    )
    data = trace.model_dump()
    assert data["session_id"] == "session-123"
    assert len(data["steps"]) == 1
    assert data["steps"][0]["step_name"] == "detect_youtube_urls"
    assert data["steps"][0]["duration_ms"] == 12.4
    assert data["steps"][0]["details"]["url_count"] == "1"


def test_response_model() -> None:
    """Verifies OmniFlowResponse properly encapsulates answer, docs, and traces."""
    doc = NormalizedDocument(
        filename="input.txt",
        source_type=SourceType.TEXT,
        mime_type="text/plain",
        content="Sample text",
        extraction_method=ExtractionMethod.DIRECT_INPUT,
    )
    resp = OmniFlowResponse(
        session_id="sess-1",
        answer="Here is the summary.",
        clarification_needed=False,
        normalized_documents=[doc],
        warnings=[],
    )
    assert resp.answer == "Here is the summary."
    assert len(resp.normalized_documents) == 1
    assert resp.clarification_needed is False


def test_agent_state_full_structure() -> None:
    """Verifies AgentState provides all containers required for future LangGraph orchestration."""
    state = AgentState(
        original_request="Analyze quarterly report",
        session_id="sess-99",
        uploaded_inputs=[
            UploadedInput(
                filename="report.pdf",
                mime_type="application/pdf",
                size_bytes=1024,
            )
        ],
        detected_intent="document_analysis",
        constraints=["focus on revenue", "bullet points"],
        plan=["extract_text", "retrieve_facts", "synthesize"],
        planned_tools=["pdf_processor", "vector_rag"],
        tool_results={"pdf_processor": "done"},
        warnings=["OCR confidence medium"],
        final_answer="Quarterly revenue increased by 15%.",
    )

    state_dict = state.model_dump()
    assert state_dict["original_request"] == "Analyze quarterly report"
    assert state_dict["uploaded_inputs"][0]["filename"] == "report.pdf"
    assert state_dict["detected_intent"] == "document_analysis"
    assert len(state_dict["constraints"]) == 2
    assert state_dict["planned_tools"] == ["pdf_processor", "vector_rag"]
    assert state_dict["final_answer"] == "Quarterly revenue increased by 15%."
