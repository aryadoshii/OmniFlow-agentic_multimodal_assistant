"""Tests for backend.agents.cross_source.analyze_cross_source().

The LLM provider is fully mocked -- no real Gemini call is ever made. These
tests verify the deterministic gating (< 2 documents -> None), that the
right structured model is requested, and that provider errors propagate
unmodified (same failure-propagation contract as intent.py/synthesizer.py).
"""

from unittest.mock import MagicMock

import pytest

from backend.agents.cross_source import (
    CrossSourceAnalysis,
    SourceUnderstanding,
    analyze_cross_source,
)
from backend.exceptions import ExternalProviderError
from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType
from backend.models.state import AgentState


def _doc(doc_id: str, filename: str, content: str) -> NormalizedDocument:
    return NormalizedDocument(
        id=doc_id,
        filename=filename,
        source_type=SourceType.TEXT,
        mime_type="text/plain",
        content=content,
        extraction_method=ExtractionMethod.DIRECT_INPUT,
    )


def _analysis(relationship: str = "partial_overlap") -> CrossSourceAnalysis:
    return CrossSourceAnalysis(
        relationship=relationship,
        sources=[
            SourceUnderstanding(document_id="d1", filename="a.txt", topic="Clinical validation", evidence="Discusses validation rules."),
            SourceUnderstanding(document_id="d2", filename="b.txt", topic="Clinical workflows", evidence="Discusses operational workflow."),
        ],
        shared_concepts=["Clinical research"],
        differences=["a.txt focuses on validation; b.txt focuses on workflow."],
        explanation="Related subject matter but not the same specific topic.",
    )


class TestGating:
    def test_fewer_than_two_documents_returns_none_without_calling_provider(self) -> None:
        provider = MagicMock()
        state = AgentState(normalized_documents=[_doc("d1", "a.txt", "some content")])

        result = analyze_cross_source(state, provider)

        assert result is None
        provider.generate_structured.assert_not_called()

    def test_zero_documents_returns_none(self) -> None:
        provider = MagicMock()
        state = AgentState(normalized_documents=[])

        assert analyze_cross_source(state, provider) is None
        provider.generate_structured.assert_not_called()


class TestStructuredCall:
    def test_requests_cross_source_analysis_model(self) -> None:
        provider = MagicMock()
        provider.generate_structured.return_value = _analysis()
        state = AgentState(
            original_request="Do these two files discuss the same topic?",
            normalized_documents=[_doc("d1", "a.txt", "validation content"), _doc("d2", "b.txt", "workflow content")],
        )

        result = analyze_cross_source(state, provider)

        assert result is not None
        call_args = provider.generate_structured.call_args
        assert call_args[0][1] is CrossSourceAnalysis
        assert result.relationship == "partial_overlap"
        assert len(result.sources) == 2

    def test_both_document_contents_appear_in_the_prompt(self) -> None:
        provider = MagicMock()
        provider.generate_structured.return_value = _analysis()
        state = AgentState(
            normalized_documents=[
                _doc("d1", "a.txt", "UNIQUE_MARKER_ALPHA"),
                _doc("d2", "b.txt", "UNIQUE_MARKER_BETA"),
            ],
        )

        analyze_cross_source(state, provider)

        prompt = provider.generate_structured.call_args[0][0]
        assert "UNIQUE_MARKER_ALPHA" in prompt
        assert "UNIQUE_MARKER_BETA" in prompt

    def test_provider_failure_propagates_unmodified(self) -> None:
        provider = MagicMock()
        provider.generate_structured.side_effect = ExternalProviderError(
            "Gemini rate limit or quota exceeded.", details={"reason": "rate_limited"}
        )
        state = AgentState(
            normalized_documents=[_doc("d1", "a.txt", "x"), _doc("d2", "b.txt", "y")],
        )

        with pytest.raises(ExternalProviderError) as exc_info:
            analyze_cross_source(state, provider)

        assert exc_info.value.details["reason"] == "rate_limited"


class TestThreeOrMoreSources:
    def test_three_documents_all_appear_in_prompt(self) -> None:
        provider = MagicMock()
        provider.generate_structured.return_value = _analysis(relationship="strong_overlap")
        state = AgentState(
            normalized_documents=[
                _doc("d1", "a.txt", "AAA_MARK"),
                _doc("d2", "b.txt", "BBB_MARK"),
                _doc("d3", "c.txt", "CCC_MARK"),
            ],
        )

        result = analyze_cross_source(state, provider)

        prompt = provider.generate_structured.call_args[0][0]
        assert "AAA_MARK" in prompt and "BBB_MARK" in prompt and "CCC_MARK" in prompt
        assert result.relationship == "strong_overlap"
