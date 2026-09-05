"""Tests for semantic intent/constraint/reference understanding (Phase 4.3).

The LLM provider is fully mocked -- no real Gemini call is ever made. Tests
verify: (1) understand_intent() correctly plumbs prompts/schemas to
BaseLLMProvider.generate_structured(), (2) it returns whatever validated
IntentResult the provider produces, and (3) it propagates provider failures
unmodified rather than swallowing or re-wrapping them.
"""

from unittest.mock import MagicMock

import pytest

from omniflow.agents.intent import IntentResult, IntentType, understand_intent
from omniflow.exceptions import ExternalProviderError
from omniflow.models.document import ExtractionMethod, NormalizedDocument, SourceType
from omniflow.models.state import AgentState


def _mock_provider_returning(result: IntentResult) -> MagicMock:
    provider = MagicMock()
    provider.generate_structured.return_value = result
    return provider


def _mock_provider_raising(exc: BaseException) -> MagicMock:
    provider = MagicMock()
    provider.generate_structured.side_effect = exc
    return provider


def _doc(filename: str, source_type: SourceType, content: str = "Some content.") -> NormalizedDocument:
    extraction = {
        SourceType.TEXT: ExtractionMethod.DIRECT_INPUT,
        SourceType.PDF: ExtractionMethod.NATIVE_TEXT,
        SourceType.IMAGE: ExtractionMethod.OCR,
        SourceType.AUDIO: ExtractionMethod.SPEECH_TO_TEXT,
    }[source_type]
    return NormalizedDocument(
        filename=filename,
        source_type=source_type,
        mime_type="application/octet-stream",
        content=content,
        extraction_method=extraction,
    )


def _intent_result(**overrides) -> IntentResult:
    defaults = dict(
        intent=IntentType.GENERAL_CONVERSATION,
        constraints=[],
        referenced_inputs=[],
        relevant_references=[],
        is_ambiguous=False,
        needs_clarification=False,
        clarification_question=None,
        explanation="A concise explanation.",
    )
    defaults.update(overrides)
    return IntentResult(**defaults)


# ---------------------------------------------------------------------------
# Intent categories (each required by the assignment)
# ---------------------------------------------------------------------------


class TestIntentCategories:
    @pytest.mark.parametrize(
        "intent_type",
        [
            IntentType.GENERAL_CONVERSATION,
            IntentType.QUESTION_ANSWERING,
            IntentType.SUMMARIZATION,
            IntentType.EXTRACTION,
            IntentType.SENTIMENT_ANALYSIS,
            IntentType.CODE_EXPLANATION,
            IntentType.TRANSCRIPTION_SUMMARY,
            IntentType.COMPARISON,
        ],
    )
    def test_each_required_intent_is_returned_faithfully(self, intent_type: IntentType) -> None:
        provider = _mock_provider_returning(_intent_result(intent=intent_type))
        result = understand_intent(AgentState(original_request="anything"), provider)
        assert result.intent == intent_type

    def test_summarization_intent_example(self) -> None:
        provider = _mock_provider_returning(
            _intent_result(intent=IntentType.SUMMARIZATION, explanation="Summarize the PDF.")
        )
        state = AgentState(original_request="Summarize this PDF.")
        result = understand_intent(state, provider)
        assert result.intent == IntentType.SUMMARIZATION

    def test_qa_intent_example(self) -> None:
        provider = _mock_provider_returning(_intent_result(intent=IntentType.QUESTION_ANSWERING))
        result = understand_intent(AgentState(original_request="What was Q3 revenue?"), provider)
        assert result.intent == IntentType.QUESTION_ANSWERING

    def test_comparison_intent_example(self) -> None:
        provider = _mock_provider_returning(_intent_result(intent=IntentType.COMPARISON))
        state = AgentState(original_request="Compare the PDF with the audio.")
        result = understand_intent(state, provider)
        assert result.intent == IntentType.COMPARISON

    def test_sentiment_intent_example(self) -> None:
        provider = _mock_provider_returning(_intent_result(intent=IntentType.SENTIMENT_ANALYSIS))
        result = understand_intent(
            AgentState(original_request="What's the sentiment of this review?"), provider
        )
        assert result.intent == IntentType.SENTIMENT_ANALYSIS

    def test_code_explanation_intent_example(self) -> None:
        provider = _mock_provider_returning(_intent_result(intent=IntentType.CODE_EXPLANATION))
        result = understand_intent(AgentState(original_request="Explain this function."), provider)
        assert result.intent == IntentType.CODE_EXPLANATION

    def test_transcription_summary_intent_example(self) -> None:
        provider = _mock_provider_returning(_intent_result(intent=IntentType.TRANSCRIPTION_SUMMARY))
        result = understand_intent(
            AgentState(original_request="Summarize the YouTube video in this PDF."), provider
        )
        assert result.intent == IntentType.TRANSCRIPTION_SUMMARY


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestConstraints:
    def test_explicit_constraints_preserved_verbatim(self) -> None:
        raw_constraints = ["exactly 3 bullets", "one-line summary", "for a beginner audience"]
        provider = _mock_provider_returning(_intent_result(constraints=raw_constraints))
        result = understand_intent(AgentState(original_request="x"), provider)
        assert result.constraints == raw_constraints

    def test_no_constraints_returns_empty_list(self) -> None:
        provider = _mock_provider_returning(_intent_result(constraints=[]))
        result = understand_intent(AgentState(original_request="x"), provider)
        assert result.constraints == []


# ---------------------------------------------------------------------------
# Reference resolution integration (deterministic hints reach the prompt)
# ---------------------------------------------------------------------------


class TestReferenceResolutionIntegration:
    def test_deterministic_hints_included_in_prompt(self) -> None:
        doc = _doc("report.pdf", SourceType.PDF)
        provider = _mock_provider_returning(_intent_result(referenced_inputs=[doc.id]))
        state = AgentState(original_request="Summarize this PDF.", normalized_documents=[doc])

        understand_intent(state, provider)

        prompt_arg = provider.generate_structured.call_args[0][0]
        assert doc.id in prompt_arg
        assert '"pdf"' in prompt_arg

    def test_document_catalog_included_in_prompt(self) -> None:
        doc = _doc("call.mp3", SourceType.AUDIO)
        provider = _mock_provider_returning(_intent_result())
        state = AgentState(original_request="Summarize the audio.", normalized_documents=[doc])

        understand_intent(state, provider)

        prompt_arg = provider.generate_structured.call_args[0][0]
        assert "call.mp3" in prompt_arg
        assert doc.id in prompt_arg

    def test_referenced_inputs_returned_from_provider_preserved(self) -> None:
        doc = _doc("report.pdf", SourceType.PDF)
        provider = _mock_provider_returning(_intent_result(referenced_inputs=[doc.id]))
        state = AgentState(original_request="Summarize this PDF.", normalized_documents=[doc])
        result = understand_intent(state, provider)
        assert result.referenced_inputs == [doc.id]

    def test_relevant_references_preserved(self) -> None:
        provider = _mock_provider_returning(
            _intent_result(relevant_references=["https://youtube.com/watch?v=abc12345678"])
        )
        result = understand_intent(AgentState(original_request="x"), provider)
        assert result.relevant_references == ["https://youtube.com/watch?v=abc12345678"]

    def test_uses_generate_structured_not_generate(self) -> None:
        """Structured output must be requested -- never plain text generation."""
        provider = _mock_provider_returning(_intent_result())
        understand_intent(AgentState(original_request="x"), provider)
        provider.generate_structured.assert_called_once()
        provider.generate.assert_not_called()

    def test_response_model_argument_is_intent_result(self) -> None:
        provider = _mock_provider_returning(_intent_result())
        understand_intent(AgentState(original_request="x"), provider)
        call_args = provider.generate_structured.call_args
        assert call_args[0][1] is IntentResult


# ---------------------------------------------------------------------------
# Ambiguity
# ---------------------------------------------------------------------------


class TestAmbiguity:
    def test_clear_request_has_no_clarification_need(self) -> None:
        provider = _mock_provider_returning(
            _intent_result(is_ambiguous=False, needs_clarification=False, clarification_question=None)
        )
        result = understand_intent(AgentState(original_request="Summarize this PDF."), provider)
        assert result.needs_clarification is False
        assert result.clarification_question is None

    def test_ambiguous_request_signals_clarification(self) -> None:
        provider = _mock_provider_returning(
            _intent_result(
                is_ambiguous=True,
                needs_clarification=True,
                clarification_question="Which document would you like summarized?",
            )
        )
        result = understand_intent(AgentState(original_request="Summarize it."), provider)
        assert result.is_ambiguous is True
        assert result.needs_clarification is True
        assert result.clarification_question == "Which document would you like summarized?"


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    def test_malformed_provider_output_propagates_unmodified(self) -> None:
        provider = _mock_provider_raising(
            ExternalProviderError(
                "Gemini's response could not be validated into IntentResult.",
                details={"reason": "invalid_structured_output"},
            )
        )
        with pytest.raises(ExternalProviderError) as exc_info:
            understand_intent(AgentState(original_request="x"), provider)
        assert exc_info.value.details["reason"] == "invalid_structured_output"

    def test_provider_failure_propagates_unmodified(self) -> None:
        provider = _mock_provider_raising(ExternalProviderError("rate limited", details={"reason": "rate_limited"}))
        with pytest.raises(ExternalProviderError) as exc_info:
            understand_intent(AgentState(original_request="x"), provider)
        assert exc_info.value.details["reason"] == "rate_limited"
