"""Tests for final response synthesis (Phase 4.7).

The LLM provider is fully mocked -- no real Gemini call is ever made.
Verifies: synthesize_answer() plumbs a plain-text generate() call (never
generate_structured), includes the right context/tool-results/constraints
in the prompt, handles the correction-feedback pass, and propagates
provider failures unmodified.
"""

from unittest.mock import MagicMock

from pydantic import BaseModel

from omniflow.agents.intent import IntentResult, IntentType
from omniflow.agents.synthesizer import synthesize_answer
from omniflow.models.state import AgentState


class _DummyToolOutput(BaseModel):
    status: str
    detail: str


def _mock_provider_returning(text: str) -> MagicMock:
    provider = MagicMock()
    provider.generate.return_value = text
    return provider


def _mock_provider_raising(exc: BaseException) -> MagicMock:
    provider = MagicMock()
    provider.generate.side_effect = exc
    return provider


def _intent_result(**overrides) -> IntentResult:
    defaults = dict(
        intent=IntentType.SUMMARIZATION,
        constraints=[],
        referenced_inputs=[],
        relevant_references=[],
        is_ambiguous=False,
        needs_clarification=False,
        clarification_question=None,
        explanation="Summarize the document.",
    )
    defaults.update(overrides)
    return IntentResult(**defaults)


class TestBasicSynthesis:
    def test_returns_provider_text_stripped(self) -> None:
        provider = _mock_provider_returning("  A clean answer.  \n")
        state = AgentState(original_request="Summarize this.")
        result = synthesize_answer(state, provider)
        assert result == "A clean answer."

    def test_uses_generate_not_generate_structured(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="Summarize this.")
        synthesize_answer(state, provider)
        provider.generate.assert_called_once()
        provider.generate_structured.assert_not_called()

    def test_prompt_includes_original_request(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="What is the Q3 revenue?")
        synthesize_answer(state, provider)
        prompt_arg = provider.generate.call_args[0][0]
        assert "What is the Q3 revenue?" in prompt_arg

    def test_prompt_includes_unified_context(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="x", unified_context="Extracted PDF content here.")
        synthesize_answer(state, provider)
        prompt_arg = provider.generate.call_args[0][0]
        assert "Extracted PDF content here." in prompt_arg

    def test_prompt_includes_intent_and_constraints(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(
            original_request="x",
            intent_result=_intent_result(
                intent=IntentType.SENTIMENT_ANALYSIS, constraints=["one-line summary"]
            ),
        )
        synthesize_answer(state, provider)
        prompt_arg = provider.generate.call_args[0][0]
        assert "sentiment_analysis" in prompt_arg
        assert "one-line summary" in prompt_arg

    def test_prompt_includes_tool_results(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(
            original_request="x",
            tool_results={"1": _DummyToolOutput(status="evidence_found", detail="Q3 revenue rose.")},
        )
        synthesize_answer(state, provider)
        prompt_arg = provider.generate.call_args[0][0]
        assert "evidence_found" in prompt_arg
        assert "Q3 revenue rose." in prompt_arg

    def test_prompt_includes_errors_when_present(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="x", errors=["Step 0 (rag_search) failed: boom"])
        synthesize_answer(state, provider)
        prompt_arg = provider.generate.call_args[0][0]
        assert "boom" in prompt_arg


class TestCorrectionPass:
    def test_correction_feedback_included_in_prompt(self) -> None:
        provider = _mock_provider_returning("A corrected answer.")
        state = AgentState(original_request="x", final_answer="- only one bullet")
        synthesize_answer(state, provider, correction_feedback="Constraint requires 3 bullets.")
        prompt_arg = provider.generate.call_args[0][0]
        assert "Constraint requires 3 bullets." in prompt_arg
        assert "- only one bullet" in prompt_arg

    def test_correction_system_instruction_differs_from_initial(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="x")

        synthesize_answer(state, provider)
        first_system_instruction = provider.generate.call_args.kwargs["system_instruction"]

        synthesize_answer(state, provider, correction_feedback="Missing a required section.")
        second_system_instruction = provider.generate.call_args.kwargs["system_instruction"]

        assert first_system_instruction != second_system_instruction
        assert "CORRECTION" in second_system_instruction

    def test_no_correction_feedback_omits_previous_answer_section(self) -> None:
        provider = _mock_provider_returning("An answer.")
        state = AgentState(original_request="x", final_answer="stale prior answer")
        synthesize_answer(state, provider, correction_feedback=None)
        prompt_arg = provider.generate.call_args[0][0]
        assert "stale prior answer" not in prompt_arg


class TestIntentSpecificSystemInstruction:
    """Hardening pass: the system instruction must explicitly ask for the
    assignment-required output shapes per intent, not just a generic
    'explain'/'assess' instruction -- otherwise sentiment/code responses
    have no reliable reason to include a confidence level, bug callouts, or
    a complexity assessment."""

    def test_system_instruction_requests_sentiment_confidence(self) -> None:
        provider = _mock_provider_returning("An answer.")
        synthesize_answer(AgentState(original_request="x"), provider)
        system_instruction = provider.generate.call_args.kwargs["system_instruction"]
        assert "confidence" in system_instruction.lower()
        assert "label" in system_instruction.lower()

    def test_system_instruction_requests_code_bugs_and_complexity(self) -> None:
        provider = _mock_provider_returning("An answer.")
        synthesize_answer(AgentState(original_request="x"), provider)
        system_instruction = provider.generate.call_args.kwargs["system_instruction"]
        assert "bug" in system_instruction.lower()
        assert "complexity" in system_instruction.lower()
        assert "programming language" in system_instruction.lower()

    def test_system_instruction_requests_audio_duration_when_available(self) -> None:
        provider = _mock_provider_returning("An answer.")
        synthesize_answer(AgentState(original_request="x"), provider)
        system_instruction = provider.generate.call_args.kwargs["system_instruction"]
        assert "duration" in system_instruction.lower()


class TestFailurePropagation:
    def test_provider_failure_propagates_unmodified(self) -> None:
        from omniflow.exceptions import ExternalProviderError

        provider = _mock_provider_raising(ExternalProviderError("rate limited"))
        state = AgentState(original_request="x")
        try:
            synthesize_answer(state, provider)
        except ExternalProviderError as exc:
            assert exc.message == "rate limited"
        else:
            raise AssertionError("Expected ExternalProviderError to propagate.")
