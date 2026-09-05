"""Tests for the structured planner (Phase 4.3).

The LLM provider is fully mocked -- no real Gemini call is ever made, and
no tool is ever actually invoked (ToolRegistry.execute is never called from
this module or these tests). Verifies: prompt/schema plumbing, minimum-plan
shape, tool-catalog scoping, nonexistent-tool rejection, and failure
propagation.
"""

from unittest.mock import MagicMock

import pytest

from omniflow.agents.intent import IntentResult, IntentType
from omniflow.agents.planner import Plan, PlanStep, create_plan
from omniflow.exceptions import ExternalProviderError, OrchestrationError
from omniflow.tools.base import BaseTool
from omniflow.tools.registry import ToolRegistry
from pydantic import BaseModel


def _mock_provider_returning(plan: Plan) -> MagicMock:
    provider = MagicMock()
    provider.generate_structured.return_value = plan
    return provider


def _mock_provider_raising(exc: BaseException) -> MagicMock:
    provider = MagicMock()
    provider.generate_structured.side_effect = exc
    return provider


class _DummyInput(BaseModel):
    query: str = "x"


class _DummyOutput(BaseModel):
    result: str = "ok"


class _DummyTool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Dummy tool '{self._name}' for planner tests."

    @property
    def input_model(self):
        return _DummyInput

    def run(self, tool_input: _DummyInput) -> _DummyOutput:  # pragma: no cover - never actually called
        raise AssertionError("Planner must never execute a tool.")


def _registry_with(*tool_names: str) -> ToolRegistry:
    registry = ToolRegistry()
    for name in tool_names:
        registry.register(_DummyTool(name))
    return registry


def _intent(**overrides) -> IntentResult:
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


def _step(step_id: int = 0, tool_name: str | None = None, **overrides) -> PlanStep:
    defaults = dict(
        step_id=step_id,
        tool_name=tool_name,
        purpose="Do the thing.",
        inputs={},
        expected_result="The thing is done.",
        depends_on=None,
    )
    defaults.update(overrides)
    return PlanStep(**defaults)


# ---------------------------------------------------------------------------
# Minimum viable plan shape
# ---------------------------------------------------------------------------


class TestMinimumViablePlan:
    def test_no_tool_needed_produces_single_null_tool_step(self) -> None:
        """'Summarize this PDF' with full content already available -> no
        RAG/YouTube tool needed, a single direct-synthesis step suffices."""
        plan = Plan(steps=[_step(0, tool_name=None, purpose="Summarize the already-ingested PDF content.")])
        provider = _mock_provider_returning(plan)
        registry = _registry_with("rag_search", "youtube_transcript")

        result = create_plan(_intent(), provider, registry)

        assert len(result.steps) == 1
        assert result.steps[0].tool_name is None

    def test_plan_with_registered_tool_accepted(self) -> None:
        plan = Plan(steps=[_step(0, tool_name="rag_search", purpose="Search the report for revenue figures.")])
        provider = _mock_provider_returning(plan)
        registry = _registry_with("rag_search")

        result = create_plan(_intent(intent=IntentType.QUESTION_ANSWERING), provider, registry)

        assert result.steps[0].tool_name == "rag_search"

    def test_multi_step_plan_with_dependency(self) -> None:
        """'Summarize the YouTube video in this PDF' -> transcript first, then synthesis."""
        plan = Plan(
            steps=[
                _step(0, tool_name="youtube_transcript", purpose="Fetch the video transcript."),
                _step(1, tool_name=None, purpose="Summarize the transcript.", depends_on=0),
            ]
        )
        provider = _mock_provider_returning(plan)
        registry = _registry_with("youtube_transcript")

        result = create_plan(_intent(intent=IntentType.TRANSCRIPTION_SUMMARY), provider, registry)

        assert len(result.steps) == 2
        assert result.steps[1].depends_on == 0

    def test_comparison_plan_uses_both_sources(self) -> None:
        plan = Plan(
            steps=[
                _step(0, tool_name=None, purpose="Extract key points from the PDF."),
                _step(1, tool_name=None, purpose="Extract key points from the audio transcript.", depends_on=None),
                _step(2, tool_name=None, purpose="Compare the two sets of key points.", depends_on=1),
            ]
        )
        provider = _mock_provider_returning(plan)
        registry = _registry_with()

        result = create_plan(_intent(intent=IntentType.COMPARISON), provider, registry)

        assert len(result.steps) == 3


# ---------------------------------------------------------------------------
# Tool catalog scoping / nonexistent tool rejection
# ---------------------------------------------------------------------------


class TestToolCatalogScoping:
    def test_tool_catalog_passed_to_prompt(self) -> None:
        provider = _mock_provider_returning(Plan(steps=[_step(0)]))
        registry = _registry_with("rag_search", "youtube_transcript")

        create_plan(_intent(), provider, registry)

        prompt_arg = provider.generate_structured.call_args[0][0]
        assert "rag_search" in prompt_arg
        assert "youtube_transcript" in prompt_arg

    def test_response_model_argument_is_plan(self) -> None:
        provider = _mock_provider_returning(Plan(steps=[_step(0)]))
        registry = _registry_with()
        create_plan(_intent(), provider, registry)
        call_args = provider.generate_structured.call_args
        assert call_args[0][1] is Plan

    def test_nonexistent_tool_raises_orchestration_error(self) -> None:
        """The LLM hallucinating a tool name must be rejected, never executed."""
        plan = Plan(steps=[_step(0, tool_name="tool_that_does_not_exist")])
        provider = _mock_provider_returning(plan)
        registry = _registry_with("rag_search")

        with pytest.raises(OrchestrationError) as exc_info:
            create_plan(_intent(), provider, registry)

        assert exc_info.value.error_code == "ORCHESTRATION_ERROR"
        assert exc_info.value.details["tool_name"] == "tool_that_does_not_exist"

    def test_nonexistent_tool_in_second_step_also_rejected(self) -> None:
        plan = Plan(
            steps=[
                _step(0, tool_name="rag_search"),
                _step(1, tool_name="invented_tool", depends_on=0),
            ]
        )
        provider = _mock_provider_returning(plan)
        registry = _registry_with("rag_search")

        with pytest.raises(OrchestrationError):
            create_plan(_intent(), provider, registry)

    def test_empty_registry_allows_only_null_tool_steps(self) -> None:
        provider = _mock_provider_returning(Plan(steps=[_step(0, tool_name=None)]))
        registry = _registry_with()
        result = create_plan(_intent(), provider, registry)
        assert result.steps[0].tool_name is None

    def test_empty_registry_rejects_any_tool_selection(self) -> None:
        provider = _mock_provider_returning(Plan(steps=[_step(0, tool_name="rag_search")]))
        registry = _registry_with()  # no tools registered at all
        with pytest.raises(OrchestrationError):
            create_plan(_intent(), provider, registry)

    def test_planner_never_executes_a_tool(self) -> None:
        """DummyTool.run() raises AssertionError if ever called -- confirms
        the planner only introspects the registry, never invokes execute()."""
        provider = _mock_provider_returning(Plan(steps=[_step(0, tool_name="rag_search")]))
        registry = _registry_with("rag_search")
        create_plan(_intent(), provider, registry)  # must not raise AssertionError


# ---------------------------------------------------------------------------
# Structured output / failure propagation
# ---------------------------------------------------------------------------


class TestStructuredOutputAndFailures:
    def test_uses_generate_structured_not_generate(self) -> None:
        provider = _mock_provider_returning(Plan(steps=[_step(0)]))
        registry = _registry_with()
        create_plan(_intent(), provider, registry)
        provider.generate_structured.assert_called_once()
        provider.generate.assert_not_called()

    def test_malformed_provider_output_propagates_unmodified(self) -> None:
        provider = _mock_provider_raising(
            ExternalProviderError(
                "Gemini's response could not be validated into Plan.",
                details={"reason": "invalid_structured_output"},
            )
        )
        registry = _registry_with()
        with pytest.raises(ExternalProviderError) as exc_info:
            create_plan(_intent(), provider, registry)
        assert exc_info.value.details["reason"] == "invalid_structured_output"

    def test_provider_failure_propagates_unmodified(self) -> None:
        provider = _mock_provider_raising(ExternalProviderError("down", details={"reason": "service_unavailable"}))
        registry = _registry_with()
        with pytest.raises(ExternalProviderError) as exc_info:
            create_plan(_intent(), provider, registry)
        assert exc_info.value.details["reason"] == "service_unavailable"

    def test_empty_plan_is_valid(self) -> None:
        """A plan with zero steps (e.g. general conversation, no tools/context
        needed) is structurally valid, not an error."""
        provider = _mock_provider_returning(Plan(steps=[]))
        registry = _registry_with()
        result = create_plan(_intent(intent=IntentType.GENERAL_CONVERSATION), provider, registry)
        assert result.steps == []
