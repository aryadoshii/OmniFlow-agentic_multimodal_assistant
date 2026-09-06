"""Tests for the OmniFlow LangGraph workflow (Phase 4.2 skeleton + Phase 4.4 execution wiring).

Most nodes remain honest placeholders (no real intent/planning/RAG/LLM logic
exists yet to test) -- ``execute_tool`` is the one exception (Phase 4.4),
wired to a real (but test-local dummy) ToolRegistry. No Gemini, YouTube,
FAISS, or any external service is called anywhere in this file.
"""

import json
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from backend.agents.intent import IntentResult, IntentType
from backend.agents.planner import Plan, PlanStep, _PlanSchema, _PlanStepSchema
from backend.exceptions import InvalidInputError, OrchestrationError
from backend.graph import build_graph, run_graph
from backend.graph.routing import route_after_check_clarity, route_after_route_next
from backend.models.state import AgentState, WorkflowStatus
from backend.models.trace import ToolExecutionTrace
from backend.providers.base import BaseLLMProvider
from backend.tools.base import BaseTool
from backend.tools.registry import ToolRegistry


class _EchoInput(BaseModel):
    message: str = "hello"


class _EchoOutput(BaseModel):
    echoed: str


class _EchoTool(BaseTool):
    """A trivial, deterministic dummy tool used only by these graph tests."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echoes its input message."

    @property
    def input_model(self):
        return _EchoInput

    def run(self, tool_input: _EchoInput) -> _EchoOutput:
        return _EchoOutput(echoed=tool_input.message)


class _AlwaysFailsTool(BaseTool):
    """A dummy tool whose run() always raises, for failure-path tests."""

    @property
    def name(self) -> str:
        return "always_fails"

    @property
    def description(self) -> str:
        return "Always raises for failure-path tests."

    @property
    def input_model(self):
        return _EchoInput

    def run(self, tool_input: _EchoInput) -> _EchoOutput:
        raise RuntimeError("simulated internal tool bug")


def _registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_EchoTool())
    return registry


def _step(step_id: int, tool_name: str | None = None, **overrides) -> PlanStep:
    defaults = dict(
        step_id=step_id,
        tool_name=tool_name,
        purpose="test purpose",
        inputs={"message": "hi"} if tool_name == "echo" else {},
        expected_result="test result",
        depends_on=None,
    )
    defaults.update(overrides)
    return PlanStep(**defaults)


def _fake_intent_result(**overrides) -> IntentResult:
    defaults = dict(intent=IntentType.QUESTION_ANSWERING, explanation="test explanation")
    defaults.update(overrides)
    return IntentResult(**defaults)


def _plan_to_wire_schema(plan: Plan) -> _PlanSchema:
    """Converts a Plan into the Gemini-facing wire shape (inputs as a JSON
    string) a real GeminiProvider actually returns -- see planner.py's
    _PlanSchema docstring."""
    return _PlanSchema(
        steps=[
            _PlanStepSchema(
                step_id=step.step_id,
                tool_name=step.tool_name,
                purpose=step.purpose,
                inputs=json.dumps(step.inputs),
                expected_result=step.expected_result,
                depends_on=step.depends_on,
            )
            for step in plan.steps
        ]
    )


class _FakeLLMProvider(BaseLLMProvider):
    """Deterministic BaseLLMProvider test double for graph-level tests.

    Returns pre-canned structured outputs keyed by response_model, so these
    tests exercise the REAL understand_intent/plan/synthesize nodes (Phase
    4.5/4.7) with no network access. A list value is consumed one item per
    call to that response model (repeating the final item for any extra
    calls, e.g. a replan cycle this test didn't anticipate); a bare value
    is returned unchanged on every call.

    ``text_outputs`` configures generate() (used only by synthesize) the
    same way; it defaults to a plain, non-empty, unconstrained answer so
    tests that don't care about synthesis/validation behavior get a single
    clean synthesize/validate_output pass rather than triggering the
    bounded structural-correction loop.
    """

    def __init__(self, outputs: dict, text_outputs: list[str] | str = "This is a test answer.") -> None:
        self._outputs = outputs
        self._text_outputs = text_outputs
        self._call_counts: dict = {}
        self._text_call_count = 0

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        if isinstance(self._text_outputs, list):
            idx = self._text_call_count
            self._text_call_count += 1
            return self._text_outputs[min(idx, len(self._text_outputs) - 1)]
        return self._text_outputs

    def generate_structured(self, prompt, response_model, system_instruction=None):
        # The planner requests _PlanSchema (the Gemini-safe wire shape),
        # not Plan directly -- see planner.py's _PlanSchema docstring. Test
        # fixtures declare their canned output as a plain Plan for
        # readability, so translate it here rather than in every test.
        lookup_model = Plan if response_model is _PlanSchema else response_model
        value = self._outputs[lookup_model]
        if isinstance(value, list):
            idx = self._call_counts.get(lookup_model, 0)
            self._call_counts[lookup_model] = idx + 1
            value = value[min(idx, len(value) - 1)]
        if lookup_model is Plan and response_model is _PlanSchema:
            return _plan_to_wire_schema(value)
        return value


# ---------------------------------------------------------------------------
# Graph construction / compilation
# ---------------------------------------------------------------------------


class TestGraphConstruction:
    def test_build_graph_returns_compiled_graph(self) -> None:
        graph = build_graph()
        assert graph is not None
        # A compiled LangGraph graph exposes .invoke -- confirms real compilation.
        assert hasattr(graph, "invoke")

    def test_build_graph_can_be_called_multiple_times_independently(self) -> None:
        """No shared/global state should leak between separately built graphs."""
        graph_a = build_graph()
        graph_b = build_graph()
        assert graph_a is not graph_b

    def test_all_expected_nodes_present(self) -> None:
        graph = build_graph()
        node_names = set(graph.get_graph().nodes.keys())
        expected = {
            "prepare_context",
            "understand_intent",
            "check_clarity",
            "clarification",
            "plan",
            "execute_tool",
            "observe_result",
            "route_next",
            "synthesize",
            "validate_output",
        }
        assert expected <= node_names


# ---------------------------------------------------------------------------
# Initial state acceptance / field preservation
# ---------------------------------------------------------------------------


class TestInitialStateAcceptance:
    def test_accepts_default_agent_state(self) -> None:
        graph = build_graph()
        result = run_graph(AgentState(), compiled_graph=graph)
        assert isinstance(result, AgentState)

    def test_preserves_original_request_and_session_id(self) -> None:
        graph = build_graph()
        state = AgentState(original_request="Summarize this document.", session_id="sess-42")
        result = run_graph(state, compiled_graph=graph)
        assert result.original_request == "Summarize this document."
        assert result.session_id == "sess-42"

    def test_preserves_uploaded_and_normalized_document_fields(self) -> None:
        from backend.models.document import (
            ExtractionMethod,
            NormalizedDocument,
            SourceType,
        )
        from backend.models.request import UploadedInput

        graph = build_graph()
        state = AgentState(
            uploaded_inputs=[UploadedInput(filename="a.txt", mime_type="text/plain", size_bytes=10)],
            normalized_documents=[
                NormalizedDocument(
                    filename="a.txt",
                    source_type=SourceType.TEXT,
                    mime_type="text/plain",
                    content="hello",
                    extraction_method=ExtractionMethod.DIRECT_INPUT,
                )
            ],
        )
        result = run_graph(state, compiled_graph=graph)
        assert len(result.uploaded_inputs) == 1
        assert result.uploaded_inputs[0].filename == "a.txt"
        assert len(result.normalized_documents) == 1
        assert result.normalized_documents[0].content == "hello"

    def test_preserves_constraints_and_references(self) -> None:
        graph = build_graph()
        state = AgentState(constraints=["be concise"], references=["Q3 report"])
        result = run_graph(state, compiled_graph=graph)
        assert result.constraints == ["be concise"]
        assert result.references == ["Q3 report"]


# ---------------------------------------------------------------------------
# prepare_context: unified_context aggregation
# ---------------------------------------------------------------------------


class TestPrepareContext:
    """Hardening pass: unified_context must carry document metadata (e.g. an
    audio clip's duration_seconds), not just content -- synthesis reads
    only unified_context, never normalized_documents directly, so a
    metadata-only fact would otherwise never reach the final answer."""

    def test_no_documents_is_a_noop(self) -> None:
        from backend.graph.nodes import prepare_context

        update = prepare_context(AgentState())
        assert "unified_context" not in update

    def test_content_included_in_unified_context(self) -> None:
        from backend.graph.nodes import prepare_context
        from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType

        doc = NormalizedDocument(
            filename="report.pdf",
            source_type=SourceType.PDF,
            mime_type="application/pdf",
            content="Quarterly revenue rose 15%.",
            extraction_method=ExtractionMethod.NATIVE_TEXT,
        )
        update = prepare_context(AgentState(normalized_documents=[doc]))
        assert "Quarterly revenue rose 15%." in update["unified_context"]

    def test_audio_duration_metadata_reaches_unified_context(self) -> None:
        from backend.graph.nodes import prepare_context
        from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType

        doc = NormalizedDocument(
            filename="call.mp3",
            source_type=SourceType.AUDIO,
            mime_type="audio/mpeg",
            content="Welcome to the call.",
            extraction_method=ExtractionMethod.SPEECH_TO_TEXT,
            metadata={"duration_seconds": 125.4, "language": "en"},
        )
        update = prepare_context(AgentState(normalized_documents=[doc]))
        assert "125.4" in update["unified_context"]
        assert "duration_seconds" in update["unified_context"]

    def test_multiple_documents_each_carry_their_own_metadata(self) -> None:
        from backend.graph.nodes import prepare_context
        from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType

        audio_doc = NormalizedDocument(
            filename="call.mp3",
            source_type=SourceType.AUDIO,
            mime_type="audio/mpeg",
            content="Call transcript.",
            extraction_method=ExtractionMethod.SPEECH_TO_TEXT,
            metadata={"duration_seconds": 60.0},
        )
        pdf_doc = NormalizedDocument(
            filename="report.pdf",
            source_type=SourceType.PDF,
            mime_type="application/pdf",
            content="PDF body text.",
            extraction_method=ExtractionMethod.NATIVE_TEXT,
            metadata={"total_pages": 3},
        )
        update = prepare_context(AgentState(normalized_documents=[audio_doc, pdf_doc]))
        assert "60.0" in update["unified_context"]
        assert "total_pages" in update["unified_context"]
        assert "Call transcript." in update["unified_context"]
        assert "PDF body text." in update["unified_context"]


# ---------------------------------------------------------------------------
# Graph execution with placeholder nodes
# ---------------------------------------------------------------------------


class TestGraphExecution:
    def test_default_run_with_no_llm_provider_fails_honestly(self) -> None:
        """As of Phase 4.5, understand_intent/plan require a real
        llm_provider. Running the default graph with none configured
        correctly reports FAILED with a clear reason, rather than silently
        reporting COMPLETED despite never having understood the request."""
        graph = build_graph()  # no provider, no registry
        result = run_graph(AgentState(), compiled_graph=graph)
        assert result.status == WorkflowStatus.FAILED
        assert "No llm_provider was configured" in result.errors[0]

    def test_run_with_valid_plan_and_registered_tool_completes(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(),
                Plan: [Plan(steps=[_step(0, tool_name="echo")]), Plan(steps=[])],
            }
        )
        graph = build_graph(_registry_with_echo(), llm_provider=provider)
        result = run_graph(AgentState(original_request="echo something"), compiled_graph=graph)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.errors == []

    def test_direct_answer_plan_with_null_tool_step_reaches_synthesis_without_replanning(self) -> None:
        """Regression test for a real bug: a legitimate direct-answer plan
        (planner.py's own documented case -- "'Summarize this PDF' with
        full content already available -> ... a single direct-synthesis
        step suffices" -- using tool_name=None) must reach synthesis after
        ONE plan/execute/observe cycle, not loop back into replanning
        merely because the plan list is technically non-empty (a single
        PlanStep, not zero).

        Configures a bare (non-list) Plan value, so the SAME null-tool plan
        would be returned again on every call if the graph ever replanned:
        if the bug were still present, this would loop until
        max_agent_steps and FAIL, never reaching synthesis. Also confirms
        no tool was ever invoked -- this is a direct-answer path, not a
        tool-execution one."""
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(intent=IntentType.SUMMARIZATION),
                Plan: Plan(
                    steps=[_step(0, tool_name=None, purpose="Summarize the already-ingested PDF content.")]
                ),
            }
        )
        graph = build_graph(ToolRegistry(), llm_provider=provider)
        result = run_graph(AgentState(original_request="Summarize this PDF."), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        assert result.errors == []
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:plan") == 1
        assert step_names.count("node:synthesize") == 1
        assert not any(name.startswith("tool:") for name in step_names)

    def test_multi_tool_plan_still_replans_as_before(self) -> None:
        """Confirms the fix is narrowly scoped: a plan whose steps use real
        tools continues to replan after every cycle exactly as before."""
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(),
                Plan: [Plan(steps=[_step(0, tool_name="echo")]), Plan(steps=[])],
            }
        )
        graph = build_graph(_registry_with_echo(), llm_provider=provider)
        result = run_graph(AgentState(original_request="echo something"), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:plan") == 2
        assert step_names.count("tool:echo") == 1

    def test_clarification_needed_routes_to_clarification_and_stops(self) -> None:
        graph = build_graph()
        result = run_graph(AgentState(clarification_needed=True), compiled_graph=graph)
        assert result.status == WorkflowStatus.AWAITING_CLARIFICATION
        step_names = [t.step_name for t in result.execution_trace]
        assert "node:clarification" in step_names
        # The clarification path must never reach plan/execute_tool/synthesize.
        assert "node:plan" not in step_names
        assert "node:synthesize" not in step_names

    def test_two_replan_cycles_then_completes(self) -> None:
        """Phase 4.5 replans after every single execute/observe cycle (the
        planner returns AT MOST one actionable step per replan -- see
        planner.py) -- this exercises two such cycles before the planner
        signals completion with an empty plan."""
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(),
                Plan: [
                    Plan(steps=[_step(0, tool_name="echo")]),
                    Plan(steps=[_step(0, tool_name="echo")]),
                    Plan(steps=[]),
                ],
            }
        )
        graph = build_graph(_registry_with_echo(), llm_provider=provider)
        result = run_graph(AgentState(original_request="echo twice"), compiled_graph=graph)

        assert result.status == WorkflowStatus.COMPLETED
        step_names = [t.step_name for t in result.execution_trace]
        assert step_names.count("node:plan") == 3
        assert step_names.count("tool:echo") == 2
        assert step_names.count("node:observe_result") == 3
        assert step_names.count("node:route_next") == 3
        assert step_names.count("node:synthesize") == 1

    def test_nodes_do_not_mutate_semantic_fields_beyond_execute_tool(self) -> None:
        """Placeholder nodes must be honest: no real intent/answer content
        should appear just from running the graph, even though execute_tool
        (the one real node) correctly reports failure for a plan-less run."""
        graph = build_graph()
        result = run_graph(AgentState(original_request="anything"), compiled_graph=graph)
        assert result.detected_intent is None
        assert result.plan is None
        assert result.tool_results == {}
        assert result.retrieved_evidence is None
        assert result.final_answer is None
        assert result.clarification_prompt is None


# ---------------------------------------------------------------------------
# Request isolation
# ---------------------------------------------------------------------------


class TestRequestIsolation:
    def test_two_invocations_do_not_share_state(self) -> None:
        graph = build_graph()
        state_a = AgentState(original_request="request A", session_id="a")
        state_b = AgentState(original_request="request B", session_id="b")

        result_a = run_graph(state_a, compiled_graph=graph)
        result_b = run_graph(state_b, compiled_graph=graph)

        assert result_a.session_id == "a"
        assert result_b.session_id == "b"
        assert result_a.original_request != result_b.original_request
        assert result_a.execution_trace is not result_b.execution_trace

    def test_input_state_object_is_not_mutated_in_place(self) -> None:
        """The caller's original AgentState instance must remain untouched --
        nodes return updates, they do not mutate the input object."""
        graph = build_graph()
        original = AgentState(original_request="untouched")
        original_trace_len = len(original.execution_trace)

        run_graph(original, compiled_graph=graph)

        assert len(original.execution_trace) == original_trace_len
        assert original.status == WorkflowStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


class TestRoutingFunctions:
    def test_route_after_check_clarity_goes_to_plan_by_default(self) -> None:
        assert route_after_check_clarity(AgentState()) == "plan"

    def test_route_after_check_clarity_goes_to_clarification_when_flagged(self) -> None:
        assert route_after_check_clarity(AgentState(clarification_needed=True)) == "clarification"

    def test_route_after_route_next_goes_to_synthesize_when_no_plan(self) -> None:
        assert route_after_route_next(AgentState(plan=None, current_step=0)) == "synthesize"

    def test_route_after_route_next_goes_to_synthesize_when_no_steps_remain(self) -> None:
        state = AgentState(plan=Plan(steps=[]), current_step=0)
        assert route_after_route_next(state) == "synthesize"

    def test_route_after_route_next_goes_to_plan_when_steps_remain(self) -> None:
        state = AgentState(plan=Plan(steps=[_step(0), _step(1)]), current_step=0)
        assert route_after_route_next(state) == "plan"

    def test_route_after_route_next_replans_regardless_of_position_for_a_real_tool_step(self) -> None:
        """Phase 4.5 replans after every single execute/observe cycle, so
        this routing decision is driven by the step just executed, never by
        current_step's raw position -- a plan whose just-executed step used
        a real tool always triggers a replan, whatever that position is."""
        non_empty_plan = Plan(steps=[_step(0, tool_name="echo"), _step(1, tool_name="echo")])
        assert route_after_route_next(AgentState(plan=non_empty_plan, current_step=1)) == "plan"
        assert route_after_route_next(AgentState(plan=non_empty_plan, current_step=2)) == "plan"

        empty_plan = Plan(steps=[])
        assert route_after_route_next(AgentState(plan=empty_plan, current_step=0)) == "synthesize"

    def test_route_after_route_next_goes_to_synthesize_after_a_null_tool_step(self) -> None:
        """Regression test: a plan whose just-executed step has
        tool_name=None (planner.py's documented direct-answer case, e.g.
        "Summarize this PDF" needing no tool) is a no-op for execute_tool --
        it never updates tool_call_history/tool_results, so replanning would
        hand the planner an unchanged execution history and risk looping
        until max_agent_steps is exhausted instead of reaching synthesis."""
        plan = Plan(steps=[_step(0, tool_name=None)])
        state = AgentState(plan=plan, current_step=1)  # observe_result already advanced past index 0
        assert route_after_route_next(state) == "synthesize"

    def test_route_after_route_next_replans_when_a_later_step_is_a_real_tool(self) -> None:
        """A null-tool step at an EARLIER position must not suppress a
        replan triggered by a real-tool step at the position just executed."""
        plan = Plan(steps=[_step(0, tool_name=None), _step(1, tool_name="echo")])
        state = AgentState(plan=plan, current_step=2)  # just executed index 1, a real tool
        assert route_after_route_next(state) == "plan"

    def test_route_after_route_next_stops_immediately_on_prior_failure(self) -> None:
        """A FAILED status must short-circuit straight to synthesize, even if
        steps technically remain -- no automatic retry of subsequent steps."""
        plan = Plan(steps=[_step(0), _step(1)])
        state = AgentState(plan=plan, current_step=0, status=WorkflowStatus.FAILED)
        assert route_after_route_next(state) == "synthesize"


# ---------------------------------------------------------------------------
# No external calls
# ---------------------------------------------------------------------------


class TestNoExternalCalls:
    def test_no_direct_provider_or_concrete_tool_imports_in_graph_package(self) -> None:
        """The graph may import ToolRegistry/BaseTool (the abstraction it is
        explicitly meant to dispatch through, per Phase 4.4), but must never
        import a concrete tool implementation, RAGService/FAISS internals, or
        the Gemini SDK/provider directly."""
        import backend.graph.builder as builder_module
        import backend.graph.nodes as nodes_module
        import backend.graph.routing as routing_module

        forbidden = [
            "google.genai",
            "backend.providers.gemini_provider",
            "backend.tools.youtube",
            "backend.tools.rag_search",
            "backend.rag.vector_store",
            "backend.rag.service",
            "backend.rag.embeddings",
        ]
        for module in (builder_module, nodes_module, routing_module):
            with open(module.__file__, encoding="utf-8") as f:
                import_lines = [
                    line.lower()
                    for line in f.readlines()
                    if line.strip().startswith("import ") or line.strip().startswith("from ")
                ]
            for line in import_lines:
                for term in forbidden:
                    assert term not in line, f"Forbidden import '{term}' in {module.__file__}: {line!r}"

    def test_graph_package_only_imports_tool_registry_abstraction(self) -> None:
        """Confirms the ONLY backend.tools import in the graph package is
        the registry/base abstraction, never a concrete tool module."""
        import backend.graph.builder as builder_module
        import backend.graph.nodes as nodes_module

        allowed_tools_imports = {"backend.tools.registry", "backend.tools.base"}
        for module in (builder_module, nodes_module):
            with open(module.__file__, encoding="utf-8") as f:
                for line in f.readlines():
                    stripped = line.strip()
                    if "backend.tools" in stripped and (
                        stripped.startswith("import ") or stripped.startswith("from ")
                    ):
                        assert any(allowed in stripped for allowed in allowed_tools_imports), (
                            f"Unexpected backend.tools import in {module.__file__}: {stripped!r}"
                        )

    def test_build_graph_makes_no_network_calls(self) -> None:
        """Building and running the default graph must never touch a mocked
        Gemini client -- proving no accidental LLM call path exists yet."""
        with patch("backend.providers.gemini_provider.genai") as mock_genai:
            graph = build_graph()
            run_graph(AgentState(original_request="anything"), compiled_graph=graph)
            mock_genai.Client.assert_not_called()


# ---------------------------------------------------------------------------
# Trace behavior
# ---------------------------------------------------------------------------


class TestTraceBehavior:
    def test_trace_entries_are_tool_execution_trace_instances(self) -> None:
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(),
                Plan: [Plan(steps=[_step(0, tool_name="echo")]), Plan(steps=[])],
            }
        )
        graph = build_graph(_registry_with_echo(), llm_provider=provider)
        result = run_graph(AgentState(original_request="echo something"), compiled_graph=graph)
        assert len(result.execution_trace) > 0
        for entry in result.execution_trace:
            assert isinstance(entry, ToolExecutionTrace)
            assert entry.status == "success"
            assert entry.duration_ms is not None

    def test_failed_run_records_a_failed_trace_entry(self) -> None:
        """With intent/plan succeeding but the selected tool's own execution
        raising, execute_tool is the sole failure point in the run."""
        registry = ToolRegistry()
        registry.register(_AlwaysFailsTool())
        provider = _FakeLLMProvider(
            {
                IntentResult: _fake_intent_result(),
                Plan: [Plan(steps=[_step(0, tool_name="always_fails")])],
            }
        )
        graph = build_graph(registry, llm_provider=provider)
        result = run_graph(AgentState(original_request="do something"), compiled_graph=graph)
        failed_entries = [e for e in result.execution_trace if e.status == "failed"]
        assert len(failed_entries) == 1
        assert failed_entries[0].step_name == "node:execute_tool"
        assert failed_entries[0].error_message is not None

    def test_trace_entries_use_node_prefixed_step_names_for_non_tool_nodes(self) -> None:
        graph = build_graph()
        result = run_graph(AgentState(), compiled_graph=graph)
        for entry in result.execution_trace:
            if entry.tool_name is None:
                assert entry.step_name.startswith("node:")

    def test_successful_tool_step_trace_carries_the_real_tool_name(self) -> None:
        graph = build_graph(_registry_with_echo())
        plan = Plan(steps=[_step(0, tool_name="echo")])
        result = run_graph(AgentState(plan=plan), compiled_graph=graph)
        tool_entries = [e for e in result.execution_trace if e.tool_name == "echo"]
        assert len(tool_entries) == 1
        assert tool_entries[0].step_name == "tool:echo"

    def test_trace_never_contains_raw_request_content(self) -> None:
        """Trace details must never leak prompt/document content."""
        graph = build_graph()
        sensitive_text = "SENSITIVE_USER_PROMPT_CONTENT_MARKER"
        result = run_graph(AgentState(original_request=sensitive_text), compiled_graph=graph)
        for entry in result.execution_trace:
            assert sensitive_text not in str(entry.details)
            assert sensitive_text not in (entry.error_message or "")


# ---------------------------------------------------------------------------
# Error handling boundary
# ---------------------------------------------------------------------------


class TestErrorHandlingBoundary:
    def test_unexpected_exception_wrapped_as_orchestration_error(self) -> None:
        # The graph must be built INSIDE the patch context: add_node() binds
        # the function reference at build time, so patching after build_graph()
        # would not affect the already-compiled graph.
        with patch("backend.graph.nodes.prepare_context", side_effect=RuntimeError("boom")):
            graph = build_graph()
            with pytest.raises(OrchestrationError) as exc_info:
                run_graph(AgentState(), compiled_graph=graph)
        assert exc_info.value.details["error_type"] == "RuntimeError"

    def test_domain_exception_propagates_unmodified(self) -> None:
        """A future node raising an existing OmniFlow domain exception must
        not be flattened into a generic OrchestrationError."""
        with patch(
            "backend.graph.nodes.prepare_context",
            side_effect=InvalidInputError("simulated future node validation failure"),
        ):
            graph = build_graph()
            with pytest.raises(InvalidInputError) as exc_info:
                run_graph(AgentState(), compiled_graph=graph)
        assert exc_info.value.error_code == "INVALID_INPUT"
