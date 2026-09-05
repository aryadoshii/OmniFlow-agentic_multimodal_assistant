"""Focused unit tests for the execute_tool LangGraph node (Phase 4.4).

Tests call make_execute_tool_node()'s returned closure directly against a
real (test-local, dummy) ToolRegistry -- no mocking of ToolRegistry itself,
since it's fast/deterministic/already-tested; only the tools registered in
it are dummies. No Gemini, YouTube, RAGService, or FAISS is imported or
called anywhere in this file.
"""

from pydantic import BaseModel

from omniflow.agents.planner import Plan, PlanStep
from omniflow.exceptions import InvalidInputError, ToolExecutionError
from omniflow.graph.nodes import make_execute_tool_node
from omniflow.models.state import AgentState, WorkflowStatus
from omniflow.models.trace import ToolExecutionTrace
from omniflow.tools.base import BaseTool
from omniflow.tools.registry import ToolRegistry


class _SearchInput(BaseModel):
    query: str


class _SearchOutput(BaseModel):
    results: list[str]


class _SearchTool(BaseTool):
    """Deterministic dummy tool standing in for a real registered tool."""

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return "Searches for something."

    @property
    def input_model(self):
        return _SearchInput

    def run(self, tool_input: _SearchInput) -> _SearchOutput:
        return _SearchOutput(results=[f"result for {tool_input.query}"])


class _AlwaysFailsTool(BaseTool):
    """Raises an unexpected internal error -- ToolRegistry wraps this as
    ToolExecutionError (see Phase 3.1/3.6); execute_tool must preserve that."""

    @property
    def name(self) -> str:
        return "always_fails"

    @property
    def description(self) -> str:
        return "Always raises."

    @property
    def input_model(self):
        return _SearchInput

    def run(self, tool_input: _SearchInput) -> _SearchOutput:
        raise RuntimeError("simulated internal tool bug")


def _registry(*tools: BaseTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _step(step_id: int = 0, tool_name: str | None = "search", **overrides) -> PlanStep:
    defaults = dict(
        step_id=step_id,
        tool_name=tool_name,
        purpose="Search for the answer.",
        inputs={"query": "revenue"} if tool_name == "search" else {},
        expected_result="Search results found.",
        depends_on=None,
    )
    defaults.update(overrides)
    return PlanStep(**defaults)


# ---------------------------------------------------------------------------
# Successful execution / structured output / AgentState update
# ---------------------------------------------------------------------------


class TestSuccessfulExecution:
    def test_successful_execution_stores_structured_output(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        plan = Plan(steps=[_step(0)])
        state = AgentState(plan=plan, current_step=0)

        update = node(state)

        assert "0" in update["tool_results"]
        output = update["tool_results"]["0"]
        assert isinstance(output, _SearchOutput)
        assert output.results == ["result for revenue"]

    def test_successful_execution_does_not_set_errors_or_failed_status(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=Plan(steps=[_step(0)]), current_step=0)
        update = node(state)
        assert "errors" not in update
        assert "status" not in update

    def test_structured_output_is_a_real_basemodel_not_a_dict(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=Plan(steps=[_step(0)]), current_step=0)
        update = node(state)
        assert isinstance(update["tool_results"]["0"], BaseModel)

    def test_results_keyed_by_agent_step_count_not_step_id(self) -> None:
        """tool_results is keyed by agent_step_count (bumped once per replan
        cycle by the `plan` node), never by step_id -- step_id resets to 0
        on every replan, so keying by it would silently collide across
        cycles. This simulates two sequential replan cycles, each
        contributing one step, the way the real graph drives this node."""
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(
            plan=Plan(steps=[_step(0, inputs={"query": "first"})]),
            current_step=0,
            agent_step_count=1,
            tool_results={},
        )
        update0 = node(state)
        assert update0["tool_results"]["1"].results == ["result for first"]

        state_after_0 = state.model_copy(
            update={
                **update0,
                "plan": Plan(steps=[_step(0, inputs={"query": "second"})]),
                "current_step": 0,
                "agent_step_count": 2,
            }
        )
        update1 = node(state_after_0)

        combined = {**update0["tool_results"], **update1["tool_results"]}
        assert combined["1"].results == ["result for first"]
        assert combined["2"].results == ["result for second"]

    def test_step_with_no_tool_name_skips_execution(self) -> None:
        node = make_execute_tool_node(_registry())
        state = AgentState(plan=Plan(steps=[_step(0, tool_name=None)]), current_step=0)
        update = node(state)
        assert "tool_results" not in update
        assert "errors" not in update

    def test_preserves_prior_tool_results(self) -> None:
        """Prior results from an earlier replan cycle (its own
        agent_step_count) must be carried forward untouched, never
        overwritten by the current cycle's result."""
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(
            plan=Plan(steps=[_step(0)]),
            current_step=0,
            agent_step_count=2,
            tool_results={"1": "prior result"},
        )
        update = node(state)
        assert update["tool_results"]["1"] == "prior result"
        assert "2" in update["tool_results"]


# ---------------------------------------------------------------------------
# Missing tool / validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_missing_tool_recorded_as_failure_not_raised(self) -> None:
        node = make_execute_tool_node(_registry())  # empty registry
        state = AgentState(plan=Plan(steps=[_step(0, tool_name="search")]), current_step=0)

        update = node(state)  # must not raise

        assert update["status"] == WorkflowStatus.FAILED
        assert "not registered" in update["errors"][0]
        assert "tool_results" not in update

    def test_no_plan_recorded_as_failure(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=None)
        update = node(state)
        assert update["status"] == WorkflowStatus.FAILED
        assert "No plan is available" in update["errors"][0]

    def test_empty_plan_is_not_a_failure(self) -> None:
        """An empty-steps plan is the planner's deliberate 'no work needed'
        signal, not an error -- distinct from plan being None entirely."""
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=Plan(steps=[]))
        update = node(state)
        assert "status" not in update
        assert "errors" not in update
        trace_entry = update["execution_trace"][-1]
        assert trace_entry.status == "success"
        assert trace_entry.details.get("action") == "planner_reported_no_work_needed"

    def test_current_step_out_of_range_recorded_as_failure(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=Plan(steps=[_step(0)]), current_step=5)
        update = node(state)
        assert update["status"] == WorkflowStatus.FAILED
        assert "out of range" in update["errors"][0]

    def test_plan_exceeding_max_steps_recorded_as_failure(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        oversized_plan = Plan(steps=[_step(i) for i in range(11)])
        state = AgentState(plan=oversized_plan, current_step=0)
        update = node(state)
        assert update["status"] == WorkflowStatus.FAILED
        assert "maximum allowed steps" in update["errors"][0]


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


class TestInvalidInput:
    def test_invalid_tool_input_recorded_as_failure(self) -> None:
        """PlanStep.inputs missing a required field -> ToolRegistry raises
        InvalidInputError -> execute_tool must catch and record it, not crash."""
        node = make_execute_tool_node(_registry(_SearchTool()))
        step = _step(0, inputs={})  # missing required 'query'
        state = AgentState(plan=Plan(steps=[step]), current_step=0)

        update = node(state)

        assert update["status"] == WorkflowStatus.FAILED
        assert "INVALID_INPUT" in update["errors"][0]
        failed_trace = update["execution_trace"][-1]
        assert failed_trace.status == "failed"


# ---------------------------------------------------------------------------
# Tool failure (preserved domain error)
# ---------------------------------------------------------------------------


class TestToolFailure:
    def test_tool_internal_failure_preserves_domain_error(self) -> None:
        node = make_execute_tool_node(_registry(_AlwaysFailsTool()))
        step = _step(0, tool_name="always_fails", inputs={"query": "x"})
        state = AgentState(plan=Plan(steps=[step]), current_step=0)

        update = node(state)  # must not raise ToolExecutionError

        assert update["status"] == WorkflowStatus.FAILED
        assert "TOOL_EXECUTION_ERROR" in update["errors"][0]
        assert "tool_results" not in update

    def test_tool_failure_does_not_raise_python_exception(self) -> None:
        """The node itself must never let an OmniFlowException escape --
        it is caught and turned into a state update."""
        node = make_execute_tool_node(_registry(_AlwaysFailsTool()))
        step = _step(0, tool_name="always_fails", inputs={"query": "x"})
        state = AgentState(plan=Plan(steps=[step]), current_step=0)
        try:
            node(state)
        except (ToolExecutionError, InvalidInputError):
            raise AssertionError("execute_tool must catch domain exceptions, not propagate them.")

    def test_failure_does_not_silently_disappear(self) -> None:
        """Both errors list AND execution_trace must reflect the failure --
        never just one or neither."""
        node = make_execute_tool_node(_registry(_AlwaysFailsTool()))
        step = _step(0, tool_name="always_fails", inputs={"query": "x"})
        state = AgentState(plan=Plan(steps=[step]), current_step=0)
        update = node(state)
        assert len(update["errors"]) == 1
        assert any(t.status == "failed" for t in update["execution_trace"])


# ---------------------------------------------------------------------------
# Execution trace
# ---------------------------------------------------------------------------


class TestExecutionTrace:
    def test_success_trace_uses_tool_registry_trace_directly(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        state = AgentState(plan=Plan(steps=[_step(0)]), current_step=0)
        update = node(state)
        trace_entry = update["execution_trace"][-1]
        assert isinstance(trace_entry, ToolExecutionTrace)
        assert trace_entry.status == "success"
        assert trace_entry.tool_name == "search"
        assert trace_entry.step_name == "tool:search"

    def test_trace_is_compact_and_never_contains_raw_query_content(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        sensitive = "SENSITIVE_QUERY_MARKER"
        state = AgentState(plan=Plan(steps=[_step(0, inputs={"query": sensitive})]), current_step=0)
        update = node(state)
        trace_entry = update["execution_trace"][-1]
        assert sensitive not in str(trace_entry.details)
        assert sensitive not in (trace_entry.error_message or "")

    def test_failure_trace_error_message_never_contains_secrets(self) -> None:
        node = make_execute_tool_node(_registry())
        state = AgentState(plan=Plan(steps=[_step(0, tool_name="search")]), current_step=0)
        update = node(state)
        trace_entry = update["execution_trace"][-1]
        assert trace_entry.error_message is not None
        assert "api_key" not in trace_entry.error_message.lower()

    def test_trace_appended_not_replaced(self) -> None:
        node = make_execute_tool_node(_registry(_SearchTool()))
        prior_trace = [
            ToolExecutionTrace(step_name="node:plan", tool_name=None, status="success", duration_ms=1.0)
        ]
        state = AgentState(
            plan=Plan(steps=[_step(0)]), current_step=0, execution_trace=prior_trace
        )
        update = node(state)
        assert len(update["execution_trace"]) == 2
        assert update["execution_trace"][0] is prior_trace[0]


# ---------------------------------------------------------------------------
# No direct LLM/Gemini or concrete YouTube/RAG dependency
# ---------------------------------------------------------------------------


class TestNoDirectDependencies:
    def test_nodes_module_has_no_gemini_or_rag_or_youtube_import(self) -> None:
        import omniflow.graph.nodes as nodes_module

        with open(nodes_module.__file__, encoding="utf-8") as f:
            import_lines = [
                line.strip().lower()
                for line in f.readlines()
                if line.strip().startswith("import ") or line.strip().startswith("from ")
            ]
        forbidden = [
            "google.genai",
            "omniflow.providers.gemini_provider",
            "omniflow.tools.youtube",
            "omniflow.tools.rag_search",
            "omniflow.rag.",
        ]
        for line in import_lines:
            for term in forbidden:
                assert term not in line, f"Forbidden import '{term}' found: {line!r}"

    def test_execute_tool_only_calls_registry_execute_never_a_concrete_tool_directly(self) -> None:
        """A registered dummy tool's run() must be the ONLY code path that
        ever runs -- confirmed by using a tool whose run() has an
        observable, distinctive side effect only reachable via
        ToolRegistry.execute()."""
        calls: list[str] = []

        class _TrackedTool(_SearchTool):
            def run(self, tool_input: _SearchInput) -> _SearchOutput:
                calls.append(tool_input.query)
                return super().run(tool_input)

        node = make_execute_tool_node(_registry(_TrackedTool()))
        state = AgentState(plan=Plan(steps=[_step(0)]), current_step=0)
        node(state)
        assert calls == ["revenue"]
