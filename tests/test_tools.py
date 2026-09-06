"""Tests for the deterministic Tool Framework and Tool Registry (Phase 3.1).

Uses small local dummy tools to exercise the BaseTool/ToolRegistry contract
in isolation, independent of any concrete tool implementation (YouTube, RAG,
etc.) which are out of scope for this phase.
"""

import pytest
from pydantic import BaseModel

from backend.exceptions import (
    InvalidInputError,
    ToolAlreadyRegisteredError,
    ToolExecutionError,
    ToolNotFoundError,
)
from backend.models.trace import ToolExecutionTrace
from backend.tools.base import BaseTool
from backend.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Dummy tools for testing
# ---------------------------------------------------------------------------


class GreetInput(BaseModel):
    name: str


class GreetOutput(BaseModel):
    message: str


class GreetTool(BaseTool):
    """A trivial deterministic tool used to exercise the happy path."""

    @property
    def name(self) -> str:
        return "greet"

    @property
    def description(self) -> str:
        return "Greets a person by name."

    @property
    def input_model(self) -> type[BaseModel]:
        return GreetInput

    def run(self, tool_input: GreetInput) -> GreetOutput:
        return GreetOutput(message=f"Hello, {tool_input.name}!")


class FailingInput(BaseModel):
    value: int


class FailingTool(BaseTool):
    """A tool that always raises an unexpected internal error."""

    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails, for testing error handling."

    @property
    def input_model(self) -> type[BaseModel]:
        return FailingInput

    def run(self, tool_input: FailingInput) -> BaseModel:
        raise RuntimeError("Simulated internal failure.")


class DeclaredFailureTool(BaseTool):
    """A tool that raises its own ToolExecutionError deliberately."""

    @property
    def name(self) -> str:
        return "declared_failure_tool"

    @property
    def description(self) -> str:
        return "A tool that raises ToolExecutionError directly."

    @property
    def input_model(self) -> type[BaseModel]:
        return FailingInput

    def run(self, tool_input: FailingInput) -> BaseModel:
        raise ToolExecutionError(
            "Deliberate declared failure.", details={"tool": self.name}
        )


class BadOutputInput(BaseModel):
    pass


class BadOutputTool(BaseTool):
    """A misbehaving tool that returns a raw dict instead of a BaseModel."""

    @property
    def name(self) -> str:
        return "bad_output_tool"

    @property
    def description(self) -> str:
        return "A tool that violates the structured-output contract."

    @property
    def input_model(self) -> type[BaseModel]:
        return BadOutputInput

    def run(self, tool_input: BadOutputInput):
        return {"not": "a BaseModel"}


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------


class TestBaseTool:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseTool()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_success(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())
        assert "greet" in registry
        assert len(registry) == 1

    def test_register_duplicate_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())
        with pytest.raises(ToolAlreadyRegisteredError) as exc_info:
            registry.register(GreetTool())
        assert exc_info.value.error_code == "TOOL_ALREADY_REGISTERED"
        assert len(registry) == 1

    def test_unregister_removes_tool(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())
        registry.unregister("greet")
        assert "greet" not in registry
        assert len(registry) == 0

    def test_unregister_missing_tool_is_noop(self) -> None:
        registry = ToolRegistry()
        registry.unregister("does_not_exist")
        assert len(registry) == 0


# ---------------------------------------------------------------------------
# Lookup / Listing
# ---------------------------------------------------------------------------


class TestLookupAndListing:
    def test_get_registered_tool(self) -> None:
        registry = ToolRegistry()
        tool = GreetTool()
        registry.register(tool)
        assert registry.get("greet") is tool

    def test_get_missing_tool_returns_none(self) -> None:
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())
        registry.register(FailingTool())
        listing = registry.list_tools()
        names = {entry["name"] for entry in listing}
        assert names == {"greet", "failing_tool"}
        for entry in listing:
            assert set(entry.keys()) == {"name", "description"}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_execute_success_returns_structured_output_and_trace(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())

        output, trace = registry.execute("greet", name="Ada")

        assert isinstance(output, GreetOutput)
        assert output.message == "Hello, Ada!"
        assert isinstance(trace, ToolExecutionTrace)
        assert trace.status == "success"
        assert trace.tool_name == "greet"
        assert trace.duration_ms is not None

    def test_execute_missing_tool_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.execute("nonexistent", foo="bar")
        assert exc_info.value.error_code == "TOOL_NOT_FOUND"
        assert exc_info.value.status_code == 404

    def test_execute_invalid_input_raises_invalid_input_error(self) -> None:
        registry = ToolRegistry()
        registry.register(GreetTool())
        with pytest.raises(InvalidInputError) as exc_info:
            registry.execute("greet")  # missing required 'name' field
        assert exc_info.value.status_code == 400

    def test_execute_wrong_type_input_raises_invalid_input_error(self) -> None:
        registry = ToolRegistry()
        registry.register(FailingTool())
        with pytest.raises(InvalidInputError):
            registry.execute("failing_tool", value="not_an_int")

    def test_execute_unexpected_internal_error_wraps_as_tool_execution_error(
        self,
    ) -> None:
        registry = ToolRegistry()
        registry.register(FailingTool())
        with pytest.raises(ToolExecutionError) as exc_info:
            registry.execute("failing_tool", value=1)
        assert exc_info.value.error_code == "TOOL_EXECUTION_ERROR"
        assert "duration_ms" in exc_info.value.details

    def test_execute_declared_tool_execution_error_propagates(self) -> None:
        registry = ToolRegistry()
        registry.register(DeclaredFailureTool())
        with pytest.raises(ToolExecutionError) as exc_info:
            registry.execute("declared_failure_tool", value=1)
        assert exc_info.value.message == "Deliberate declared failure."
        assert "duration_ms" in exc_info.value.details

    def test_execute_enforces_structured_output(self) -> None:
        registry = ToolRegistry()
        registry.register(BadOutputTool())
        with pytest.raises(ToolExecutionError) as exc_info:
            registry.execute("bad_output_tool")
        assert "non-structured" in exc_info.value.message

    def test_registry_has_no_orchestration_imports(self) -> None:
        """The registry module must not import any LLM/orchestration library."""
        import backend.tools.registry as registry_module

        source = registry_module.__file__
        with open(source, encoding="utf-8") as f:
            content = f.read().lower()
        forbidden_imports = ["langgraph", "openai", "google.generativeai", "langchain"]
        for term in forbidden_imports:
            assert term not in content, f"Forbidden dependency '{term}' referenced in registry module."
