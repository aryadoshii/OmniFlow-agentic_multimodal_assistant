"""Deterministic tool registry for OmniFlow.

Manages tool registration, introspection, and safe execution with
structured input validation, structured output enforcement, and
timing/error capture in ToolExecutionTrace records.

This module has no knowledge of LLMs, prompts, agents, planning, or
orchestration frameworks. It is a pure, deterministic execution boundary
that a future planner/orchestrator will call into.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pydantic import BaseModel, ValidationError

from omniflow.exceptions import (
    InvalidInputError,
    OmniFlowException,
    ToolAlreadyRegisteredError,
    ToolExecutionError,
    ToolNotFoundError,
)
from omniflow.models.trace import ToolExecutionTrace
from omniflow.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for registering and safely executing deterministic tools.

    Usage::

        registry = ToolRegistry()
        registry.register(SomeTool())
        output, trace = registry.execute("some_tool", field="value")
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """Registers a tool instance by its name.

        Args:
            tool: A concrete BaseTool implementation.

        Raises:
            ToolAlreadyRegisteredError: If a tool with the same name is
                already registered.
        """
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"A tool named '{tool.name}' is already registered. "
                "Unregister it first or use a unique name.",
                details={"tool_name": tool.name},
            )
        self._tools[tool.name] = tool
        logger.debug("Registered tool: '%s'.", tool.name)

    def unregister(self, name: str) -> None:
        """Removes a registered tool by name. No-op if not found."""
        removed = self._tools.pop(name, None)
        if removed:
            logger.debug("Unregistered tool: '%s'.", name)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, name: str) -> BaseTool | None:
        """Returns the tool with the given name, or None if not registered."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, str]]:
        """Returns a description summary of all registered tools.

        Returns:
            List of dicts with ``name`` and ``description`` keys.
        """
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self, tool_name: str, **kwargs: Any
    ) -> tuple[BaseModel, ToolExecutionTrace]:
        """Validates input, executes a registered tool, and captures safe metadata.

        The registry's own parameter is named ``tool_name`` (not ``name``) so
        it cannot collide with a tool's own input field of the same name.

        Args:
            tool_name: Registered tool identifier.
            **kwargs: Keyword arguments used to construct the tool's declared
                      ``input_model``.

        Returns:
            A tuple of (structured_output, ToolExecutionTrace). The trace
            contains only safe metadata: tool name, status, duration, and
            concise string details. Raw secrets, full inputs, or large
            payloads are never stored in the trace.

        Raises:
            ToolNotFoundError: If no tool is registered under ``tool_name``.
            InvalidInputError: If ``kwargs`` fail validation against the
                                tool's declared input model.
            OmniFlowException: Any domain exception a tool raises during
                                execution (e.g. TranscriptUnavailableError,
                                RAGRetrievalError) propagates unmodified --
                                the registry only wraps truly unexpected,
                                non-domain exceptions.
            ToolExecutionError: If the tool raises a non-domain exception, or
                                 returns a non-structured (non-BaseModel) result.
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolNotFoundError(
                f"No tool registered with name '{tool_name}'.",
                details={"available_tools": ", ".join(self._tools.keys()) or "none"},
            )

        try:
            validated_input = tool.input_model(**kwargs)
        except ValidationError as exc:
            raise InvalidInputError(
                f"Invalid input for tool '{tool_name}': {exc}",
                details={"tool": tool_name, "errors": str(exc)},
            ) from exc

        input_field_summary = ", ".join(type(validated_input).model_fields.keys())
        start = time.perf_counter()

        try:
            output = tool.run(validated_input)
        except OmniFlowException as exc:
            # Any well-formed domain exception (ToolExecutionError,
            # TranscriptUnavailableError, RAGRetrievalError, InvalidInputError,
            # etc.) already carries a correct status/error code -- propagate
            # it unmodified rather than flattening it into a generic 500.
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.error(
                "Tool '%s' failed after %.1f ms: %s", tool_name, duration_ms, exc.message
            )
            exc.details.setdefault("duration_ms", str(round(duration_ms, 2)))
            raise
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            sanitized_error = type(exc).__name__
            logger.error(
                "Tool '%s' failed after %.1f ms: %s", tool_name, duration_ms, exc
            )
            raise ToolExecutionError(
                f"Tool '{tool_name}' raised an unexpected error: {sanitized_error}.",
                details={
                    "tool": tool_name,
                    "error_type": sanitized_error,
                    "duration_ms": str(round(duration_ms, 2)),
                },
            ) from exc

        if not isinstance(output, BaseModel):
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned a non-structured result of type "
                f"'{type(output).__name__}'; tools must return a BaseModel instance.",
                details={"tool": tool_name, "returned_type": type(output).__name__},
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        trace = ToolExecutionTrace(
            step_name=f"tool:{tool_name}",
            tool_name=tool_name,
            status="success",
            duration_ms=round(duration_ms, 2),
            details={"input_fields": input_field_summary},
        )
        logger.debug("Tool '%s' succeeded in %.1f ms.", tool_name, duration_ms)
        return output, trace
