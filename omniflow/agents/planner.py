"""Structured planner (Phase 4.3).

Converts a classified IntentResult into a minimum-viable, structured Plan
via BaseLLMProvider's structured-output capability. The planner NEVER
executes a tool and NEVER calls YouTube, RAG, FAISS, or any processor
directly -- it only produces a Plan for a future execution node to consume.

Every non-null tool_name in the returned Plan is validated against the
caller-supplied ToolRegistry's actually-registered tools; a tool name the
registry does not recognize is treated as a planning failure
(OrchestrationError), never silently accepted or executed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from omniflow.agents.intent import IntentResult
from omniflow.exceptions import ExternalProviderError, OrchestrationError
from omniflow.providers.base import BaseLLMProvider
from omniflow.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """A single step in a structured execution plan."""

    step_id: int = Field(..., ge=0, description="Zero-based position of this step in the plan.")
    tool_name: str | None = Field(
        default=None,
        description="Name of a tool registered in ToolRegistry to invoke for this step, "
        "or None if this step requires no tool (e.g. direct synthesis from already-available context).",
    )
    purpose: str = Field(..., description="Concise description of what this step accomplishes.")
    inputs: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured inputs/references this step operates on (e.g. "
        "{'document_id': '...'} or {'query': '...'}). Shape depends on the target tool.",
    )
    expected_result: str = Field(..., description="Concise description of this step's expected outcome.")
    depends_on: int | None = Field(
        default=None,
        description="step_id of the step this one depends on, or None if independent or first.",
    )


class Plan(BaseModel):
    """A minimum-viable, structured sequence of steps to fulfill a request."""

    steps: list[PlanStep] = Field(default_factory=list)


class _PlanStepSchema(BaseModel):
    """Gemini-facing wire schema for a single plan step.

    Identical to PlanStep except ``inputs`` is a JSON-object-encoded
    *string* rather than ``dict[str, Any]``. An open ``dict[str, Any]``
    field compiles to a JSON schema with ``additionalProperties``, which
    the Gemini API's structured-output schema translator rejects outright
    (``ValueError: additionalProperties is not supported in the Gemini
    API.``) -- this is a local, pre-request failure in google-genai's own
    schema-building code, not something any retry or error-handling
    change in GeminiProvider could address. Encoding ``inputs`` as a
    string sidesteps the unsupported schema shape while keeping the LLM's
    actual output (a JSON object) unchanged; it is parsed back into a
    real dict in ``_to_plan`` immediately after generation, so nothing
    outside this module ever sees this intermediate representation.
    """

    step_id: int = Field(..., ge=0)
    tool_name: str | None = Field(default=None)
    purpose: str
    inputs: str = Field(
        default="{}",
        description="A JSON object encoded as a string, e.g. "
        '\'{"document_id": "abc123"}\' or \'{}\' if this step needs no inputs.',
    )
    expected_result: str
    depends_on: int | None = Field(default=None)


class _PlanSchema(BaseModel):
    """Gemini-facing wire schema for Plan -- see _PlanStepSchema."""

    steps: list[_PlanStepSchema] = Field(default_factory=list)


def _to_plan(raw: _PlanSchema) -> Plan:
    """Converts the Gemini wire schema into the real, application-facing Plan.

    Raises:
        ExternalProviderError: If a step's ``inputs`` string is not valid
                                JSON, or does not decode to a JSON object --
                                the same failure category GeminiProvider
                                already raises for unvalidatable structured
                                output.
    """
    steps = []
    for raw_step in raw.steps:
        try:
            inputs = json.loads(raw_step.inputs)
        except json.JSONDecodeError as exc:
            raise ExternalProviderError(
                "Gemini's response could not be validated into Plan.",
                details={
                    "reason": "invalid_structured_output",
                    "response_model": "Plan",
                    "step_id": str(raw_step.step_id),
                },
            ) from exc
        if not isinstance(inputs, dict):
            raise ExternalProviderError(
                "Gemini's response could not be validated into Plan.",
                details={
                    "reason": "invalid_structured_output",
                    "response_model": "Plan",
                    "step_id": str(raw_step.step_id),
                },
            )
        steps.append(
            PlanStep(
                step_id=raw_step.step_id,
                tool_name=raw_step.tool_name,
                purpose=raw_step.purpose,
                inputs=inputs,
                expected_result=raw_step.expected_result,
                depends_on=raw_step.depends_on,
            )
        )
    return Plan(steps=steps)


_PLANNER_SYSTEM_INSTRUCTION = """\
You are the planning component of a deterministic multimodal assistant. \
Given a classified intent and a catalog of tools that ACTUALLY EXIST, \
produce the MINIMUM VIABLE sequence of steps to fulfill the request. You do \
not execute any step yourself -- you only describe the plan.

Rules:
- tool_name MUST be either null, or an exact name from the provided tool \
catalog. NEVER invent a tool name that is not in the catalog.
- Prefer the fewest steps that can satisfy the request. If the request can \
be answered directly from already-ingested document content (e.g. \
"summarize this PDF" when the PDF's full text is already provided), use a \
single step with tool_name=null rather than invoking retrieval or search \
tools unnecessarily.
- Only select a retrieval/search tool when the request genuinely requires \
searching or retrieving evidence beyond what is already directly available \
(e.g. answering a question against a large corpus).
- Only select a transcript-retrieval tool when the request refers to a video \
whose transcript is not already ingested as a document.
- Each step's purpose and expected_result must be concrete and specific to \
this request, not generic placeholders.
- depends_on must reference an earlier step_id, or be null.
- inputs must be a JSON object ENCODED AS A STRING (e.g. '{"document_id": \
"abc123"}' or '{}' if this step needs no inputs) -- not a nested JSON object.

REPLANNING: If "Execution history so far" below is non-empty, you are being \
asked to replan after observing the result(s) of prior tool call(s) -- this \
is NOT the initial plan. Re-evaluate from scratch based on what is now known:
- If the prior result(s) already provide enough information to fulfill the \
request, return an EMPTY steps list (no further action needed before synthesis).
- If a prior tool call returned no evidence, an empty result, or otherwise \
failed to help, do NOT simply repeat the same tool call again -- either try a \
genuinely different tool/approach, or return an empty steps list and accept \
that no more evidence is obtainable (a "no evidence" outcome is a valid, \
final answer -- it is not something to retry into existence).
- If a prior tool call surfaced a new sub-task (e.g. a document mentioned a \
video that must now be transcribed), plan the next concrete step for that, \
not the original assumption.
- Return AT MOST one actionable step (tool_name set) per replan -- you will \
be asked again after it completes, so do not queue up multiple future tool \
calls speculatively.
"""


def _build_planner_prompt(
    intent_result: IntentResult,
    tool_catalog: list[dict[str, str]],
    execution_history: list[dict[str, Any]] | None = None,
) -> str:
    history_section = (
        json.dumps(execution_history, default=str)
        if execution_history
        else "(none -- this is the initial plan for this request)"
    )
    return (
        f"Classified intent: {intent_result.intent.value}\n"
        f"Constraints: {json.dumps(intent_result.constraints)}\n"
        f"Referenced document ids: {json.dumps(intent_result.referenced_inputs)}\n"
        f"Relevant references (URLs/external sources): {json.dumps(intent_result.relevant_references)}\n"
        f"Request explanation: {intent_result.explanation}\n\n"
        f"Available tools (name/description) -- you may ONLY use names from this list:\n"
        f"{json.dumps(tool_catalog)}\n\n"
        f"Execution history so far (tool calls already made this run, in order, with their "
        f"results) -- see the REPLANNING rules above:\n{history_section}"
    )


def _validate_plan_tools(plan: Plan, tool_registry: ToolRegistry) -> None:
    """Raises OrchestrationError if any plan step selects an unregistered tool."""
    for step in plan.steps:
        if step.tool_name is not None and step.tool_name not in tool_registry:
            available = ", ".join(t["name"] for t in tool_registry.list_tools()) or "none"
            raise OrchestrationError(
                f"Planner selected a tool that is not registered: '{step.tool_name}'.",
                details={
                    "tool_name": step.tool_name,
                    "step_id": str(step.step_id),
                    "available_tools": available,
                },
            )


def create_plan(
    intent_result: IntentResult,
    llm_provider: BaseLLMProvider,
    tool_registry: ToolRegistry,
    execution_history: list[dict[str, Any]] | None = None,
) -> Plan:
    """Generates a minimum-viable structured Plan for a classified intent.

    Args:
        intent_result: The IntentResult produced by understand_intent().
        llm_provider: A BaseLLMProvider implementation (e.g. GeminiProvider).
                      This function never imports a concrete provider itself.
        tool_registry: The ToolRegistry whose currently-registered tools
                       bound what the planner may select. Never executed --
                       only introspected via list_tools()/__contains__().
        execution_history: Optional list of ``{"tool": name, "result": ...}``
                            entries describing tool calls already made this
                            run (Phase 4.5 replanning). None/empty means this
                            is the initial plan. The caller (a graph node)
                            assembles this from AgentState -- this function
                            stays decoupled from AgentState's field layout.

    Returns:
        A validated Plan whose every tool_name is confirmed registered.
        When replanning (execution_history is non-empty), an empty
        ``steps`` list is a valid, meaningful result: it signals that no
        further tool action is needed before synthesis.

    Raises:
        ConfigurationError: Propagated as-is from llm_provider if
                             credentials/model configuration are invalid.
        ExternalProviderError: Propagated as-is from llm_provider if the LLM
                                request fails or its output cannot be
                                validated into Plan.
        OrchestrationError: If the plan selects a tool_name that is not
                            actually registered in tool_registry.
    """
    tool_catalog = tool_registry.list_tools()
    prompt = _build_planner_prompt(intent_result, tool_catalog, execution_history)

    logger.info(
        "Creating plan for intent=%s with %d available tool(s) (replanning=%s).",
        intent_result.intent.value,
        len(tool_catalog),
        bool(execution_history),
    )
    raw_plan = llm_provider.generate_structured(
        prompt, _PlanSchema, system_instruction=_PLANNER_SYSTEM_INSTRUCTION
    )
    plan = _to_plan(raw_plan)

    _validate_plan_tools(plan, tool_registry)

    logger.info("Plan created with %d step(s).", len(plan.steps))
    return plan
