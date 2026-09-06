"""LangGraph nodes for the OmniFlow workflow.

Phase 4.2 established the skeleton (honest placeholders). Phase 4.4 made
``execute_tool`` real. Phase 4.5 made ``understand_intent`` and ``plan``
real too, and added bounded multi-step replanning: after every tool
execution, the graph loops back through ``plan`` (a fresh LLM call, aware of
everything executed so far) rather than blindly continuing a stale
upfront plan -- see ``backend.agents.planner``'s REPLANNING rules. Phase
4.7/4.8 make ``prepare_context``, ``synthesize``, and ``validate_output``
real: context aggregation, final-answer synthesis via BaseLLMProvider, and
bounded structural-correction validation.

``check_clarity`` remains an honest placeholder: understand_intent already
set ``clarification_needed``/``clarification_prompt`` from the real
IntentResult, so there is no further decision for this node itself to make
(the branch is taken by route_after_check_clarity).

This module never imports or calls YouTube, RAGService, FAISS, or the
Gemini SDK directly -- only the ToolRegistry/BaseLLMProvider abstractions,
threaded in via factory functions (``make_execute_tool_node``,
``make_understand_intent_node``, ``make_plan_node``, ``make_synthesize_node``).

CRITICAL: bounded execution. Three independent, Settings-driven limits
guarantee the plan/execute/observe loop always terminates regardless of
what the (mocked or real) LLM planner decides:

- ``settings.max_agent_steps``: total plan/execute/observe cycles allowed.
- ``settings.max_tool_calls``: total tool invocations allowed.
- ``settings.max_retries``: max invocations of any ONE tool (loop prevention
  -- e.g. a RAG query that already returned no evidence must not be retried
  forever).

All three are enforced as data (state updates), never raised exceptions --
hitting a limit is a controlled termination, not a crash.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel

from backend import agents as agents_pkg
from backend.config import get_settings
from backend.exceptions import ConfigurationError, OmniFlowException
from backend.models.state import AgentState, WorkflowStatus
from backend.models.trace import ToolExecutionTrace

if TYPE_CHECKING:
    from backend.agents.planner import PlanStep
    from backend.providers.base import BaseLLMProvider
    from backend.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Hard cap on the number of steps a single plan may contain. Protects
# against a single replan producing a runaway plan -- distinct from (and in
# addition to) the cross-cycle settings.max_agent_steps/max_tool_calls
# budgets below, which bound the WHOLE run rather than one plan.
_MAX_PLAN_STEPS = 10

# Maximum synthesize() calls per run: the initial attempt plus at most one
# bounded structural-correction retry (see validate_output). Deliberately a
# small, fixed constant rather than a Settings field -- it is an internal
# safety bound on this node's own retry behavior, not an operator-tunable
# execution budget like max_agent_steps/max_tool_calls/max_retries.
_MAX_SYNTHESIS_ATTEMPTS = 2

# Bounds how much of each normalized document's content prepare_context folds
# into unified_context, so the aggregated context (and, downstream, the
# synthesis prompt built from it) stays bounded regardless of document size.
_MAX_CONTEXT_CHARS_PER_DOCUMENT = 8000

# Bounds the per-document metadata summary folded into unified_context (e.g.
# audio duration_seconds/language, PDF total_pages, image width/height) --
# small compared to content, but present: without this, safe structured
# facts like an audio clip's duration would never reach synthesis at all,
# since synthesize_answer() only ever reads unified_context, never
# normalized_documents directly.
_MAX_METADATA_CHARS_PER_DOCUMENT = 500


def _trace_entry(
    node_name: str,
    duration_ms: float,
    status: str = "success",
    error_message: str | None = None,
    details: dict[str, str] | None = None,
) -> ToolExecutionTrace:
    """Builds a safe trace entry for a node's execution.

    ``tool_name`` is left None to distinguish graph-node events from actual
    tool-execution events (which carry the real tool_name, built directly
    from ToolRegistry.execute()'s own trace). ``details`` must only ever
    carry concise, non-sensitive metadata (counts, flags) -- never raw
    prompts, document content, or model output.
    """
    return ToolExecutionTrace(
        step_name=f"node:{node_name}",
        tool_name=None,
        status=status,
        duration_ms=round(duration_ms, 2),
        details=details or {},
        error_message=error_message,
    )


def _traced_noop(node_name: str, state: AgentState) -> dict[str, Any]:
    """Shared body for placeholder nodes that make no semantic state change."""
    start = time.perf_counter()
    logger.debug("Graph node '%s' starting (session_id=%s).", node_name, state.session_id)
    duration_ms = (time.perf_counter() - start) * 1000.0
    logger.debug("Graph node '%s' completed in %.2fms.", node_name, duration_ms)
    return {"execution_trace": [*state.execution_trace, _trace_entry(node_name, duration_ms)]}


def _fail(
    state: AgentState, node_name: str, message: str, duration_ms: float
) -> dict[str, Any]:
    """Shared body for a node that must stop the run under controlled
    conditions (a limit was hit, or a domain exception was raised and
    caught). Never raises -- the failure is recorded as data so the graph
    can route to a controlled termination rather than crash."""
    logger.warning("Graph node '%s' failed: %s", node_name, message)
    trace_entry = _trace_entry(node_name, duration_ms, status="failed", error_message=message)
    return {
        "execution_trace": [*state.execution_trace, trace_entry],
        "errors": [*state.errors, message],
        "status": WorkflowStatus.FAILED,
    }


def _safe_metadata_summary(metadata: dict[str, Any]) -> str:
    """Renders a document's metadata as bounded, safe JSON for unified_context.

    Metadata shapes are open-ended and processor-specific (audio's
    duration_seconds/language, a PDF's total_pages, an image's
    width/height) -- rather than hand-picking fields per source type, this
    renders the whole dict and truncates, so any safe structured fact a
    processor already extracted stays available to synthesis.
    """
    if not metadata:
        return "{}"
    rendered = json.dumps(metadata, default=str)
    if len(rendered) > _MAX_METADATA_CHARS_PER_DOCUMENT:
        rendered = rendered[:_MAX_METADATA_CHARS_PER_DOCUMENT] + "...(truncated)"
    return rendered


def prepare_context(state: AgentState) -> dict[str, Any]:
    """Aggregates normalized_documents into a single, bounded unified_context string.

    Purely deterministic string aggregation -- no LLM call, no tool
    execution. Includes each document's safe metadata (e.g. an audio clip's
    duration_seconds) alongside its content -- synthesis reads only
    unified_context, never normalized_documents directly, so metadata-only
    facts like duration would otherwise never reach the final answer.
    Consumed by intent understanding's document previews and, more fully,
    by synthesis. A no-op (no unified_context key returned) if there are no
    normalized documents for this request.
    """
    start = time.perf_counter()

    if not state.normalized_documents:
        duration_ms = (time.perf_counter() - start) * 1000.0
        trace_entry = _trace_entry("prepare_context", duration_ms, details={"document_count": "0"})
        return {"execution_trace": [*state.execution_trace, trace_entry]}

    sections = [
        f"[document_id={doc.id} filename={doc.filename} type={doc.source_type.value} "
        f"metadata={_safe_metadata_summary(doc.metadata)}]\n"
        f"{doc.content[:_MAX_CONTEXT_CHARS_PER_DOCUMENT]}"
        for doc in state.normalized_documents
    ]
    unified_context = "\n\n".join(sections)

    duration_ms = (time.perf_counter() - start) * 1000.0
    trace_entry = _trace_entry(
        "prepare_context",
        duration_ms,
        details={"document_count": str(len(state.normalized_documents))},
    )
    return {
        "unified_context": unified_context,
        "execution_trace": [*state.execution_trace, trace_entry],
    }


def make_understand_intent_node(
    llm_provider: BaseLLMProvider,
) -> Callable[[AgentState], dict[str, Any]]:
    """Builds the ``understand_intent`` node bound to a specific BaseLLMProvider.

    Calls ``backend.agents.intent.understand_intent()`` (structured LLM
    output; never a raw Gemini call) and maps the returned IntentResult onto
    AgentState's fields. Does not decide what to DO about ambiguity -- it
    only records the signal (``clarification_needed``/``clarification_prompt``);
    the clarification node itself remains a placeholder (Phase 4.6).
    """

    def understand_intent(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()
        if llm_provider is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            missing = ConfigurationError("No llm_provider was configured for this graph.")
            return _fail(
                state, "understand_intent", f"{missing.error_code}: {missing.message}", duration_ms
            )
        try:
            result = agents_pkg.understand_intent(state, llm_provider)
        except OmniFlowException as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state, "understand_intent", f"{exc.error_code}: {exc.message}", duration_ms
            )

        duration_ms = (time.perf_counter() - start) * 1000.0
        trace_entry = _trace_entry("understand_intent", duration_ms)
        return {
            "intent_result": result,
            "detected_intent": result.intent.value,
            "constraints": result.constraints,
            "references": [*result.referenced_inputs, *result.relevant_references],
            "clarification_needed": result.needs_clarification,
            "clarification_prompt": result.clarification_question,
            "execution_trace": [*state.execution_trace, trace_entry],
        }

    return understand_intent


def check_clarity(state: AgentState) -> dict[str, Any]:
    """Placeholder node preceding the clarification-vs-plan routing decision.

    understand_intent (Phase 4.5) already sets clarification_needed from the
    real IntentResult; this node itself makes no further decision -- the
    actual branch is taken by route_after_check_clarity.
    """
    return _traced_noop("check_clarity", state)


def clarification(state: AgentState) -> dict[str, Any]:
    """Terminal node for the clarification branch (Phase 4.6).

    The clarification question itself was already produced by
    understand_intent (IntentResult.clarification_question, mapped onto
    state.clarification_prompt) -- this node does not generate or guess a
    question of its own. It only records that the run stopped here (the
    graph never reaches plan/execute_tool on this branch -- see
    route_after_check_clarity) and marks the workflow
    AWAITING_CLARIFICATION so a caller can surface the question and hold
    onto the full AgentState for a later turn. Actually resuming after a
    user's clarifying reply (continuing the same run with new information)
    is not implemented -- that requires persistent multi-turn state, which
    is explicitly out of scope for this phase.
    """
    start = time.perf_counter()
    duration_ms = (time.perf_counter() - start) * 1000.0
    trace_entry = _trace_entry(
        "clarification",
        duration_ms,
        details={"has_question": str(bool(state.clarification_prompt))},
    )
    return {
        "execution_trace": [*state.execution_trace, trace_entry],
        "status": WorkflowStatus.AWAITING_CLARIFICATION,
    }


def _build_execution_history(state: AgentState) -> list[dict[str, Any]]:
    """Assembles the replanning context from AgentState's own history fields.

    Kept here (not in agents/planner.py) so the planner stays decoupled from
    AgentState's specific field layout -- it only ever sees a plain list of
    {"tool": ..., "result": ...} dicts.
    """
    history: list[dict[str, Any]] = []
    for tool_name, result in zip(state.tool_call_history, state.tool_results.values()):
        serialized = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        history.append({"tool": tool_name, "result": serialized})
    return history


def make_plan_node(
    llm_provider: BaseLLMProvider, tool_registry: ToolRegistry
) -> Callable[[AgentState], dict[str, Any]]:
    """Builds the ``plan``/replan node bound to a BaseLLMProvider and ToolRegistry.

    Called both for the INITIAL plan and every REPLAN (the graph loops back
    here after each execute_tool/observe_result cycle -- see routing.py).
    Each call is a fresh ``create_plan()`` invocation aware of everything
    executed so far, so the graph never blindly continues a plan whose
    assumptions a tool result may have invalidated.

    Enforces ``settings.max_agent_steps`` here (before generating a plan the
    run has no budget left for) -- the hard ceiling guaranteeing the
    plan/execute/observe loop always terminates regardless of what the
    planner decides.
    """

    def plan(state: AgentState) -> dict[str, Any]:
        settings = get_settings()
        start = time.perf_counter()
        new_step_count = state.agent_step_count + 1

        if llm_provider is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            missing = ConfigurationError("No llm_provider was configured for this graph.")
            update = _fail(state, "plan", f"{missing.error_code}: {missing.message}", duration_ms)
            update["agent_step_count"] = new_step_count
            return update

        if new_step_count > settings.max_agent_steps:
            duration_ms = (time.perf_counter() - start) * 1000.0
            update = _fail(
                state,
                "plan",
                f"Maximum agent steps ({settings.max_agent_steps}) exceeded.",
                duration_ms,
            )
            update["agent_step_count"] = new_step_count
            return update

        if state.intent_result is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            update = _fail(
                state, "plan", "Cannot plan without a prior intent classification.", duration_ms
            )
            update["agent_step_count"] = new_step_count
            return update

        execution_history = _build_execution_history(state)

        try:
            new_plan = agents_pkg.create_plan(
                state.intent_result,
                llm_provider,
                tool_registry,
                execution_history=execution_history,
            )
        except OmniFlowException as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            update = _fail(state, "plan", f"{exc.error_code}: {exc.message}", duration_ms)
            update["agent_step_count"] = new_step_count
            return update

        duration_ms = (time.perf_counter() - start) * 1000.0
        trace_entry = _trace_entry("plan", duration_ms)
        return {
            "plan": new_plan,
            "current_step": 0,
            "agent_step_count": new_step_count,
            "execution_trace": [*state.execution_trace, trace_entry],
        }

    return plan


def make_execute_tool_node(tool_registry: ToolRegistry) -> Callable[[AgentState], dict[str, Any]]:
    """Builds the ``execute_tool`` node bound to a specific ToolRegistry.

    The graph (via the ``plan`` node) decides WHICH tool to run; ToolRegistry
    decides HOW it runs. This node never imports or calls YouTube,
    RAGService, FAISS, or the Gemini SDK -- only ``tool_registry.execute()``.

    An EMPTY plan (``plan.steps == []``) is a deliberate, non-error signal
    from the planner that no further tool action is needed this cycle --
    distinct from ``plan is None`` (genuinely missing, an error). Loop
    prevention (``settings.max_tool_calls``/``settings.max_retries``) is
    enforced here, immediately before any tool would actually be invoked.
    """

    def execute_tool(state: AgentState) -> dict[str, Any]:
        settings = get_settings()
        start = time.perf_counter()

        if state.plan is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(state, "execute_tool", "No plan is available to execute.", duration_ms)

        if not state.plan.steps:
            # The planner explicitly decided nothing needs to run this cycle
            # -- a benign "no work to do", not a failure.
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace_entry = ToolExecutionTrace(
                step_name="node:execute_tool",
                tool_name=None,
                status="success",
                duration_ms=round(duration_ms, 2),
                details={"action": "planner_reported_no_work_needed"},
            )
            return {"execution_trace": [*state.execution_trace, trace_entry]}

        if len(state.plan.steps) > _MAX_PLAN_STEPS:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state,
                "execute_tool",
                f"Plan exceeds the maximum allowed steps "
                f"({len(state.plan.steps)} > {_MAX_PLAN_STEPS}).",
                duration_ms,
            )

        if state.current_step < 0 or state.current_step >= len(state.plan.steps):
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state,
                "execute_tool",
                f"current_step ({state.current_step}) is out of range for a "
                f"plan with {len(state.plan.steps)} step(s).",
                duration_ms,
            )

        step: PlanStep = state.plan.steps[state.current_step]

        if step.tool_name is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            trace_entry = ToolExecutionTrace(
                step_name="node:execute_tool",
                tool_name=None,
                status="success",
                duration_ms=round(duration_ms, 2),
                details={"step_id": str(step.step_id), "action": "no_tool_required"},
            )
            return {"execution_trace": [*state.execution_trace, trace_entry]}

        if step.tool_name not in tool_registry:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state,
                "execute_tool",
                f"Plan step {step.step_id} references a tool that is not "
                f"registered: '{step.tool_name}'.",
                duration_ms,
            )

        if len(state.tool_call_history) >= settings.max_tool_calls:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state,
                "execute_tool",
                f"Maximum tool calls ({settings.max_tool_calls}) exceeded.",
                duration_ms,
            )

        prior_calls_to_this_tool = state.tool_call_history.count(step.tool_name)
        if prior_calls_to_this_tool >= settings.max_retries:
            duration_ms = (time.perf_counter() - start) * 1000.0
            return _fail(
                state,
                "execute_tool",
                f"Tool '{step.tool_name}' has already been called "
                f"{prior_calls_to_this_tool} time(s) (max_retries="
                f"{settings.max_retries}); refusing to call it again to "
                f"prevent a loop.",
                duration_ms,
            )

        try:
            output, tool_trace = tool_registry.execute(step.tool_name, **step.inputs)
        except OmniFlowException as exc:
            # Preserve the domain error, record it safely in state, and
            # stop -- no automatic retry beyond the max_retries budget
            # above, no silent discard.
            duration_ms = (time.perf_counter() - start) * 1000.0
            safe_message = f"{exc.error_code}: {exc.message}"
            logger.error(
                "Tool '%s' failed for step %d after %.1fms: %s",
                step.tool_name,
                step.step_id,
                duration_ms,
                exc.error_code,
            )
            trace_entry = ToolExecutionTrace(
                step_name="node:execute_tool",
                tool_name=step.tool_name,
                status="failed",
                duration_ms=round(duration_ms, 2),
                details={"step_id": str(step.step_id)},
                error_message=safe_message,
            )
            return {
                "execution_trace": [*state.execution_trace, trace_entry],
                "errors": [*state.errors, f"Step {step.step_id} ({step.tool_name}) failed: {safe_message}"],
                "status": WorkflowStatus.FAILED,
                "tool_call_history": [*state.tool_call_history, step.tool_name],
            }

        # Keyed by agent_step_count (not step_id): every replan resets
        # step_id back to 0, which would silently collide/overwrite results
        # across cycles if used as the key instead.
        updated_results = {**state.tool_results, str(state.agent_step_count): output}
        return {
            "tool_results": updated_results,
            "tool_call_history": [*state.tool_call_history, step.tool_name],
            "execution_trace": [*state.execution_trace, tool_trace],
        }

    return execute_tool


def observe_result(state: AgentState) -> dict[str, Any]:
    """Placeholder: advances current_step so execute_tool's bounds check
    reflects the step just completed.

    Advancing the step counter is mechanical bookkeeping, not an
    observation/synthesis decision -- it does not inspect tool_results or
    retrieved_evidence content. The actual "should we replan or finish"
    decision is made by route_after_route_next, informed by whether the
    plan just executed was empty (see routing.py).
    """
    update = _traced_noop("observe_result", state)
    update["current_step"] = state.current_step + 1
    return update


def route_next(state: AgentState) -> dict[str, Any]:
    """Placeholder node preceding the replan-vs-complete routing decision.

    The actual decision is made by route_after_route_next; this node
    itself makes no semantic change.
    """
    return _traced_noop("route_next", state)


def make_synthesize_node(llm_provider: BaseLLMProvider) -> Callable[[AgentState], dict[str, Any]]:
    """Builds the ``synthesize`` node bound to a specific BaseLLMProvider.

    Calls ``backend.agents.synthesizer.synthesize_answer()`` (a single
    plain-text LLM call); never calls RAGService/FAISS/YouTube/the Gemini
    SDK directly. Runs once for the initial attempt, and -- only if
    validate_output found a structural violation and the bounded attempt
    budget allows it (see _MAX_SYNTHESIS_ATTEMPTS) -- exactly one corrective
    revision pass, using ``state.validation_issues`` as feedback.

    Runs regardless of a prior FAILED status (e.g. a tool or planning
    failure short-circuited the loop -- see route_after_route_next) so the
    run still produces a best-effort, honest final_answer grounded in
    whatever was gathered, rather than leaving the user with nothing.
    """

    def synthesize(state: AgentState) -> dict[str, Any]:
        start = time.perf_counter()

        if llm_provider is None:
            duration_ms = (time.perf_counter() - start) * 1000.0
            missing = ConfigurationError("No llm_provider was configured for this graph.")
            update = _fail(
                state, "synthesize", f"{missing.error_code}: {missing.message}", duration_ms
            )
            update["synthesis_attempts"] = state.synthesis_attempts + 1
            return update

        correction_feedback = "; ".join(state.validation_issues) if state.validation_issues else None

        try:
            answer = agents_pkg.synthesize_answer(
                state, llm_provider, correction_feedback=correction_feedback
            )
        except OmniFlowException as exc:
            duration_ms = (time.perf_counter() - start) * 1000.0
            update = _fail(state, "synthesize", f"{exc.error_code}: {exc.message}", duration_ms)
            update["synthesis_attempts"] = state.synthesis_attempts + 1
            return update

        duration_ms = (time.perf_counter() - start) * 1000.0
        trace_entry = _trace_entry(
            "synthesize",
            duration_ms,
            details={
                "attempt": str(state.synthesis_attempts + 1),
                "correction": str(bool(correction_feedback)),
            },
        )
        return {
            "final_answer": answer,
            "synthesis_attempts": state.synthesis_attempts + 1,
            "validation_issues": [],
            "execution_trace": [*state.execution_trace, trace_entry],
        }

    return synthesize


def validate_output(state: AgentState) -> dict[str, Any]:
    """Performs deterministic structural validation of ``final_answer``.

    Never re-validates once a prior FAILED status is already recorded --
    that status must never be silently overwritten back to COMPLETED, and
    a failed run gets no correction pass (there is nothing sound to
    correct). Otherwise runs ``backend.agents.validator.validate_structure``
    (non-empty, plain text, requested bullet counts, required sections --
    never semantic correctness). When a violation is found and the bounded
    synthesis-attempt budget (_MAX_SYNTHESIS_ATTEMPTS) still allows it,
    records the violations onto ``validation_issues`` for
    route_after_validate_output to send the run back to ``synthesize`` for
    exactly one corrective pass; otherwise finalizes as COMPLETED
    regardless of outcome, so the loop always terminates.
    """
    start = time.perf_counter()

    if state.status == WorkflowStatus.FAILED:
        duration_ms = (time.perf_counter() - start) * 1000.0
        trace_entry = _trace_entry(
            "validate_output", duration_ms, details={"reason": "prior_failure_preserved"}
        )
        return {"execution_trace": [*state.execution_trace, trace_entry]}

    constraints = state.intent_result.constraints if state.intent_result else state.constraints
    result = agents_pkg.validate_structure(state.final_answer, constraints)
    duration_ms = (time.perf_counter() - start) * 1000.0

    if result.is_valid or state.synthesis_attempts >= _MAX_SYNTHESIS_ATTEMPTS:
        trace_entry = _trace_entry(
            "validate_output",
            duration_ms,
            details={"valid": str(result.is_valid), "issue_count": str(len(result.violations))},
        )
        return {
            "status": WorkflowStatus.COMPLETED,
            "validation_issues": [],
            "execution_trace": [*state.execution_trace, trace_entry],
        }

    trace_entry = _trace_entry(
        "validate_output",
        duration_ms,
        details={
            "valid": "False",
            "issue_count": str(len(result.violations)),
            "action": "requesting_bounded_correction",
        },
    )
    return {
        "validation_issues": result.violations,
        "execution_trace": [*state.execution_trace, trace_entry],
    }
