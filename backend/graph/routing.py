"""Routing functions for conditional edges in the OmniFlow workflow graph.

Each function is a pure, mechanical decision over existing typed AgentState
fields -- bounds/flag checks only, never semantic reasoning about intent,
plans, or tool selection. That reasoning belongs to future planner/intent
nodes; routing functions only decide which already-built edge to take based
on state those future nodes will have set.
"""

from __future__ import annotations

from typing import Literal

from backend.models.state import AgentState, WorkflowStatus

# NOTE: AgentState must be a real (non-TYPE_CHECKING) import here -- unlike
# backend/agents/*, this module is never imported by backend.models.state,
# so there is no circular-import risk. It also cannot be TYPE_CHECKING-only
# regardless: LangGraph's add_conditional_edges() calls typing.get_type_hints()
# on these functions at graph-build time to infer routing schemas, which
# requires AgentState to be resolvable in this module's real runtime namespace.

CheckClarityRoute = Literal["clarification", "plan", "synthesize"]
RouteNextRoute = Literal["plan", "synthesize"]
ValidateOutputRoute = Literal["synthesize", "end"]


def route_after_check_clarity(state: AgentState) -> CheckClarityRoute:
    """Branches to the clarification path if the request was flagged ambiguous.

    Checks ``status == WorkflowStatus.FAILED`` first, mirroring
    ``route_after_route_next``'s identical short-circuit: if
    understand_intent itself failed (e.g. a transient Gemini error that
    exhausted retries), there is no ``clarification_needed`` signal worth
    trusting -- that field was never meaningfully set by a node that didn't
    complete. Proceeding into ``plan``/``execute_tool`` regardless would
    turn one root-cause failure into a stacked cascade of derived failures
    ("cannot plan without prior intent classification", then "no plan
    available to execute"). Going straight to synthesize instead lets that
    single failure produce one clear error in the final response.

    Otherwise, reads the existing ``clarification_needed`` flag as-is; this
    function does not decide ambiguity itself (a future intent/clarity node
    does).
    """
    if state.status == WorkflowStatus.FAILED:
        return "synthesize"
    return "clarification" if state.clarification_needed else "plan"


def route_after_route_next(state: AgentState) -> RouteNextRoute:
    """Decides whether to replan (more work) or proceed to synthesis (complete).

    Three purely mechanical checks, none involving semantic reasoning about
    WHAT to do next -- that reasoning happens inside the ``plan`` node's LLM
    call (backend.agents.planner), not here:

    1. If the previous step execution left ``status == FAILED`` (a domain
       error, or any of the Phase 4.5 bounded limits -- max_agent_steps,
       max_tool_calls, max_retries -- was hit), stop the loop immediately
       and proceed to synthesize. No automatic retry beyond what
       execute_tool's own max_retries check already allows.
    2. If the plan is missing or empty, the planner already decided, on
       this very call, that nothing further is required.
    3. Otherwise, look at the single step ``execute_tool`` just processed
       (index ``current_step - 1`` -- ``observe_result`` already advanced
       ``current_step`` past it). If that step's ``tool_name`` was None,
       ``execute_tool`` performed a no-op: it never touched
       ``tool_call_history``/``tool_results`` (see execute_tool's own
       "no_tool_required" branch), so a replan now would hand the planner
       an execution history IDENTICAL to before this cycle -- nothing new
       happened for it to react to. Left unhandled, this makes a
       legitimate direct-answer plan (planner.py's own documented
       "'Summarize this PDF' ... use a single step with tool_name=null"
       case) loop through replans that keep reproducing the same
       null-tool step, silently burning the whole max_agent_steps budget
       instead of reaching synthesis. A plan with a real tool_name at that
       index (or any other case) still replans as before -- this is
       narrowly scoped to the no-tool no-op, not a change to the general
       "always ask the planner again" policy.
    """
    if state.status == WorkflowStatus.FAILED:
        return "synthesize"

    if state.plan is None or not state.plan.steps:
        return "synthesize"

    last_executed_index = state.current_step - 1
    if 0 <= last_executed_index < len(state.plan.steps):
        last_step = state.plan.steps[last_executed_index]
        if last_step.tool_name is None:
            return "synthesize"

    return "plan"


def route_after_validate_output(state: AgentState) -> ValidateOutputRoute:
    """Routes to exactly one bounded synthesis correction pass, or ends the run.

    Reads the ``validation_issues`` signal exactly as validate_output left
    it -- that node is the one place deciding WHETHER a correction is
    warranted (it already accounts for the bounded attempt budget before
    ever setting a non-empty ``validation_issues``, see nodes.py). This
    function stays purely mechanical: a non-empty list means go back for
    one more synthesize pass; an empty list (either because the output was
    valid, or because the correction budget was already exhausted) means end.
    """
    return "synthesize" if state.validation_issues else "end"
