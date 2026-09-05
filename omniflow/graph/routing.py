"""Routing functions for conditional edges in the OmniFlow workflow graph.

Each function is a pure, mechanical decision over existing typed AgentState
fields -- bounds/flag checks only, never semantic reasoning about intent,
plans, or tool selection. That reasoning belongs to future planner/intent
nodes; routing functions only decide which already-built edge to take based
on state those future nodes will have set.
"""

from __future__ import annotations

from typing import Literal

from omniflow.models.state import AgentState, WorkflowStatus

# NOTE: AgentState must be a real (non-TYPE_CHECKING) import here -- unlike
# omniflow/agents/*, this module is never imported by omniflow.models.state,
# so there is no circular-import risk. It also cannot be TYPE_CHECKING-only
# regardless: LangGraph's add_conditional_edges() calls typing.get_type_hints()
# on these functions at graph-build time to infer routing schemas, which
# requires AgentState to be resolvable in this module's real runtime namespace.

CheckClarityRoute = Literal["clarification", "plan"]
RouteNextRoute = Literal["plan", "synthesize"]
ValidateOutputRoute = Literal["synthesize", "end"]


def route_after_check_clarity(state: AgentState) -> CheckClarityRoute:
    """Branches to the clarification path if the request was flagged ambiguous.

    Reads the existing ``clarification_needed`` flag as-is; this function
    does not decide ambiguity itself (a future intent/clarity node does).
    """
    return "clarification" if state.clarification_needed else "plan"


def route_after_route_next(state: AgentState) -> RouteNextRoute:
    """Decides whether to replan (more work) or proceed to synthesis (complete).

    Two purely mechanical checks, neither involving semantic reasoning about
    WHAT to do next -- that reasoning happens inside the ``plan`` node's LLM
    call (omniflow.agents.planner), not here:

    1. If the previous step execution left ``status == FAILED`` (a domain
       error, or any of the Phase 4.5 bounded limits -- max_agent_steps,
       max_tool_calls, max_retries -- was hit), stop the loop immediately
       and proceed to synthesize. No automatic retry beyond what
       execute_tool's own max_retries check already allows.
    2. Otherwise: the plan that was just executed (``state.plan``) is
       inspected, NOT re-derived from step position. Phase 4.5 replans after
       every single execute_tool/observe_result cycle (see the ``plan``
       node), so the planner's own output IS the "more work needed" signal:
       a non-empty plan means at least one action was just taken and the
       planner should be asked again (aware of the new result) whether more
       is needed; an EMPTY plan means the planner already decided, on this
       very call, that nothing further is required.
    """
    if state.status == WorkflowStatus.FAILED:
        return "synthesize"

    plan_was_empty = state.plan is None or not state.plan.steps
    return "synthesize" if plan_was_empty else "plan"


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
