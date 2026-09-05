"""LangGraph workflow builder for OmniFlow.

Establishes the graph structure and state contract, and wires every node to
its real implementation in omniflow/graph/nodes.py -- intent understanding,
planning, tool execution, replanning, synthesis, and output validation are
all real as of Phase 4.5/4.7/4.8. ``check_clarity`` is the one remaining
honest placeholder: understand_intent already set the clarification signal,
so there is no further decision for that node itself to make. This builder
threads ToolRegistry/BaseLLMProvider through via factory functions -- it
never imports RAGService, GeminiProvider, the YouTube implementation,
FAISS, or any processor directly.

Graph shape::

    START -> prepare_context -> understand_intent -> check_clarity
                                                          |
                                    +---------------------+---------------------+
                                    v                                           v
                              clarification                                   plan
                                    |                                           |
                                   END                                    execute_tool
                                                                                |
                                                                         observe_result
                                                                                |
                                                                           route_next
                                                                                |
                                                      +-------------------------+-------------------------+
                                                      v (more work)                                       v (complete)
                                                    plan                                            synthesize
                                                                                                          |
                                                                                                  validate_output
                                                                                                          |
                                                                                   +----------------------+----------------------+
                                                                                   v (structural violation, budget allows)        v (valid, or budget exhausted)
                                                                                synthesize                                      END

This module owns workflow control flow only. It imports ToolRegistry solely
to thread it through to execute_tool -- it never imports RAGService,
GeminiProvider, the YouTube implementation, FAISS, or any processor; those
remain the responsibility of the layers that implement real tool/node logic.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from omniflow.exceptions import OmniFlowException, OrchestrationError
from omniflow.graph import nodes
from omniflow.graph.routing import (
    route_after_check_clarity,
    route_after_route_next,
    route_after_validate_output,
)
from omniflow.models.state import AgentState
from omniflow.providers.base import BaseLLMProvider
from omniflow.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def build_graph(
    tool_registry: ToolRegistry | None = None,
    llm_provider: BaseLLMProvider | None = None,
) -> CompiledStateGraph:
    """Constructs and compiles the OmniFlow workflow graph.

    Args:
        tool_registry: The registry ``execute_tool`` will dispatch to. The
                        graph decides WHICH tool to run (by reading the
                        current plan step); this registry decides HOW it
                        runs. Defaults to a fresh, empty ToolRegistry if not
                        provided -- meaning any plan step naming a tool will
                        correctly fail validation rather than silently
                        succeeding, since no tool is actually available.
        llm_provider: The BaseLLMProvider ``understand_intent``/``plan``
                      dispatch to for structured LLM calls (Phase 4.5). Must
                      be provided to run a graph past ``understand_intent``
                      -- there is no default construction here (unlike
                      ToolRegistry's empty default) since a provider
                      requires real configuration (API key, etc.) this
                      builder should never guess at silently.

    Returns a fresh, stateless CompiledStateGraph -- it holds no per-request
    state itself; each ``.invoke(initial_state)`` call operates on its own
    independent AgentState, so concurrent requests never share or leak state.
    Safe to call multiple times (e.g. once per test) with no side effects.
    """
    tool_registry = tool_registry if tool_registry is not None else ToolRegistry()
    builder = StateGraph(AgentState)

    builder.add_node("prepare_context", nodes.prepare_context)
    builder.add_node("understand_intent", nodes.make_understand_intent_node(llm_provider))
    builder.add_node("check_clarity", nodes.check_clarity)
    builder.add_node("clarification", nodes.clarification)
    builder.add_node("plan", nodes.make_plan_node(llm_provider, tool_registry))
    builder.add_node("execute_tool", nodes.make_execute_tool_node(tool_registry))
    builder.add_node("observe_result", nodes.observe_result)
    builder.add_node("route_next", nodes.route_next)
    builder.add_node("synthesize", nodes.make_synthesize_node(llm_provider))
    builder.add_node("validate_output", nodes.validate_output)

    builder.add_edge(START, "prepare_context")
    builder.add_edge("prepare_context", "understand_intent")
    builder.add_edge("understand_intent", "check_clarity")
    builder.add_conditional_edges(
        "check_clarity",
        route_after_check_clarity,
        {"clarification": "clarification", "plan": "plan"},
    )
    builder.add_edge("clarification", END)
    builder.add_edge("plan", "execute_tool")
    builder.add_edge("execute_tool", "observe_result")
    builder.add_edge("observe_result", "route_next")
    builder.add_conditional_edges(
        "route_next",
        route_after_route_next,
        {"plan": "plan", "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", "validate_output")
    builder.add_conditional_edges(
        "validate_output",
        route_after_validate_output,
        {"synthesize": "synthesize", "end": END},
    )

    return builder.compile()


def run_graph(
    initial_state: AgentState,
    tool_registry: ToolRegistry | None = None,
    llm_provider: BaseLLMProvider | None = None,
    compiled_graph: CompiledStateGraph | None = None,
) -> AgentState:
    """Executes the workflow graph and returns the resulting AgentState.

    This is the error boundary between graph execution and callers: any
    exception a node raises that is already a well-formed OmniFlowException
    propagates unmodified -- notably, an ordinary tool/LLM-call failure
    inside execute_tool/understand_intent/plan is caught by that node itself
    (see nodes.py) and recorded into state as a controlled FAILED status, so
    it does NOT reach this boundary as a raised exception; only genuinely
    unexpected node-implementation bugs do. Any such unexpected exception is
    wrapped in OrchestrationError so it never escapes without context --
    without this builder ever needing to know about provider-specific
    exception types itself.

    Args:
        initial_state: A fresh, request-scoped AgentState. Never a shared or
                        module-level instance.
        tool_registry: Passed to build_graph() when compiled_graph is not
                       supplied; ignored if compiled_graph is provided (that
                       graph's registry was already bound at build time).
        llm_provider: Passed to build_graph() when compiled_graph is not
                      supplied; ignored if compiled_graph is provided.
        compiled_graph: Optional pre-built graph (e.g. for reuse across
                        calls in a long-lived process); defaults to building
                        a new one via build_graph(tool_registry, llm_provider).

    Returns:
        The final AgentState after the graph reaches END.

    Raises:
        OmniFlowException: Propagated unmodified from node execution.
        OrchestrationError: Wraps any other unexpected exception.
    """
    graph = compiled_graph or build_graph(tool_registry, llm_provider)
    try:
        result = graph.invoke(initial_state)
    except OmniFlowException:
        raise
    except Exception as exc:
        logger.error("Graph execution failed unexpectedly: %s", type(exc).__name__)
        raise OrchestrationError(
            f"Graph execution failed: {type(exc).__name__}.",
            details={"error_type": type(exc).__name__},
        ) from exc

    return AgentState.model_validate(result)
