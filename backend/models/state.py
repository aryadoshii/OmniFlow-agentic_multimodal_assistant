"""Agent state model used as the LangGraph workflow state schema.

This is the single, shared state container passed between graph nodes
(Phase 4.2+). It is never a global/singleton -- each request constructs its
own fresh AgentState instance, and the compiled graph passes it through
nodes as an explicit value, never mutating shared process state.
"""

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from backend.agents.intent import IntentResult
from backend.agents.planner import Plan
from backend.models.document import NormalizedDocument
from backend.models.request import UploadedInput
from backend.models.trace import ToolExecutionTrace
from backend.rag.service import RAGResult


class WorkflowStatus(str, Enum):
    """Overall completion status of an agent workflow run."""

    IN_PROGRESS = "in_progress"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    """Unified state container shared across LangGraph workflow nodes.

    Includes typed containers for the full agent lifecycle: ingestion, intent
    classification, clarification checks, planning, tool execution, RAG retrieval,
    and final synthesis.
    """

    # Request and Ingestion inputs
    original_request: str = Field(
        default="",
        description="The raw prompt or instruction submitted by the user.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session or conversation tracking ID.",
    )
    uploaded_inputs: list[UploadedInput] = Field(
        default_factory=list,
        description="Metadata of raw uploads before processing.",
    )
    normalized_documents: list[NormalizedDocument] = Field(
        default_factory=list,
        description="Normalized extracted content from all input modalities.",
    )
    unified_context: str = Field(
        default="",
        description="Aggregated and aligned context generated from normalized documents.",
    )

    # Intent and Constraints
    intent_result: IntentResult | None = Field(
        default=None,
        description="Full structured result from backend.agents.intent.understand_intent(), "
        "consumed by the planner. None until intent understanding has run for this request. "
        "detected_intent/constraints/references/clarification_* below are convenience views "
        "derived from this same object, kept for existing routing/consumer compatibility.",
    )
    detected_intent: str | None = Field(
        default=None,
        description="Classified semantic intent of the user request.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Extracted constraints or boundary conditions for the response.",
    )
    references: list[str] = Field(
        default_factory=list,
        description="Extracted entities, topics, or reference markers.",
    )

    # Clarification State
    clarification_needed: bool = Field(
        default=False,
        description="Flag indicating if the user request is ambiguous or underspecified.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Clarifying question posed back to the user when ambiguous.",
    )

    # Planning & Tool Execution
    plan: Plan | None = Field(
        default=None,
        description="Structured plan produced by the planner (see backend.agents.planner.Plan). "
        "None if planning has not occurred yet for this request.",
    )
    current_step: int = Field(
        default=0,
        ge=0,
        description="Zero-based index into plan.steps identifying the step currently being "
        "executed. Reset to 0 each time the planner produces a fresh plan (Phase 4.5 replans "
        "after every execute/observe cycle, so a plan's own step_id is not unique across the "
        "whole run -- see tool_results/tool_call_history below).",
    )
    agent_step_count: int = Field(
        default=0,
        ge=0,
        description="Total number of plan/execute/observe cycles completed so far in this "
        "run, incremented once per planning call regardless of plan size. Bounded by "
        "settings.max_agent_steps -- the hard ceiling guaranteeing the agent loop terminates.",
    )
    tool_call_history: list[str] = Field(
        default_factory=list,
        description="Tool names in the order they were actually invoked this run (one entry "
        "per successful or failed tool call, never for tool_name=None/skipped steps). Used for "
        "loop-prevention checks (settings.max_tool_calls total, settings.max_retries per tool).",
    )
    tool_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Tool outputs keyed by str(agent_step_count) at the time of execution -- "
        "NOT by step_id, since every replan resets step_id back to 0, which would silently "
        "collide/overwrite across cycles. Kept as dict[str, Any] deliberately -- tool output "
        "shapes are open-ended/pluggable per registered tool, unlike retrieved_evidence which "
        "has one well-known shape.",
    )

    # Retrieval and Synthesis
    retrieved_evidence: RAGResult | None = Field(
        default=None,
        description="Structured, source-attributed evidence from RAGService.retrieve() "
        "(conditional RAG). None if RAG has not been invoked yet for this request. NOTE: "
        "left unpopulated by the current graph -- RAG evidence flows through tool_results "
        "generically like any other tool output, since the graph package is deliberately "
        "barred from importing the concrete RAGSearchOutput type (see graph/nodes.py).",
    )
    synthesis_attempts: int = Field(
        default=0,
        ge=0,
        description="Number of synthesize() calls made so far this run: the initial attempt "
        "plus at most one bounded structural-correction retry (see validate_output / "
        "route_after_validate_output). Bounded so the correction loop always terminates.",
    )
    validation_issues: list[str] = Field(
        default_factory=list,
        description="Structural violations found by the most recent validate_output call "
        "(backend.agents.validator.validate_structure), if any -- e.g. a missing required "
        "section or wrong bullet count. Cleared once resolved or once the bounded correction "
        "budget is exhausted. Never contains semantic-correctness feedback, only deterministic "
        "structural checks.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Operational warnings accrued across nodes.",
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal or fatal error messages logged during state transitions.",
    )
    execution_trace: list[ToolExecutionTrace] = Field(
        default_factory=list,
        description="Safe trace entries logged by individual nodes and tools "
        "(node/tool started, completed, duration, warning, failure). Never "
        "contains full prompts, raw documents, transcripts, or sensitive model output.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final synthesized response for the user.",
    )
    status: WorkflowStatus = Field(
        default=WorkflowStatus.IN_PROGRESS,
        description="Overall completion status of this workflow run.",
    )
