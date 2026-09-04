"""Agent state model designed for future LangGraph workflow orchestration."""

from typing import Any
from pydantic import BaseModel, Field

from omniflow.models.document import NormalizedDocument
from omniflow.models.request import UploadedInput
from omniflow.models.trace import ToolExecutionTrace


class AgentState(BaseModel):
    """Unified state container shared across future LangGraph workflow nodes.

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
    plan: list[str] = Field(
        default_factory=list,
        description="Sequential list of planned execution steps.",
    )
    planned_tools: list[str] = Field(
        default_factory=list,
        description="Identifiers of deterministic tools chosen by the planner.",
    )
    tool_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Outputs and return values mapped by tool identifier.",
    )

    # Retrieval and Synthesis
    retrieved_context: str | None = Field(
        default=None,
        description="Content retrieved from vector index (conditional RAG).",
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
        description="Safe trace entries logged by individual nodes and tools.",
    )
    final_answer: str | None = Field(
        default=None,
        description="Final synthesized response for the user.",
    )
