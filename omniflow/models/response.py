"""Response models returned by the OmniFlow assistant."""

from pydantic import BaseModel, Field
from omniflow.models.document import NormalizedDocument
from omniflow.models.trace import ExecutionTrace


class OmniFlowResponse(BaseModel):
    """Standardized response schema returned to clients and UI."""

    session_id: str | None = Field(
        default=None,
        description="Active session identifier for multi-turn tracking.",
    )
    answer: str | None = Field(
        default=None,
        description="Synthesized final answer produced by the assistant.",
    )
    clarification_needed: bool = Field(
        default=False,
        description="True if the request was ambiguous and requires clarification.",
    )
    clarification_prompt: str | None = Field(
        default=None,
        description="Specific question or prompt asked to resolve user ambiguity.",
    )
    normalized_documents: list[NormalizedDocument] = Field(
        default_factory=list,
        description="Normalized extracted content from any processed inputs.",
    )
    execution_trace: ExecutionTrace | None = Field(
        default=None,
        description="Safe execution trace displaying tools and operations executed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Any non-fatal operational warnings generated during execution.",
    )
