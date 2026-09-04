"""Safe execution trace models for OmniFlow.

Contains only non-sensitive metadata such as operation names, statuses,
durations, and concise diagnostic summaries. Does NOT store API keys,
full prompts, raw document contents, or large model outputs.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ToolExecutionTrace(BaseModel):
    """Metadata record for an individual tool or processor execution step."""

    step_name: str = Field(
        ...,
        description="Logical name of the executed step (e.g. 'detect_youtube_urls').",
    )
    tool_name: str | None = Field(
        default=None,
        description="Name of the specific deterministic tool or processor used.",
    )
    status: Literal["pending", "success", "failed"] = Field(
        default="success",
        description="Terminal or current status of this execution step.",
    )
    duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Duration of the step in milliseconds.",
    )
    details: dict[str, str] = Field(
        default_factory=dict,
        description="Concise, non-sensitive diagnostic details (e.g. item counts).",
    )
    error_message: str | None = Field(
        default=None,
        description="Sanitized error description if the step failed.",
    )


class ExecutionTrace(BaseModel):
    """Aggregated, safe trace of all steps taken during request fulfillment."""

    session_id: str | None = Field(
        default=None,
        description="Session identifier associated with this trace.",
    )
    total_duration_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Total processing time in milliseconds.",
    )
    steps: list[ToolExecutionTrace] = Field(
        default_factory=list,
        description="Ordered sequence of safe execution trace records.",
    )
