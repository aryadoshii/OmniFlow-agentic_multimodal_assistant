"""Response models returned by the OmniFlow assistant."""

from pydantic import BaseModel, Field
from backend.agents.cross_source import CrossSourceAnalysis
from backend.models.document import NormalizedDocument
from backend.models.evidence import EvidenceReference
from backend.models.trace import ExecutionTrace


class OmniFlowResponse(BaseModel):
    """Standardized response schema returned to clients and UI."""

    session_id: str | None = Field(
        default=None,
        description="Active session identifier for multi-turn tracking.",
    )
    status: str = Field(
        default="completed",
        description="Terminal workflow status: 'completed', 'failed', or "
        "'awaiting_clarification'. A tool/LLM failure inside the workflow is caught and "
        "still yields an HTTP 200 with a best-effort `answer` -- check this field (and "
        "`errors`) rather than assuming a 200 response means the run fully succeeded.",
    )
    answer: str | None = Field(
        default=None,
        description="Synthesized final answer produced by the assistant. May be a "
        "best-effort, partial answer if status is 'failed'; None if status is "
        "'awaiting_clarification' (synthesis never runs on that path).",
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
    errors: list[str] = Field(
        default_factory=list,
        description="Non-fatal or fatal error messages recorded during execution (e.g. a "
        "tool failure, an exhausted execution budget, or a Gemini provider error the "
        "workflow absorbed) -- empty when status is 'completed' with no issues.",
    )
    evidence: list[EvidenceReference] = Field(
        default_factory=list,
        description="Source-attributed evidence references backing the answer (RAG chunks, "
        "directly-ingested documents, YouTube transcript results), where available. Empty "
        "for e.g. general conversation with no documents/tools involved. Purely additive -- "
        "existing clients that don't read this field are unaffected.",
    )
    cross_source_analysis: CrossSourceAnalysis | None = Field(
        default=None,
        description="Structured relationship/consistency analysis across 2+ ingested "
        "sources. Present only for comparison-style requests over multiple sources; "
        "None otherwise (including single-source or tool-free requests). Purely "
        "additive -- existing clients that don't read this field are unaffected.",
    )
