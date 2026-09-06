"""Cross-source evidence & consistency analysis (multi-input comparison).

Answers requests like "do the audio and PDF discuss the same topic?" or
"find contradictions across these files" with an explicit, structured
relationship classification -- not a plain-text opinion buried in the
regular answer.

This is NOT a second agent: it is one more deterministic-structure +
single-LLM-call function, following exactly the same pattern as
backend.agents.intent.understand_intent() and
backend.agents.synthesizer.synthesize_answer() (a BaseLLMProvider
structured-output call over already-ingested content). It is invoked once,
after the graph's normal run completes, only when the classified intent is
IntentType.COMPARISON and 2+ documents were actually ingested (see
backend/api/routes/agent.py) -- it never runs inside the graph itself and
never influences planning/tool selection, so the existing agent
architecture is unchanged.

Kept generic (a relationship enum + shared/differences lists) so future
comparison-style requests can reuse this same function/model rather than
each needing their own bespoke analysis path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from backend.providers.base import BaseLLMProvider

if TYPE_CHECKING:
    # Deferred to break a circular import -- see reference_resolver.py /
    # intent.py for the same pattern and why it's needed: AgentState (in
    # backend.models.state) transitively imports backend.agents.intent,
    # which would otherwise import this module's package before it
    # finishes initializing.
    from backend.models.state import AgentState

logger = logging.getLogger(__name__)

# Same reasoning as graph.nodes._MAX_CONTEXT_CHARS_PER_DOCUMENT -- bounds the
# prompt regardless of document size. Kept smaller than that constant since
# this prompt includes ALL sources at once (unlike synthesis, which already
# has unified_context built once) plus per-source structure instructions.
_MAX_CONTENT_CHARS_PER_SOURCE = 4000

RelationshipType = Literal[
    "strong_overlap",
    "partial_overlap",
    "different_topics",
    "contradiction",
    "insufficient_evidence",
]


class SourceUnderstanding(BaseModel):
    """One source's detected subject, grounded in its own content only."""

    document_id: str = Field(..., description="The NormalizedDocument id this understanding is about.")
    filename: str = Field(..., description="The originating filename, for display.")
    topic: str = Field(..., description="Concise (a few words) detected subject/topic for this source.")
    evidence: str = Field(
        ..., description="Concise (1 sentence) supporting evidence for the detected topic, from this source's own content."
    )


class CrossSourceAnalysis(BaseModel):
    """Structured result of comparing 2+ ingested sources.

    Reusable for any future comparison-style request -- the shape says
    nothing file-type- or pair-specific.
    """

    relationship: RelationshipType = Field(
        ..., description="Overall classified relationship between the sources."
    )
    sources: list[SourceUnderstanding] = Field(default_factory=list)
    shared_concepts: list[str] = Field(
        default_factory=list, description="Concepts/topics present across multiple sources, if any."
    )
    differences: list[str] = Field(
        default_factory=list, description="Key differences or contradictions found, if any."
    )
    explanation: str = Field(
        ..., description="Concise (1-3 sentence) explanation of WHY the relationship was classified that way."
    )


_SYSTEM_INSTRUCTION = """\
You are the cross-source analysis component of a deterministic multimodal \
assistant. You are given 2 or more already-ingested sources (documents, \
audio transcripts, images) and must classify how they relate to each \
other. You do not answer the user's original question directly -- you only \
produce this structured comparison.

Rules:
- Ground every claim ONLY in the source content actually provided below. \
Never invent evidence, facts, or topics not present in the sources.
- Classify `relationship` as exactly one of: strong_overlap (sources \
substantially discuss the same specific topic), partial_overlap (related \
subject matter but not the same specific topic), different_topics (no \
meaningful topical connection), contradiction (sources make conflicting \
claims about the same topic), or insufficient_evidence (a source's content \
is too sparse/unclear to classify confidently).
- Produce one `sources` entry per source, each with a concise topic and \
one sentence of grounding evidence FROM THAT SOURCE.
- `shared_concepts` and `differences` must both be concise bullet-style \
phrases, not paragraphs. Either list may be empty if genuinely not \
applicable (e.g. no shared concepts for different_topics).
- `explanation` must be 1-3 sentences, plain language, no chain-of-thought, \
no meta-commentary about your own reasoning process -- just the conclusion \
and its basis.
"""


def _build_prompt(state: AgentState) -> str:
    sections = []
    for doc in state.normalized_documents:
        content = doc.content[:_MAX_CONTENT_CHARS_PER_SOURCE]
        sections.append(
            f"[document_id={doc.id} filename={doc.filename} type={doc.source_type.value}]\n{content}"
        )
    sources_text = "\n\n".join(sections)
    return (
        f"User's original request: {state.original_request}\n\n"
        f"Sources to compare:\n\n{sources_text}"
    )


def analyze_cross_source(
    state: AgentState, llm_provider: BaseLLMProvider
) -> CrossSourceAnalysis | None:
    """Produces a CrossSourceAnalysis for a multi-document request.

    Returns None (rather than raising) when there are fewer than 2 ingested
    documents -- there is nothing to compare, and callers (see
    backend/api/routes/agent.py) already gate on intent + document count
    before calling this, so None here signals "not applicable", not an
    error.

    Raises:
        ConfigurationError: Propagated as-is from llm_provider.
        ExternalProviderError: Propagated as-is from llm_provider if the
                                request fails or output can't be validated.
    """
    if len(state.normalized_documents) < 2:
        return None

    prompt = _build_prompt(state)
    logger.info(
        "Running cross-source analysis for %d document(s).", len(state.normalized_documents)
    )
    result = llm_provider.generate_structured(
        prompt, CrossSourceAnalysis, system_instruction=_SYSTEM_INSTRUCTION
    )
    logger.info("Cross-source analysis classified relationship=%s", result.relationship)
    return result
