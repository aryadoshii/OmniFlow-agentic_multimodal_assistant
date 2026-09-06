"""Semantic intent/constraint/reference understanding (Phase 4.3).

Converts a user request + deterministic document context into a structured
IntentResult via BaseLLMProvider's structured-output capability. This module
performs no tool execution, no RAG/FAISS/YouTube calls, and no clarification
behavior -- it only produces the structured signal (``needs_clarification``,
``clarification_question``) that a future clarification node (Phase 4.6)
will act on.

Reference resolution is split deliberately: obvious single-document phrases
("this PDF" when there is exactly one PDF) are resolved deterministically by
reference_resolver.py before the LLM is ever called -- the LLM is only asked
to handle genuine semantic ambiguity (e.g. "the video mentioned in the PDF",
which requires reading document content), not basic id lookup.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from backend.agents.reference_resolver import (
    build_document_catalog,
    resolve_unambiguous_references,
)
from backend.providers.base import BaseLLMProvider

if TYPE_CHECKING:
    # Deferred to break a circular import -- see reference_resolver.py for why.
    from backend.models.state import AgentState

logger = logging.getLogger(__name__)

# Caps how much of each document's content is included in the prompt, to
# keep requests bounded regardless of document size. This is an internal
# prompt-construction detail, not a user-facing tunable -- deliberately not
# added to Settings. Matches graph.nodes._MAX_CONTEXT_CHARS_PER_DOCUMENT so
# intent understanding sees at least as much of each document as synthesis
# later will -- otherwise a reference mentioned only beyond this cutoff
# (e.g. a YouTube URL past the first few thousand characters of a long PDF)
# would be visible to synthesis but invisible to intent/planning, silently
# breaking a "the video mentioned in this PDF" style request.
_MAX_DOCUMENT_PREVIEW_CHARS = 8000


class IntentType(str, Enum):
    """Supported semantic intents. Deliberately kept to the assignment's
    required set -- not a large, speculative taxonomy."""

    GENERAL_CONVERSATION = "general_conversation"
    QUESTION_ANSWERING = "question_answering"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    CODE_EXPLANATION = "code_explanation"
    TRANSCRIPTION_SUMMARY = "transcription_summary"
    COMPARISON = "comparison"


class IntentResult(BaseModel):
    """Structured understanding of a user's request.

    ``needs_clarification``/``clarification_question`` are only the signal --
    the clarification node itself (Phase 4.6) decides what to do with them.
    """

    intent: IntentType = Field(..., description="Primary classified intent.")
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit user constraints verbatim (e.g. 'exactly 3 bullets', "
        "'one-line summary', 'for a beginner audience') -- never rewritten or reworded.",
    )
    referenced_inputs: list[str] = Field(
        default_factory=list,
        description="document_id values (from the provided catalog) that this request refers to.",
    )
    relevant_references: list[str] = Field(
        default_factory=list,
        description="Relevant URLs or free-text references mentioned in the request or "
        "found within document content (e.g. a YouTube URL referenced inside a PDF).",
    )
    is_ambiguous: bool = Field(
        default=False, description="True if the request could reasonably be interpreted multiple ways."
    )
    needs_clarification: bool = Field(
        default=False,
        description="True if the request cannot be safely fulfilled without asking the user a question.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="The question to ask the user, if needs_clarification is True; otherwise None.",
    )
    explanation: str = Field(
        ..., description="Concise (1-2 sentence) explanation of what is required to fulfill this request."
    )


_INTENT_SYSTEM_INSTRUCTION = """\
You are the intent-understanding component of a deterministic multimodal \
assistant. Classify the user's request and extract structured information \
about it. You do not answer the request and you do not execute any tool.

Rules:
- Choose exactly one primary intent from the provided enum.
- Extract constraints VERBATIM from the user's own wording -- do not \
paraphrase, summarize, or add constraints the user did not state.
- Populate referenced_inputs using ONLY document_id values from the provided \
document catalog or the deterministic reference hints -- never invent an id.
- Use relevant_references for URLs or named external sources the request or \
document content mentions (e.g. a YouTube link), not for document ids.
- Set needs_clarification=true only when the request genuinely cannot be \
safely fulfilled as stated (e.g. it references a document that does not \
exist, or is fundamentally unclear about what output is wanted). Do not \
request clarification for stylistic preferences you can reasonably assume.
- Keep explanation concise: 1-2 sentences, no chain-of-thought.
"""


def _build_document_context(state: AgentState) -> str:
    """Builds the document-catalog and content-preview section of the prompt."""
    catalog = build_document_catalog(state)
    catalog_json = json.dumps([entry.model_dump() for entry in catalog])

    deterministic_refs = resolve_unambiguous_references(state)
    refs_json = json.dumps(deterministic_refs)

    previews = []
    for doc in state.normalized_documents:
        preview = doc.content[:_MAX_DOCUMENT_PREVIEW_CHARS]
        previews.append(f"[document_id={doc.id} filename={doc.filename}]\n{preview}")
    previews_text = "\n\n".join(previews) if previews else "(no ingested document content)"

    return (
        f"Document catalog (id/filename/type):\n{catalog_json}\n\n"
        f"Deterministic reference hints (phrase -> document_id list, already "
        f"resolved -- reuse these directly, do not re-resolve them):\n{refs_json}\n\n"
        f"Document content previews:\n{previews_text}"
    )


def understand_intent(state: AgentState, llm_provider: BaseLLMProvider) -> IntentResult:
    """Classifies intent, extracts constraints, and resolves references for a request.

    Args:
        state: The current request's AgentState. Only read, never mutated --
               callers (a future graph node) are responsible for merging the
               returned IntentResult into a new AgentState.
        llm_provider: A BaseLLMProvider implementation (e.g. GeminiProvider).
                      This function never imports a concrete provider itself.

    Returns:
        A validated IntentResult.

    Raises:
        ConfigurationError: Propagated as-is from llm_provider if
                             credentials/model configuration are invalid.
        ExternalProviderError: Propagated as-is from llm_provider if the LLM
                                request fails or its output cannot be
                                validated into IntentResult.
    """
    document_context = _build_document_context(state)
    prompt = (
        f"User request:\n{state.original_request}\n\n"
        f"{document_context}"
    )

    logger.info(
        "Understanding intent for request (len=%d, documents=%d).",
        len(state.original_request),
        len(state.normalized_documents),
    )
    result = llm_provider.generate_structured(
        prompt, IntentResult, system_instruction=_INTENT_SYSTEM_INSTRUCTION
    )
    logger.info(
        "Intent classified: intent=%s needs_clarification=%s",
        result.intent.value,
        result.needs_clarification,
    )
    return result
