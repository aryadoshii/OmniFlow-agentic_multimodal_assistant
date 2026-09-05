"""Deterministic reference resolution for intent understanding (Phase 4.3).

Resolves obvious phrase -> document references (e.g. "this PDF" when exactly
one PDF is present) directly from AgentState's own typed metadata -- no LLM
call. Only genuinely ambiguous or cross-document semantic references (e.g.
"the video mentioned in the PDF", or "this PDF" when there are two PDFs) are
left unresolved here, for the LLM to reason about using the catalog this
module also builds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from omniflow.models.document import NormalizedDocument, SourceType

if TYPE_CHECKING:
    # Deferred to break a circular import: AgentState (models/state.py) itself
    # references Plan (agents/planner.py), so agents/* must not import
    # AgentState at runtime. Only used here for type hints -- this module
    # only ever does attribute access on `state`, never isinstance/construction.
    from omniflow.models.state import AgentState

# Maps a modality to the singular noun phrase a user would naturally use
# ("this PDF", "the image", "the audio", "this text") when exactly one
# document of that type is present.
_PHRASE_BY_SOURCE_TYPE: dict[SourceType, str] = {
    SourceType.PDF: "pdf",
    SourceType.IMAGE: "image",
    SourceType.AUDIO: "audio",
    SourceType.TEXT: "text",
}


class DocumentCatalogEntry(BaseModel):
    """A compact, typed summary of one ingested document for LLM context."""

    document_id: str = Field(..., description="NormalizedDocument.id")
    filename: str = Field(..., description="Original filename.")
    source_type: str = Field(..., description="Modality (pdf, image, audio, text).")


def build_document_catalog(state: AgentState) -> list[DocumentCatalogEntry]:
    """Builds a compact, typed catalog of the request's normalized documents.

    Given directly to the LLM as context so it can reference real document
    ids in its structured output rather than inventing or guessing them.
    """
    return [
        DocumentCatalogEntry(
            document_id=doc.id, filename=doc.filename, source_type=doc.source_type.value
        )
        for doc in state.normalized_documents
    ]


def resolve_unambiguous_references(state: AgentState) -> dict[str, list[str]]:
    """Deterministically resolves "this <type>"-style phrases to document ids.

    A phrase like "pdf" is resolved only when EXACTLY ONE document of that
    source_type is present -- if there are two PDFs, "this PDF" is
    genuinely ambiguous and is deliberately left unresolved (the LLM, or a
    future clarification step, must handle that case; this function never
    guesses). "documents" resolves to every ingested document whenever at
    least one exists, since "these documents" is unambiguous regardless of
    type mix or count.

    Returns:
        Mapping of phrase key ("pdf", "image", "audio", "text", "documents")
        to the list of matching document ids.
    """
    by_type: dict[SourceType, list[NormalizedDocument]] = {}
    for doc in state.normalized_documents:
        by_type.setdefault(doc.source_type, []).append(doc)

    resolved: dict[str, list[str]] = {}
    for source_type, docs in by_type.items():
        if len(docs) == 1:
            phrase = _PHRASE_BY_SOURCE_TYPE.get(source_type)
            if phrase:
                resolved[phrase] = [docs[0].id]

    if state.normalized_documents:
        resolved["documents"] = [doc.id for doc in state.normalized_documents]

    return resolved
