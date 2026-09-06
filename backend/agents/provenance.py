"""Builds answer provenance (EvidenceReference list) from an AgentState.

Purely deterministic post-processing over data the pipeline already
produced -- RAG chunks, directly-ingested documents, and YouTube transcript
tool results -- performed once, after a run completes (see
backend/api/routes/agent.py). No LLM call, no new tool, no change to the
graph/planner: this module only *reads* AgentState.tool_results and
AgentState.normalized_documents, which already exist.

Never fabricates a page number or timestamp: fields are left None when the
underlying data doesn't support them (see each field's docstring in
backend/models/evidence.py and _page_for_chunk below).
"""

from __future__ import annotations

from typing import Any

from backend.models.evidence import EvidenceReference
from backend.models.state import AgentState
from backend.tools.rag_search import RAGSearchOutput
from backend.tools.youtube import YouTubeTranscriptOutput

_EXCERPT_MAX_CHARS = 240


def _excerpt(text: str) -> str:
    """Trims a source excerpt to a short, safe display length."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= _EXCERPT_MAX_CHARS:
        return cleaned
    return cleaned[:_EXCERPT_MAX_CHARS].rstrip() + "…"


def _page_for_chunk(metadata: dict[str, Any]) -> int | None:
    """Returns a PDF page number ONLY when it can be attributed unambiguously.

    DocumentChunker (backend/rag/chunking.py) copies the WHOLE source
    document's metadata onto every chunk it produces -- it does not track
    which page each chunk's text actually came from within a multi-page
    PDF. Reporting a page number for a chunk from a multi-page document
    would therefore be a guess, which the provenance contract explicitly
    forbids. The one case where attribution is genuinely unambiguous is a
    single-page PDF: every chunk of it can only have come from that one
    page. Multi-page PDFs currently report no page number for RAG chunks --
    a known, documented limitation, not a bug (see the Phase report).
    """
    if metadata.get("total_pages") != 1:
        return None
    pages = metadata.get("pages") or []
    if len(pages) == 1 and isinstance(pages[0], dict):
        page_number = pages[0].get("page_number")
        return page_number if isinstance(page_number, int) else None
    return None


def build_evidence_references(state: AgentState) -> list[EvidenceReference]:
    """Collects source-attributed evidence for everything a run actually used.

    RAG chunks and YouTube transcript results (both already structured,
    source-attributed tool outputs) take priority; any remaining ingested
    document not already covered by one of those gets a generic
    document-level reference, so every source the user attached is
    represented even when the planner answered directly from context
    without invoking a tool.
    """
    evidence: list[EvidenceReference] = []
    covered_document_ids: set[str] = set()

    for result in state.tool_results.values():
        if isinstance(result, RAGSearchOutput):
            for chunk in result.evidence:
                evidence.append(
                    EvidenceReference(
                        source="rag",
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        source_type=chunk.source_type,
                        extraction_method=chunk.extraction_method,
                        page=_page_for_chunk(chunk.metadata),
                        chunk_id=chunk.chunk_id,
                        score=round(chunk.score, 4),
                        excerpt=_excerpt(chunk.content),
                    )
                )
                covered_document_ids.add(chunk.document_id)
        elif isinstance(result, YouTubeTranscriptOutput):
            if result.transcript.strip():
                # No per-segment "which part was actually used" attribution
                # is available (synthesis reads the flattened transcript
                # text, not individual segments) -- the honest range to
                # report is the full span actually retained, not a
                # fabricated single timestamp.
                first_segment_start = result.segments[0].start_seconds if result.segments else None
                evidence.append(
                    EvidenceReference(
                        source="youtube_transcript",
                        filename=result.source_url,
                        source_type="youtube",
                        segment_start_seconds=first_segment_start,
                        segment_end_seconds=result.total_duration_seconds,
                        excerpt=_excerpt(result.transcript),
                    )
                )

    for doc in state.normalized_documents:
        if doc.id in covered_document_ids or not doc.content.strip():
            continue
        evidence.append(
            EvidenceReference(
                source="document",
                document_id=doc.id,
                filename=doc.filename,
                source_type=doc.source_type.value,
                extraction_method=doc.extraction_method.value,
                page=_page_for_chunk(doc.metadata),
                excerpt=_excerpt(doc.content),
            )
        )

    return evidence
