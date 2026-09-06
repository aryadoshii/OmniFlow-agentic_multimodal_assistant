"""Tests for backend.agents.provenance.build_evidence_references().

All fixtures are constructed directly (no real RAG/embedding/YouTube call) --
this module only reads AgentState.tool_results/normalized_documents, which
these tests populate by hand.
"""

from backend.agents.provenance import build_evidence_references
from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType
from backend.models.state import AgentState
from backend.rag.service import RetrievedChunk
from backend.tools.rag_search import RAGSearchOutput
from backend.tools.youtube import TranscriptSegment, YouTubeTranscriptOutput


def _doc(**overrides) -> NormalizedDocument:
    defaults = dict(
        id="doc-1",
        filename="report.pdf",
        source_type=SourceType.PDF,
        mime_type="application/pdf",
        content="Revenue rose 15% in Q3 across all regions.",
        extraction_method=ExtractionMethod.NATIVE_TEXT,
        metadata={"total_pages": 3, "pages": [{"page_number": 1}, {"page_number": 2}, {"page_number": 3}]},
    )
    defaults.update(overrides)
    return NormalizedDocument(**defaults)


class TestRAGEvidence:
    def test_rag_chunk_becomes_evidence_reference(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="doc-1",
            filename="report.pdf",
            source_type="pdf",
            extraction_method="native_text",
            content="Revenue rose 15% in Q3 across all regions.",
            score=0.87,
            chunk_index=0,
            total_chunks=2,
            metadata={"total_pages": 3, "pages": [{"page_number": 1}]},
        )
        output = RAGSearchOutput(
            query="revenue growth",
            status="evidence_found",
            evidence=[chunk],
            evidence_count=1,
            top_k=4,
            score_threshold=0.2,
        )
        state = AgentState(
            normalized_documents=[_doc()],
            tool_results={"0": output},
        )

        evidence = build_evidence_references(state)

        assert len(evidence) == 1
        ref = evidence[0]
        assert ref.source == "rag"
        assert ref.document_id == "doc-1"
        assert ref.filename == "report.pdf"
        assert ref.chunk_id == "c1"
        assert ref.score == 0.87
        assert "Revenue rose" in ref.excerpt

    def test_multi_page_document_never_gets_a_fabricated_page_number(self) -> None:
        """A chunk from a 3-page PDF must NOT report a page -- the chunker
        doesn't track which page a chunk's text actually came from."""
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="doc-1",
            filename="report.pdf",
            source_type="pdf",
            extraction_method="native_text",
            content="some text",
            score=0.5,
            chunk_index=0,
            total_chunks=5,
            metadata={"total_pages": 3, "pages": [{"page_number": 1}, {"page_number": 2}, {"page_number": 3}]},
        )
        output = RAGSearchOutput(
            query="q", status="evidence_found", evidence=[chunk], evidence_count=1, top_k=4, score_threshold=0.2
        )
        state = AgentState(normalized_documents=[_doc()], tool_results={"0": output})

        evidence = build_evidence_references(state)

        assert evidence[0].page is None

    def test_single_page_document_reports_the_page_unambiguously(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1",
            document_id="doc-1",
            filename="one_pager.pdf",
            source_type="pdf",
            extraction_method="native_text",
            content="some text",
            score=0.5,
            chunk_index=0,
            total_chunks=1,
            metadata={"total_pages": 1, "pages": [{"page_number": 1}]},
        )
        output = RAGSearchOutput(
            query="q", status="evidence_found", evidence=[chunk], evidence_count=1, top_k=4, score_threshold=0.2
        )
        state = AgentState(normalized_documents=[_doc()], tool_results={"0": output})

        evidence = build_evidence_references(state)

        assert evidence[0].page == 1

    def test_no_evidence_rag_result_produces_no_rag_references(self) -> None:
        output = RAGSearchOutput(
            query="q", status="no_evidence", evidence=[], evidence_count=0, top_k=4, score_threshold=0.2,
            message="No evidence found.",
        )
        state = AgentState(normalized_documents=[_doc()], tool_results={"0": output})

        evidence = build_evidence_references(state)

        # The document itself still gets a generic direct-context reference.
        assert len(evidence) == 1
        assert evidence[0].source == "document"


class TestYouTubeEvidence:
    def test_youtube_transcript_becomes_evidence_with_timing(self) -> None:
        output = YouTubeTranscriptOutput(
            video_id="abc12345678",
            source_url="https://www.youtube.com/watch?v=abc12345678",
            transcript="This lecture covers clinical trial workflows.",
            segments=[
                TranscriptSegment(text="This lecture covers", start_seconds=0.0, duration_seconds=2.0),
                TranscriptSegment(text="clinical trial workflows.", start_seconds=2.0, duration_seconds=3.0),
            ],
            segment_count=2,
            total_duration_seconds=5.0,
        )
        state = AgentState(normalized_documents=[], tool_results={"0": output})

        evidence = build_evidence_references(state)

        assert len(evidence) == 1
        ref = evidence[0]
        assert ref.source == "youtube_transcript"
        assert ref.segment_start_seconds == 0.0
        assert ref.segment_end_seconds == 5.0
        assert "clinical trial workflows" in ref.excerpt

    def test_empty_transcript_produces_no_evidence(self) -> None:
        output = YouTubeTranscriptOutput(
            video_id="abc12345678",
            source_url="https://www.youtube.com/watch?v=abc12345678",
            transcript="",
        )
        state = AgentState(normalized_documents=[], tool_results={"0": output})

        assert build_evidence_references(state) == []


class TestDirectDocumentEvidence:
    def test_document_not_covered_by_any_tool_gets_generic_reference(self) -> None:
        state = AgentState(normalized_documents=[_doc()], tool_results={})

        evidence = build_evidence_references(state)

        assert len(evidence) == 1
        assert evidence[0].source == "document"
        assert evidence[0].document_id == "doc-1"

    def test_document_already_covered_by_rag_is_not_duplicated(self) -> None:
        chunk = RetrievedChunk(
            chunk_id="c1", document_id="doc-1", filename="report.pdf", source_type="pdf",
            extraction_method="native_text", content="text", score=0.5, chunk_index=0, total_chunks=1,
        )
        output = RAGSearchOutput(
            query="q", status="evidence_found", evidence=[chunk], evidence_count=1, top_k=4, score_threshold=0.2
        )
        state = AgentState(normalized_documents=[_doc()], tool_results={"0": output})

        evidence = build_evidence_references(state)

        assert len(evidence) == 1
        assert evidence[0].source == "rag"

    def test_empty_content_document_produces_no_evidence(self) -> None:
        state = AgentState(normalized_documents=[_doc(content="")], tool_results={})

        assert build_evidence_references(state) == []

    def test_excerpt_is_truncated_for_long_content(self) -> None:
        long_content = "word " * 200
        state = AgentState(normalized_documents=[_doc(content=long_content)], tool_results={})

        evidence = build_evidence_references(state)

        assert len(evidence[0].excerpt) <= 241
        assert evidence[0].excerpt.endswith("…")
