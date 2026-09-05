"""Tests for deterministic reference resolution (Phase 4.3).

No LLM, no mocking -- resolve_unambiguous_references() and
build_document_catalog() are pure functions over AgentState.
"""

from omniflow.agents.reference_resolver import (
    build_document_catalog,
    resolve_unambiguous_references,
)
from omniflow.models.document import ExtractionMethod, NormalizedDocument, SourceType
from omniflow.models.state import AgentState


def _doc(
    filename: str, source_type: SourceType, content: str = "content"
) -> NormalizedDocument:
    extraction = {
        SourceType.TEXT: ExtractionMethod.DIRECT_INPUT,
        SourceType.PDF: ExtractionMethod.NATIVE_TEXT,
        SourceType.IMAGE: ExtractionMethod.OCR,
        SourceType.AUDIO: ExtractionMethod.SPEECH_TO_TEXT,
    }[source_type]
    return NormalizedDocument(
        filename=filename,
        source_type=source_type,
        mime_type="application/octet-stream",
        content=content,
        extraction_method=extraction,
    )


class TestBuildDocumentCatalog:
    def test_empty_state_returns_empty_catalog(self) -> None:
        assert build_document_catalog(AgentState()) == []

    def test_catalog_entries_match_documents(self) -> None:
        doc = _doc("report.pdf", SourceType.PDF)
        state = AgentState(normalized_documents=[doc])
        catalog = build_document_catalog(state)
        assert len(catalog) == 1
        assert catalog[0].document_id == doc.id
        assert catalog[0].filename == "report.pdf"
        assert catalog[0].source_type == "pdf"


class TestResolveUnambiguousReferences:
    def test_empty_state_resolves_nothing(self) -> None:
        assert resolve_unambiguous_references(AgentState()) == {}

    def test_single_pdf_resolves_this_pdf(self) -> None:
        doc = _doc("report.pdf", SourceType.PDF)
        state = AgentState(normalized_documents=[doc])
        resolved = resolve_unambiguous_references(state)
        assert resolved["pdf"] == [doc.id]

    def test_single_image_resolves_this_image(self) -> None:
        doc = _doc("photo.png", SourceType.IMAGE)
        state = AgentState(normalized_documents=[doc])
        resolved = resolve_unambiguous_references(state)
        assert resolved["image"] == [doc.id]

    def test_single_audio_resolves_this_audio(self) -> None:
        doc = _doc("call.mp3", SourceType.AUDIO)
        state = AgentState(normalized_documents=[doc])
        resolved = resolve_unambiguous_references(state)
        assert resolved["audio"] == [doc.id]

    def test_two_pdfs_leaves_pdf_phrase_unresolved(self) -> None:
        """Genuinely ambiguous -- must NOT guess which PDF 'this PDF' means."""
        doc_a = _doc("report_a.pdf", SourceType.PDF)
        doc_b = _doc("report_b.pdf", SourceType.PDF)
        state = AgentState(normalized_documents=[doc_a, doc_b])
        resolved = resolve_unambiguous_references(state)
        assert "pdf" not in resolved

    def test_mixed_types_resolves_each_singular_type_independently(self) -> None:
        pdf = _doc("report.pdf", SourceType.PDF)
        audio = _doc("call.mp3", SourceType.AUDIO)
        state = AgentState(normalized_documents=[pdf, audio])
        resolved = resolve_unambiguous_references(state)
        assert resolved["pdf"] == [pdf.id]
        assert resolved["audio"] == [audio.id]

    def test_these_documents_resolves_to_all_ids_regardless_of_type_or_count(self) -> None:
        pdf = _doc("report.pdf", SourceType.PDF)
        audio = _doc("call.mp3", SourceType.AUDIO)
        image = _doc("photo.png", SourceType.IMAGE)
        state = AgentState(normalized_documents=[pdf, audio, image])
        resolved = resolve_unambiguous_references(state)
        assert set(resolved["documents"]) == {pdf.id, audio.id, image.id}

    def test_two_pdfs_still_resolves_documents_phrase(self) -> None:
        doc_a = _doc("a.pdf", SourceType.PDF)
        doc_b = _doc("b.pdf", SourceType.PDF)
        state = AgentState(normalized_documents=[doc_a, doc_b])
        resolved = resolve_unambiguous_references(state)
        assert set(resolved["documents"]) == {doc_a.id, doc_b.id}
        assert "pdf" not in resolved
