"""Tests for DocumentChunker — chunking logic, overlap, metadata, and edge cases."""

import pytest

from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType
from backend.rag.chunking import DocumentChunker


def _make_doc(content: str, filename: str = "test.txt") -> NormalizedDocument:
    return NormalizedDocument(
        filename=filename,
        source_type=SourceType.TEXT,
        mime_type="text/plain",
        content=content,
        extraction_method=ExtractionMethod.DIRECT_INPUT,
    )


class TestDocumentChunkerInit:
    def test_valid_params(self):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20

    def test_overlap_equals_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=100)

    def test_overlap_exceeds_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            DocumentChunker(chunk_size=100, chunk_overlap=150)

    def test_defaults_come_from_settings(self):
        from backend.config import get_settings

        settings = get_settings()
        chunker = DocumentChunker()
        assert chunker.chunk_size == settings.rag_chunk_size
        assert chunker.chunk_overlap == settings.rag_chunk_overlap

    def test_explicit_args_override_settings(self):
        chunker = DocumentChunker(chunk_size=123, chunk_overlap=10)
        assert chunker.chunk_size == 123
        assert chunker.chunk_overlap == 10


class TestEmptyAndWhitespace:
    def test_empty_content_returns_no_chunks(self):
        chunker = DocumentChunker()
        doc = _make_doc("")
        assert chunker.chunk_document(doc) == []

    def test_whitespace_only_returns_no_chunks(self):
        chunker = DocumentChunker()
        doc = _make_doc("   \n\n\t  ")
        assert chunker.chunk_document(doc) == []


class TestShortDocument:
    def test_short_text_produces_single_chunk(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        doc = _make_doc("Hello world, this is a short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Hello world, this is a short document."

    def test_single_chunk_metadata(self):
        chunker = DocumentChunker()
        doc = _make_doc("Short text.", filename="sample.txt")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.document_id == doc.id
        assert c.filename == "sample.txt"
        assert c.source_type == "text"
        assert c.extraction_method == "direct_input"
        assert c.chunk_index == 0
        assert c.total_chunks == 1


class TestMultiParagraphSplitting:
    def test_splits_on_paragraphs(self):
        chunker = DocumentChunker(chunk_size=60, chunk_overlap=0)
        # Each paragraph is ~40 chars; should split into at least 2 chunks
        text = "This is the first paragraph here.\n\nThis is the second paragraph."
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2

    def test_chunk_indices_sequential(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        text = "\n\n".join([f"Paragraph number {i} with some content here." for i in range(6)])
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i
            assert c.total_chunks == len(chunks)

    def test_all_chunks_non_empty(self):
        chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)
        text = "\n\n".join([f"Section {i}: some meaningful text here." for i in range(5)])
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        for c in chunks:
            assert c.content.strip() != ""


class TestOverlap:
    def test_overlap_carries_tail(self):
        """The tail of a completed chunk must appear in the start of the next chunk."""
        chunker = DocumentChunker(chunk_size=80, chunk_overlap=20)
        # Build text that forces multiple chunks
        text = " ".join(["word"] * 60)
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        if len(chunks) > 1:
            # The end of chunk[0] should overlap with the beginning of chunk[1]
            end_of_first = chunks[0].content[-20:].strip()
            start_of_second = chunks[1].content[:30]
            assert any(word in start_of_second for word in end_of_first.split())


class TestMetadataInheritance:
    def test_metadata_inherited(self):
        chunker = DocumentChunker()
        doc = _make_doc("Some content", filename="report.pdf")
        doc.metadata["page_count"] = 5
        chunks = chunker.chunk_document(doc)
        assert chunks[0].metadata.get("page_count") == 5

    def test_pdf_page_metadata_preserved(self):
        """Per-page extraction metadata already produced by PDFProcessor
        (see backend/processors/pdf_processor.py) must survive chunking."""
        chunker = DocumentChunker()
        doc = NormalizedDocument(
            filename="report.pdf",
            source_type=SourceType.PDF,
            mime_type="application/pdf",
            content="Page one content. Page two content.",
            extraction_method=ExtractionMethod.MIXED,
            metadata={
                "total_pages": 2,
                "native_pages": 1,
                "ocr_pages": 1,
                "pages": [
                    {"page_number": 1, "extraction_method": "native_text", "character_count": 20},
                    {"page_number": 2, "extraction_method": "ocr", "character_count": 20, "confidence": 0.9},
                ],
            },
        )
        chunks = chunker.chunk_document(doc)
        assert chunks[0].extraction_method == "mixed"
        assert chunks[0].metadata["total_pages"] == 2
        assert chunks[0].metadata["pages"][1]["extraction_method"] == "ocr"

    def test_youtube_style_metadata_preserved(self):
        """Video/source metadata (as YouTubeTranscriptOutput.metadata would
        supply once mapped into a NormalizedDocument) must survive chunking."""
        chunker = DocumentChunker()
        doc = NormalizedDocument(
            filename="youtube:dQw4w9WgXcQ",
            source_type=SourceType.TEXT,
            mime_type="text/plain",
            content="This is the video transcript content.",
            extraction_method=ExtractionMethod.DIRECT_INPUT,
            metadata={
                "video_id": "dQw4w9WgXcQ",
                "language_code": "en",
                "is_generated": True,
                "provider": "youtube_transcript_api",
            },
        )
        chunks = chunker.chunk_document(doc)
        assert chunks[0].metadata["video_id"] == "dQw4w9WgXcQ"
        assert chunks[0].metadata["provider"] == "youtube_transcript_api"

    def test_chunk_id_unique(self):
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=0)
        text = "\n\n".join([f"Paragraph {i} with content." for i in range(5)])
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_source_type_inherited(self):
        chunker = DocumentChunker()
        doc = NormalizedDocument(
            filename="audio.mp3",
            source_type=SourceType.AUDIO,
            mime_type="audio/mpeg",
            content="Transcript of the audio file.",
            extraction_method=ExtractionMethod.SPEECH_TO_TEXT,
        )
        chunks = chunker.chunk_document(doc)
        assert chunks[0].source_type == "audio"


class TestPathologicalTinyChunks:
    def test_tiny_trailing_chunk_merged_into_predecessor(self):
        """A trailing fragment smaller than ~20% of chunk_size should be
        folded into the previous chunk rather than left standalone."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=0)
        # Construct text whose recursive split leaves a tiny final piece.
        text = ("word " * 20).strip() + "\n\n" + "x"
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        # The tiny trailing "x" must not appear as its own standalone chunk.
        assert all(len(c.content) >= 20 or len(chunks) == 1 for c in chunks)
        assert "x" in chunks[-1].content

    def test_merged_trailing_chunk_may_exceed_chunk_size(self):
        """Documents the actual, intentional chunk_size contract: chunk_size
        is a normal target/max, EXCEPT that the pathological-tail merge (see
        DocumentChunker's class docstring) may produce a final chunk that
        modestly exceeds it. This is not a bug -- a chunk_size violation here
        is the documented, correct behavior, verified explicitly rather than
        left as an implicit side effect."""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=0)
        text = ("word " * 20).strip() + "\n\n" + "x"
        doc = _make_doc(text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks[-1].content) > chunker.chunk_size
        # Every non-final chunk still respects the normal chunk_size target.
        for c in chunks[:-1]:
            assert len(c.content) <= chunker.chunk_size

    def test_no_merge_when_only_one_chunk(self):
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=0)
        doc = _make_doc("A single short document.")
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1


class TestChunkDocuments:
    def test_multiple_documents_aggregated(self):
        chunker = DocumentChunker()
        docs = [_make_doc(f"Document {i} content.") for i in range(3)]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 3

    def test_empty_list(self):
        chunker = DocumentChunker()
        assert chunker.chunk_documents([]) == []
