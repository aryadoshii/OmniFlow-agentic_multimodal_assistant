"""Integration-style tests proving Chunker -> EmbeddingService -> FAISSVectorStore
compose correctly end-to-end (Phase 3.4).

This deliberately does NOT introduce a new pipeline/orchestration class --
per Phase 3.4 scope, the full RAG pipeline (RAGService) is a separate,
already-drafted component out of scope here. These tests manually wire the
three independent pieces together to prove the composition works, using a
mocked embedding boundary so no ML model is loaded.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from omniflow.exceptions import EmbeddingGenerationError
from omniflow.models.document import ExtractionMethod, NormalizedDocument, SourceType
from omniflow.rag.chunking import DocumentChunker
from omniflow.rag.vector_store import FAISSVectorStore

DIMENSION = 16


def _fake_embedding_service() -> MagicMock:
    """A mock EmbeddingService that returns deterministic, distinct vectors
    per call based on a hash of the input text, L2-normalized."""
    service = MagicMock()

    def _embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.random(DIMENSION).astype(np.float32)
            vecs.append(v / np.linalg.norm(v))
        return np.vstack(vecs).astype(np.float32)

    service.embed_texts.side_effect = _embed
    return service


def _make_doc(
    content: str, filename: str, source_type: SourceType, extraction_method: ExtractionMethod
) -> NormalizedDocument:
    return NormalizedDocument(
        filename=filename,
        source_type=source_type,
        mime_type="text/plain",
        content=content,
        extraction_method=extraction_method,
    )


def _index_document(doc, chunker, embedding_service, store) -> list:
    chunks = chunker.chunk_document(doc)
    if not chunks:
        return []
    texts = [c.content for c in chunks]
    vectors = embedding_service.embed_texts(texts)
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "filename": c.filename,
            "source_type": c.source_type,
            "extraction_method": c.extraction_method,
        }
        for c in chunks
    ]
    store.add_texts(texts, metadatas, embeddings=vectors)
    return chunks


class TestChunkEmbedStoreComposition:
    def test_single_document_round_trip(self) -> None:
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        embedding_service = _fake_embedding_service()
        store = FAISSVectorStore(dimension=DIMENSION)

        doc = _make_doc(
            "This is a small document about OmniFlow's RAG pipeline.",
            filename="notes.txt",
            source_type=SourceType.TEXT,
            extraction_method=ExtractionMethod.DIRECT_INPUT,
        )

        chunks = _index_document(doc, chunker, embedding_service, store)
        assert len(chunks) >= 1
        assert store.count() == len(chunks)

        query_vector = embedding_service.embed_texts(["OmniFlow RAG pipeline"])
        results = store.search(query_vector, top_k=1)

        assert len(results) == 1
        assert results[0]["metadata"]["filename"] == "notes.txt"
        assert results[0]["metadata"]["document_id"] == doc.id

    def test_multiple_documents_source_attribution_preserved(self) -> None:
        """Chunks from different documents must remain individually
        attributable to their correct source after round-tripping through
        embedding and FAISS storage."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=0)
        embedding_service = _fake_embedding_service()
        store = FAISSVectorStore(dimension=DIMENSION)

        doc_a = _make_doc(
            "Content unique to document A about invoices.",
            filename="invoice.pdf",
            source_type=SourceType.PDF,
            extraction_method=ExtractionMethod.NATIVE_TEXT,
        )
        doc_b = _make_doc(
            "Content unique to document B about audio transcripts.",
            filename="call.mp3",
            source_type=SourceType.AUDIO,
            extraction_method=ExtractionMethod.SPEECH_TO_TEXT,
        )

        chunks_a = _index_document(doc_a, chunker, embedding_service, store)
        chunks_b = _index_document(doc_b, chunker, embedding_service, store)

        assert store.count() == len(chunks_a) + len(chunks_b)

        # Query with the exact embedding of doc_a's own chunk -> must retrieve doc_a.
        query_vector = embedding_service.embed_texts([chunks_a[0].content])
        results = store.search(query_vector, top_k=1)

        assert results[0]["metadata"]["filename"] == "invoice.pdf"
        assert results[0]["metadata"]["source_type"] == "pdf"
        assert results[0]["metadata"]["document_id"] == doc_a.id
        assert results[0]["metadata"]["document_id"] != doc_b.id

    def test_empty_document_produces_no_chunks_and_nothing_indexed(self) -> None:
        chunker = DocumentChunker()
        embedding_service = _fake_embedding_service()
        store = FAISSVectorStore(dimension=DIMENSION)

        doc = _make_doc(
            "   ",
            filename="empty.txt",
            source_type=SourceType.TEXT,
            extraction_method=ExtractionMethod.DIRECT_INPUT,
        )
        chunks = _index_document(doc, chunker, embedding_service, store)

        assert chunks == []
        assert store.count() == 0
        embedding_service.embed_texts.assert_not_called()

    def test_embedding_failure_leaves_store_untouched(self) -> None:
        """If embedding raises, no partial/corrupt entries should land in the store."""
        chunker = DocumentChunker()
        store = FAISSVectorStore(dimension=DIMENSION)

        failing_embedding_service = MagicMock()
        failing_embedding_service.embed_texts.side_effect = EmbeddingGenerationError(
            "Simulated embedding backend failure."
        )

        doc = _make_doc(
            "Some content that will fail to embed.",
            filename="doc.txt",
            source_type=SourceType.TEXT,
            extraction_method=ExtractionMethod.DIRECT_INPUT,
        )

        with pytest.raises(EmbeddingGenerationError):
            _index_document(doc, chunker, failing_embedding_service, store)

        assert store.count() == 0

    def test_search_on_empty_store_before_any_indexing(self) -> None:
        embedding_service = _fake_embedding_service()
        store = FAISSVectorStore(dimension=DIMENSION)

        query_vector = embedding_service.embed_texts(["anything"])
        results = store.search(query_vector, top_k=4)
        assert results == []
