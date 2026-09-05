"""RAGService: orchestrates document chunking, embedding, indexing, and retrieval.

This is the primary entry point for the RAG layer. It composes
DocumentChunker, EmbeddingService, and FAISSVectorStore into a single
interface that indexes NormalizedDocuments and retrieves ranked evidence
chunks for a query.

The output of retrieve() is EVIDENCE for a future LLM synthesis node --
this service never generates an answer, never calls an LLM, and has no
dependency on FastAPI, LangGraph, or any specific LLM provider.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from omniflow.config import get_settings
from omniflow.exceptions import InvalidInputError
from omniflow.models.document import NormalizedDocument
from omniflow.rag.chunking import DocumentChunk, DocumentChunker
from omniflow.rag.embeddings import EmbeddingService
from omniflow.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)

# Two retrieved chunks with a text similarity ratio at or above this value are
# considered near-duplicate evidence (typically adjacent, overlapping chunks
# from the same document); the lower-scoring one is dropped.
_DEDUP_SIMILARITY_RATIO = 0.9


class RetrievedChunk(BaseModel):
    """A single chunk of evidence returned by semantic search, with full source attribution."""

    chunk_id: str = Field(..., description="Unique chunk identifier.")
    document_id: str = Field(..., description="ID of the originating NormalizedDocument.")
    filename: str = Field(..., description="Filename of the originating document.")
    source_type: str = Field(..., description="Modality of the source (pdf, text, image, audio).")
    extraction_method: str = Field(
        ..., description="Extraction method used on the source document (native_text, ocr, mixed, speech_to_text, direct_input)."
    )
    content: str = Field(..., description="Text content of the retrieved chunk.")
    score: float = Field(..., description="Cosine similarity score (0.0-1.0) against the query.")
    chunk_index: int = Field(..., description="Zero-based position of this chunk within its source document.")
    total_chunks: int = Field(..., description="Total number of chunks produced for the source document.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Page/segment/source metadata inherited from the source document, when available.",
    )


class RAGResult(BaseModel):
    """Structured outcome of a retrieval query -- evidence, not an answer.

    ``has_evidence`` is False whenever no chunk cleared the similarity
    threshold (including an empty index or a corpus with no relevant
    content). A future LLM synthesis node should treat that as a signal to
    decline answering or ask for clarification, rather than being handed
    empty/irrelevant context and risking a hallucinated response.
    """

    query: str = Field(..., description="The original retrieval query.")
    results: list[RetrievedChunk] = Field(
        default_factory=list, description="Ranked evidence chunks, highest score first."
    )
    has_evidence: bool = Field(
        ..., description="False if no chunk met the similarity threshold for this query."
    )
    top_k: int = Field(..., description="The top_k value used for this query.")
    score_threshold: float = Field(..., description="The similarity threshold used for this query.")
    message: str | None = Field(
        default=None,
        description="Human-readable explanation when has_evidence is False.",
    )


class RAGService:
    """Coordinates the full RAG pipeline: chunk -> embed -> index -> retrieve.

    Usage::

        service = RAGService()
        n_indexed = service.index_documents(normalized_docs)
        result = service.retrieve("What is the revenue?")
        if result.has_evidence:
            ...  # hand result.results to a future LLM synthesis node
    """

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: FAISSVectorStore | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self._embedding_service = embedding_service or EmbeddingService()
        self._chunker = chunker or DocumentChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # Sized from EmbeddingService.dimension, which does not force a model
        # load (see EmbeddingService docstring). If a non-default embedding
        # model with a different dimension is configured, pass an explicitly
        # sized FAISSVectorStore here -- otherwise a dimension mismatch will
        # surface as a clear RAGRetrievalError on first indexing call.
        self._vector_store = vector_store or FAISSVectorStore(
            dimension=self._embedding_service.dimension
        )
        # Tracks chunk_id -> DocumentChunk for potential future introspection.
        self._chunk_registry: dict[str, DocumentChunk] = {}

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_documents(self, documents: list[NormalizedDocument]) -> int:
        """Chunks, embeds, and indexes documents into the FAISS vector store.

        Args:
            documents: NormalizedDocuments to ingest. An empty list, or
                       documents whose content is empty/whitespace, are
                       silently skipped (0 chunks produced, no error).

        Returns:
            Total number of chunks successfully indexed.

        Raises:
            InvalidInputError: Propagated from EmbeddingService/FAISSVectorStore
                                for malformed input (e.g. all-blank content).
            EmbeddingGenerationError: Propagated as-is from EmbeddingService if
                                      the embedding backend is unavailable or fails.
            RAGRetrievalError: Propagated as-is from FAISSVectorStore if indexing
                                fails (e.g. a dimension mismatch).
        """
        if not documents:
            return 0

        chunks = self._chunker.chunk_documents(documents)
        if not chunks:
            logger.debug("All documents produced zero chunks (empty content).")
            return 0

        texts = [c.content for c in chunks]

        # Deliberately NOT wrapped in a try/except here: EmbeddingGenerationError
        # and RAGRetrievalError are already well-formed OmniFlowException
        # subclasses with correct status/error codes. Re-wrapping them into a
        # generic exception would only destroy that information.
        embeddings = self._embedding_service.embed_texts(texts)
        metadatas = [self._chunk_to_metadata(chunk) for chunk in chunks]
        self._vector_store.add_texts(texts, metadatas, embeddings=embeddings)

        for chunk in chunks:
            self._chunk_registry[chunk.chunk_id] = chunk

        logger.info("Indexed %d chunks from %d documents.", len(chunks), len(documents))
        return len(chunks)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RAGResult:
        """Encodes the query, searches FAISS, filters, dedupes, and ranks evidence.

        Args:
            query: Free-text retrieval query. Must be non-empty (an empty or
                   blank query is a caller error, not "no evidence" -- see
                   Raises below).
            top_k: Maximum number of candidate chunks considered, before
                   threshold filtering and deduplication. Defaults to
                   ``settings.rag_top_k``.
            score_threshold: Minimum cosine similarity (0.0-1.0) for a chunk
                             to be treated as relevant evidence. Defaults to
                             ``settings.rag_similarity_threshold``.

        Returns:
            A RAGResult. ``has_evidence`` is False (with an empty ``results``
            list and an explanatory ``message``) if nothing is indexed yet,
            or if nothing indexed clears the similarity threshold for this
            query -- this is a deliberate refusal to hand a future LLM
            synthesis step irrelevant context.

        Raises:
            InvalidInputError: If query is empty/blank, top_k is provided but
                                is not a positive integer, or score_threshold
                                is provided but falls outside [0.0, 1.0].
                                Validated by this method directly so callers
                                invoking RAGService.retrieve() outside of the
                                ToolRegistry/Pydantic boundary (e.g. directly,
                                or from a future orchestrator) still get a
                                clear, typed rejection rather than a confusing
                                downstream failure.
            EmbeddingGenerationError: Propagated as-is if query embedding fails.
            RAGRetrievalError: Propagated as-is if FAISS search fails.
        """
        if not query or not query.strip():
            raise InvalidInputError("query must be a non-empty string.")
        if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
            raise InvalidInputError(
                f"top_k must be a positive integer, got {top_k!r}.",
                details={"top_k": str(top_k)},
            )
        if score_threshold is not None and not (0.0 <= score_threshold <= 1.0):
            raise InvalidInputError(
                f"score_threshold must be between 0.0 and 1.0, got {score_threshold!r}.",
                details={"score_threshold": str(score_threshold)},
            )

        settings = get_settings()
        resolved_top_k = top_k if top_k is not None else settings.rag_top_k
        resolved_threshold = (
            score_threshold if score_threshold is not None else settings.rag_similarity_threshold
        )

        # query blankness is already rejected above; EmbeddingService's own
        # check remains as a defense-in-depth backstop, not the primary guard.
        query_vector = self._embedding_service.embed_query(query)

        if self._vector_store.count() == 0:
            logger.debug("Vector store is empty; returning no-evidence result.")
            return RAGResult(
                query=query,
                results=[],
                has_evidence=False,
                top_k=resolved_top_k,
                score_threshold=resolved_threshold,
                message="No documents have been indexed yet.",
            )

        raw_results = self._vector_store.search(query_vector, top_k=resolved_top_k)

        above_threshold = [r for r in raw_results if r["score"] >= resolved_threshold]
        deduped = self._deduplicate(above_threshold)
        retrieved = [self._to_retrieved_chunk(r) for r in deduped]

        if not retrieved:
            logger.debug(
                "No chunks met the similarity threshold (%.2f) for this query.",
                resolved_threshold,
            )
            return RAGResult(
                query=query,
                results=[],
                has_evidence=False,
                top_k=resolved_top_k,
                score_threshold=resolved_threshold,
                message="No indexed content met the similarity threshold for this query.",
            )

        logger.debug(
            "Retrieved %d evidence chunk(s) for query (len=%d, threshold=%.2f).",
            len(retrieved),
            len(query),
            resolved_threshold,
        )
        return RAGResult(
            query=query,
            results=retrieved,
            has_evidence=True,
            top_k=resolved_top_k,
            score_threshold=resolved_threshold,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clears the vector store and chunk registry (useful for testing)."""
        self._vector_store.clear()
        self._chunk_registry.clear()

    @property
    def indexed_chunk_count(self) -> int:
        """Returns the number of chunks currently in the vector store."""
        return self._vector_store.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_to_metadata(self, chunk: DocumentChunk) -> dict[str, Any]:
        """Builds the FAISS-store metadata payload for one chunk.

        ``chunk_metadata`` nests whatever the chunker inherited from the
        source NormalizedDocument (e.g. a PDF's per-page list, or a
        YouTube-derived document's video_id/language_code) so it survives
        the round trip through the vector store untouched.
        """
        return {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "filename": chunk.filename,
            "source_type": chunk.source_type,
            "extraction_method": chunk.extraction_method,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "chunk_metadata": chunk.metadata,
        }

    def _to_retrieved_chunk(self, raw_result: dict[str, Any]) -> RetrievedChunk:
        meta = raw_result.get("metadata", {})
        return RetrievedChunk(
            chunk_id=meta.get("chunk_id", ""),
            document_id=meta.get("document_id", ""),
            filename=meta.get("filename", ""),
            source_type=meta.get("source_type", ""),
            extraction_method=meta.get("extraction_method", ""),
            content=raw_result["text"],
            score=raw_result["score"],
            chunk_index=meta.get("chunk_index", 0),
            total_chunks=meta.get("total_chunks", 0),
            metadata=meta.get("chunk_metadata", {}),
        )

    def _deduplicate(self, raw_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drops near-duplicate evidence, keeping the highest-scoring copy.

        Assumes ``raw_results`` is already ranked highest-score-first (as
        FAISSVectorStore.search() returns); iterates greedily, keeping a
        candidate only if it is not a near-duplicate of anything already
        kept, so relevance ordering among kept results is preserved exactly.
        """
        kept: list[dict[str, Any]] = []
        for candidate in raw_results:
            is_duplicate = any(
                SequenceMatcher(None, candidate["text"], existing["text"]).ratio()
                >= _DEDUP_SIMILARITY_RATIO
                for existing in kept
            )
            if not is_duplicate:
                kept.append(candidate)
        return kept
