"""Tests for RAGSearchTool and its integration into ToolRegistry (Phase 3.6).

RAGService itself is mocked here (already comprehensively tested in
test_rag_service.py) -- these tests focus purely on the tool wrapper's
contract: input validation, output shaping, exception propagation through
the registry, and coexistence with YouTubeTranscriptTool in one registry.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from backend.exceptions import (
    EmbeddingGenerationError,
    InvalidInputError,
    RAGRetrievalError,
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
)
from backend.models.document import ExtractionMethod, NormalizedDocument, SourceType
from backend.rag.service import RAGResult, RetrievedChunk
from backend.tools.rag_search import RAGSearchInput, RAGSearchOutput, RAGSearchTool
from backend.tools.registry import ToolRegistry
from backend.tools.youtube import YouTubeTranscriptTool


def _make_document(content: str = "Some document content.") -> NormalizedDocument:
    return NormalizedDocument(
        filename="doc.txt",
        source_type=SourceType.TEXT,
        mime_type="text/plain",
        content=content,
        extraction_method=ExtractionMethod.DIRECT_INPUT,
    )


def _make_chunk(content: str = "Evidence text.", score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        filename="doc.txt",
        source_type="text",
        extraction_method="direct_input",
        content=content,
        score=score,
        chunk_index=0,
        total_chunks=1,
        metadata={"note": "example"},
    )


def _mock_rag_service_returning(result: RAGResult) -> MagicMock:
    service = MagicMock()
    service.retrieve.return_value = result
    return service


def _mock_rag_service_raising(exc: BaseException) -> MagicMock:
    service = MagicMock()
    service.retrieve.side_effect = exc
    return service


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestRAGSearchInput:
    def test_valid_query_constructs(self) -> None:
        model = RAGSearchInput(query="find revenue figures")
        assert model.top_k is None
        assert model.score_threshold is None

    def test_empty_query_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RAGSearchInput(query="")

    def test_whitespace_only_query_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            RAGSearchInput(query="   ")

    def test_explicit_top_k_and_threshold_preserved(self) -> None:
        model = RAGSearchInput(query="q", top_k=10, score_threshold=0.5)
        assert model.top_k == 10
        assert model.score_threshold == 0.5

    def test_out_of_range_threshold_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RAGSearchInput(query="q", score_threshold=1.5)

    def test_zero_top_k_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RAGSearchInput(query="q", top_k=0)


# ---------------------------------------------------------------------------
# Successful execution / structured output
# ---------------------------------------------------------------------------


class TestSuccessfulExecution:
    def test_evidence_found_produces_structured_output(self) -> None:
        chunk = _make_chunk()
        rag_result = RAGResult(
            query="revenue",
            results=[chunk],
            has_evidence=True,
            top_k=4,
            score_threshold=0.2,
        )
        rag_service = _mock_rag_service_returning(rag_result)
        tool = RAGSearchTool(rag_service)

        output = tool.run(RAGSearchInput(query="revenue"))

        assert isinstance(output, BaseModel)
        assert isinstance(output, RAGSearchOutput)
        assert output.status == "evidence_found"
        assert output.evidence_count == 1
        assert output.evidence[0] == chunk
        assert output.top_k == 4
        assert output.score_threshold == 0.2
        assert output.message is None

    def test_no_evidence_produces_structured_no_evidence_status(self) -> None:
        rag_result = RAGResult(
            query="unrelated topic",
            results=[],
            has_evidence=False,
            top_k=4,
            score_threshold=0.2,
            message="No indexed content met the similarity threshold for this query.",
        )
        rag_service = _mock_rag_service_returning(rag_result)
        tool = RAGSearchTool(rag_service)

        output = tool.run(RAGSearchInput(query="unrelated topic"))

        assert output.status == "no_evidence"
        assert output.evidence == []
        assert output.evidence_count == 0
        assert output.message is not None

    def test_forwards_top_k_and_threshold_to_rag_service(self) -> None:
        rag_result = RAGResult(query="q", results=[], has_evidence=False, top_k=7, score_threshold=0.6)
        rag_service = _mock_rag_service_returning(rag_result)
        tool = RAGSearchTool(rag_service)

        tool.run(RAGSearchInput(query="q", top_k=7, score_threshold=0.6))

        rag_service.retrieve.assert_called_once_with(query="q", top_k=7, score_threshold=0.6)

    def test_tool_metadata(self) -> None:
        tool = RAGSearchTool(MagicMock())
        assert tool.name == "rag_search"
        assert tool.input_model is RAGSearchInput
        assert "evidence" in tool.description.lower()
        assert "llm" in tool.description.lower()


# ---------------------------------------------------------------------------
# Lazy indexing (Phase 5 performance fix)
# ---------------------------------------------------------------------------


class TestLazyIndexing:
    """RAGSearchTool must index its documents lazily, on its own first
    run() call, at most once per instance -- never at construction time.
    This is what lets a direct-context request avoid loading the embedding
    backend (sentence-transformers/Torch) just because files were uploaded;
    see backend/api/routes/agent.py's _build_tool_registry()."""

    def _tool_with_no_evidence(self, documents: list[NormalizedDocument] | None) -> tuple[RAGSearchTool, MagicMock]:
        rag_service = _mock_rag_service_returning(
            RAGResult(query="q", results=[], has_evidence=False, top_k=4, score_threshold=0.2, message="none")
        )
        return RAGSearchTool(rag_service, documents), rag_service

    def test_documents_not_indexed_at_construction(self) -> None:
        rag_service = MagicMock()
        RAGSearchTool(rag_service, [_make_document()])
        rag_service.index_documents.assert_not_called()

    def test_first_run_indexes_the_given_documents_before_retrieving(self) -> None:
        docs = [_make_document("Quarterly revenue rose 15%.")]
        tool, rag_service = self._tool_with_no_evidence(docs)

        tool.run(RAGSearchInput(query="revenue"))

        rag_service.index_documents.assert_called_once_with(docs)
        call_order = [c[0] for c in rag_service.method_calls]
        assert call_order.index("index_documents") < call_order.index("retrieve")

    def test_second_run_on_the_same_instance_does_not_reindex(self) -> None:
        tool, rag_service = self._tool_with_no_evidence([_make_document()])

        tool.run(RAGSearchInput(query="first question"))
        tool.run(RAGSearchInput(query="second question"))

        rag_service.index_documents.assert_called_once()
        assert rag_service.retrieve.call_count == 2

    def test_no_documents_provided_indexes_an_empty_list_not_none(self) -> None:
        """Constructing without documents (the common case for a request
        with no uploaded files, or existing callers that never pass any)
        must still index an empty list -- a cheap, explicit no-op per
        RAGService.index_documents()'s own contract -- never skip the call
        or pass None."""
        tool, rag_service = self._tool_with_no_evidence(None)

        tool.run(RAGSearchInput(query="q"))

        rag_service.index_documents.assert_called_once_with([])

    def test_indexing_failure_propagates_and_a_later_retry_indexes_again(self) -> None:
        """If indexing itself fails, the failure must propagate like any
        other domain exception (never silently swallowed into a
        no-evidence result), and must not be mistaken for 'already
        indexed' -- a later retry attempt (e.g. a subsequent replan cycle
        calling rag_search again) must try indexing again rather than
        proceeding straight to retrieve() against an unindexed corpus."""
        rag_service = MagicMock()
        rag_service.index_documents.side_effect = [EmbeddingGenerationError("backend down"), None]
        rag_service.retrieve.return_value = RAGResult(
            query="q", results=[], has_evidence=False, top_k=4, score_threshold=0.2, message="none"
        )
        tool = RAGSearchTool(rag_service, [_make_document()])

        with pytest.raises(EmbeddingGenerationError):
            tool.run(RAGSearchInput(query="q"))
        rag_service.retrieve.assert_not_called()

        tool.run(RAGSearchInput(query="q"))  # retry succeeds
        assert rag_service.index_documents.call_count == 2
        rag_service.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# Failure propagation
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    def test_embedding_failure_propagates_unwrapped(self) -> None:
        rag_service = _mock_rag_service_raising(EmbeddingGenerationError("backend down"))
        tool = RAGSearchTool(rag_service)

        with pytest.raises(EmbeddingGenerationError):
            tool.run(RAGSearchInput(query="revenue"))

    def test_vector_store_failure_propagates_unwrapped(self) -> None:
        rag_service = _mock_rag_service_raising(RAGRetrievalError("faiss failed"))
        tool = RAGSearchTool(rag_service)

        with pytest.raises(RAGRetrievalError):
            tool.run(RAGSearchInput(query="revenue"))


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_registration_success(self) -> None:
        registry = ToolRegistry()
        registry.register(RAGSearchTool(MagicMock()))
        assert "rag_search" in registry
        assert len(registry) == 1

    def test_duplicate_registration_raises(self) -> None:
        registry = ToolRegistry()
        registry.register(RAGSearchTool(MagicMock()))
        with pytest.raises(ToolAlreadyRegisteredError):
            registry.register(RAGSearchTool(MagicMock()))

    def test_execute_success_via_registry(self) -> None:
        chunk = _make_chunk()
        rag_result = RAGResult(query="revenue", results=[chunk], has_evidence=True, top_k=4, score_threshold=0.2)
        registry = ToolRegistry()
        registry.register(RAGSearchTool(_mock_rag_service_returning(rag_result)))

        output, trace = registry.execute("rag_search", query="revenue")

        assert isinstance(output, RAGSearchOutput)
        assert output.status == "evidence_found"
        assert trace.status == "success"
        assert trace.tool_name == "rag_search"

    def test_execute_no_evidence_via_registry(self) -> None:
        rag_result = RAGResult(query="q", results=[], has_evidence=False, top_k=4, score_threshold=0.2, message="none")
        registry = ToolRegistry()
        registry.register(RAGSearchTool(_mock_rag_service_returning(rag_result)))

        output, trace = registry.execute("rag_search", query="q")

        assert output.status == "no_evidence"
        assert trace.status == "success"  # a "no evidence" result is still a successful tool run

    def test_execute_invalid_query_raises_invalid_input_error(self) -> None:
        registry = ToolRegistry()
        registry.register(RAGSearchTool(MagicMock()))

        with pytest.raises(InvalidInputError):
            registry.execute("rag_search", query="")

    def test_execute_missing_query_raises_invalid_input_error(self) -> None:
        registry = ToolRegistry()
        registry.register(RAGSearchTool(MagicMock()))

        with pytest.raises(InvalidInputError):
            registry.execute("rag_search")  # missing required 'query' field

    def test_execute_embedding_failure_preserves_type_through_registry(self) -> None:
        """The Phase 3.6 registry fix: a domain exception raised inside a
        tool's run() must reach the caller with its original type intact."""
        registry = ToolRegistry()
        registry.register(RAGSearchTool(_mock_rag_service_raising(EmbeddingGenerationError("boom"))))

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            registry.execute("rag_search", query="revenue")

        assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
        assert "duration_ms" in exc_info.value.details

    def test_execute_retrieval_failure_preserves_type_through_registry(self) -> None:
        registry = ToolRegistry()
        registry.register(RAGSearchTool(_mock_rag_service_raising(RAGRetrievalError("faiss down"))))

        with pytest.raises(RAGRetrievalError) as exc_info:
            registry.execute("rag_search", query="revenue")

        assert exc_info.value.error_code == "RAG_RETRIEVAL_ERROR"

    def test_execute_nonexistent_tool_raises_tool_not_found(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError):
            registry.execute("rag_search", query="revenue")


# ---------------------------------------------------------------------------
# Registry now exposes both tools together
# ---------------------------------------------------------------------------


class TestToolDiscovery:
    def test_registry_exposes_both_youtube_and_rag_search(self) -> None:
        registry = ToolRegistry()
        registry.register(YouTubeTranscriptTool())
        registry.register(RAGSearchTool(MagicMock()))

        listing = registry.list_tools()
        names = {entry["name"] for entry in listing}

        assert names == {"youtube_transcript", "rag_search"}
        for entry in listing:
            assert set(entry.keys()) == {"name", "description"}
            assert entry["description"]  # non-empty

    def test_both_tools_independently_executable_in_same_registry(self) -> None:
        registry = ToolRegistry()
        registry.register(YouTubeTranscriptTool())
        rag_result = RAGResult(query="q", results=[_make_chunk()], has_evidence=True, top_k=4, score_threshold=0.2)
        registry.register(RAGSearchTool(_mock_rag_service_returning(rag_result)))

        rag_output, _ = registry.execute("rag_search", query="q")
        assert rag_output.status == "evidence_found"

        # youtube_transcript tool is present and its own input validation
        # runs independently of rag_search's presence in the same registry.
        with pytest.raises(InvalidInputError):
            registry.execute("youtube_transcript", url="https://example.com/not-youtube")
