"""Tests for RAGService (Phase 3.5) — indexing, retrieval, thresholding, dedup.

All unit tests use a real EmbeddingService with its underlying ML model
mocked out (only the SentenceTransformer.encode() boundary is faked, via a
controllable text->vector map) so that real validation logic (blank-query
checks, etc.) still runs, per this project's "mock only the external
boundary" convention. FAISSVectorStore is used for real -- it's lightweight,
deterministic, and already covered independently in test_vector_store.py.

A separate, opt-in integration test at the bottom exercises the real
embedding model end-to-end and is skipped unless explicitly enabled.
"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from omniflow.exceptions import (
    EmbeddingGenerationError,
    InvalidInputError,
    RAGRetrievalError,
)
from omniflow.models.document import ExtractionMethod, NormalizedDocument, SourceType
from omniflow.rag.embeddings import EmbeddingService
from omniflow.rag.service import RAGResult, RAGService, RetrievedChunk
from omniflow.rag.vector_store import FAISSVectorStore

DIM = 4


def _unit(vec: list[float]) -> np.ndarray:
    v = np.array(vec, dtype=np.float32)
    return v / np.linalg.norm(v)


def _scripted_embedding_service(
    vector_map: dict[str, list[float]], dimension: int = DIM
) -> EmbeddingService:
    """Real EmbeddingService with only the SentenceTransformer boundary mocked.

    Texts not present in vector_map get a deterministic (hash-seeded)
    fallback vector, so unspecified chunk text never raises a KeyError.
    """
    service = EmbeddingService()
    fake_model = MagicMock()
    fake_model.get_embedding_dimension.return_value = dimension
    fake_model.get_sentence_embedding_dimension.return_value = dimension

    def _encode(texts, **kwargs):
        vecs = []
        for text in texts:
            if text in vector_map:
                vecs.append(_unit(vector_map[text]))
            else:
                rng = np.random.default_rng(abs(hash(text)) % (2**32))
                v = rng.random(dimension).astype(np.float32)
                vecs.append(v / np.linalg.norm(v))
        return np.vstack(vecs).astype(np.float32)

    fake_model.encode.side_effect = _encode
    service._model = fake_model
    return service


def _doc(
    content: str,
    filename: str = "doc.txt",
    source_type: SourceType = SourceType.TEXT,
    extraction_method: ExtractionMethod = ExtractionMethod.DIRECT_INPUT,
    metadata: dict | None = None,
) -> NormalizedDocument:
    return NormalizedDocument(
        filename=filename,
        source_type=source_type,
        mime_type="text/plain",
        content=content,
        extraction_method=extraction_method,
        metadata=metadata or {},
    )


RECIPE_TEXT = "Apple pie recipe with cinnamon and sugar."
BREAD_TEXT = "Banana bread recipe with walnuts."
CAR_TEXT = "Car engine repair manual and troubleshooting guide."
MID_TEXT = "A moderately related note about baking times."


def _standard_service() -> RAGService:
    """A RAGService pre-wired with a fixed, hand-crafted similarity geometry:
    RECIPE and BREAD are near-parallel (highly similar); CAR is orthogonal
    to both; MID sits at a known intermediate similarity to a recipe query."""
    vector_map = {
        RECIPE_TEXT: [1.0, 0.0, 0.0, 0.0],
        BREAD_TEXT: [0.98, 0.2, 0.0, 0.0],
        CAR_TEXT: [0.0, 1.0, 0.0, 0.0],
        MID_TEXT: [0.6, 0.8, 0.0, 0.0],  # cosine sim to [1,0,0,0] query is exactly 0.6
        "apple pie recipe": [1.0, 0.0, 0.0, 0.0],
        "space travel and astronomy": [0.0, 0.0, 0.0, 1.0],
    }
    embedding_service = _scripted_embedding_service(vector_map)
    vector_store = FAISSVectorStore(dimension=DIM)
    return RAGService(embedding_service=embedding_service, vector_store=vector_store)


# ---------------------------------------------------------------------------
# Construction / laziness
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_construction_does_not_load_embedding_model(self) -> None:
        service = RAGService()
        assert not service._embedding_service.is_loaded

    def test_starts_with_empty_index(self) -> None:
        service = _standard_service()
        assert service.indexed_chunk_count == 0


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


class TestIndexDocuments:
    def test_index_one_document(self) -> None:
        service = _standard_service()
        n = service.index_documents([_doc(RECIPE_TEXT, filename="recipe.txt")])
        assert n == 1
        assert service.indexed_chunk_count == 1

    def test_index_multiple_documents(self) -> None:
        service = _standard_service()
        docs = [
            _doc(RECIPE_TEXT, filename="recipe.txt"),
            _doc(BREAD_TEXT, filename="bread.txt"),
            _doc(CAR_TEXT, filename="car.pdf", source_type=SourceType.PDF, extraction_method=ExtractionMethod.NATIVE_TEXT),
        ]
        n = service.index_documents(docs)
        assert n == 3
        assert service.indexed_chunk_count == 3

    def test_index_empty_list_is_noop(self) -> None:
        service = _standard_service()
        assert service.index_documents([]) == 0
        assert service.indexed_chunk_count == 0

    def test_index_document_with_empty_content_is_noop(self) -> None:
        service = _standard_service()
        n = service.index_documents([_doc("   ")])
        assert n == 0
        assert service.indexed_chunk_count == 0

    def test_embedding_failure_propagates_unwrapped(self) -> None:
        """EmbeddingGenerationError must propagate as-is, not be re-wrapped
        into a generic exception -- doing so would destroy its status/error code."""
        embedding_service = MagicMock()
        embedding_service.dimension = DIM
        embedding_service.embed_texts.side_effect = EmbeddingGenerationError("boom")
        service = RAGService(embedding_service=embedding_service, vector_store=FAISSVectorStore(dimension=DIM))

        with pytest.raises(EmbeddingGenerationError):
            service.index_documents([_doc(RECIPE_TEXT)])

    def test_vector_store_failure_propagates_unwrapped(self) -> None:
        """RAGRetrievalError from the vector store must propagate as-is."""
        embedding_service = _scripted_embedding_service({RECIPE_TEXT: [1, 0, 0, 0]})
        broken_store = MagicMock()
        broken_store.add_texts.side_effect = RAGRetrievalError("dimension mismatch")
        service = RAGService(embedding_service=embedding_service, vector_store=broken_store)

        with pytest.raises(RAGRetrievalError):
            service.index_documents([_doc(RECIPE_TEXT)])

    def test_malformed_but_structurally_valid_metadata_survives(self) -> None:
        """Odd-shaped (nested list/bool/int) source metadata must round-trip
        through indexing without raising."""
        service = _standard_service()
        doc = _doc(
            RECIPE_TEXT,
            filename="weird.pdf",
            source_type=SourceType.PDF,
            extraction_method=ExtractionMethod.MIXED,
            metadata={"pages": [{"page_number": 1, "ok": True}], "count": 3, "flag": False},
        )
        n = service.index_documents([doc])
        assert n == 1


# ---------------------------------------------------------------------------
# Retrieval — relevance, threshold, top-k
# ---------------------------------------------------------------------------


class TestRetrieveRelevance:
    def test_relevant_retrieval_returns_evidence(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT, filename="recipe.txt")])

        result = service.retrieve("apple pie recipe")

        assert isinstance(result, RAGResult)
        assert result.has_evidence is True
        assert len(result.results) == 1
        assert isinstance(result.results[0], RetrievedChunk)
        assert result.results[0].content == RECIPE_TEXT
        assert result.results[0].score == pytest.approx(1.0, abs=1e-4)

    def test_irrelevant_query_returns_insufficient_evidence(self) -> None:
        service = _standard_service()
        service.index_documents(
            [_doc(RECIPE_TEXT, filename="recipe.txt"), _doc(CAR_TEXT, filename="car.pdf")]
        )

        result = service.retrieve("space travel and astronomy")

        assert result.has_evidence is False
        assert result.results == []
        assert result.message is not None
        assert "threshold" in result.message.lower()

    def test_empty_index_returns_insufficient_evidence(self) -> None:
        service = _standard_service()
        result = service.retrieve("apple pie recipe")
        assert result.has_evidence is False
        assert "indexed" in result.message.lower()

    def test_threshold_filters_mid_relevance_chunk(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(MID_TEXT, filename="mid.txt")])

        low_threshold_result = service.retrieve("apple pie recipe", score_threshold=0.5)
        assert low_threshold_result.has_evidence is True

        high_threshold_result = service.retrieve("apple pie recipe", score_threshold=0.7)
        assert high_threshold_result.has_evidence is False

    def test_top_k_limits_result_count(self) -> None:
        service = _standard_service()
        docs = [_doc(f"Unique document body number {i} about recipes.", filename=f"d{i}.txt") for i in range(5)]
        service.index_documents(docs)

        result = service.retrieve("apple pie recipe", top_k=2, score_threshold=0.0)
        assert len(result.results) <= 2

    def test_default_top_k_and_threshold_come_from_settings(self) -> None:
        from omniflow.config import get_settings

        settings = get_settings()
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT, filename="recipe.txt")])
        result = service.retrieve("apple pie recipe")
        assert result.top_k == settings.rag_top_k
        assert result.score_threshold == settings.rag_similarity_threshold


# ---------------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    def test_empty_query_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError):
            service.retrieve("")

    def test_whitespace_query_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError):
            service.retrieve("   ")


class TestDirectServiceLevelValidation:
    """Regression coverage: RAGService.retrieve() must validate its own
    parameters when called directly, not only when invoked through the
    ToolRegistry/RAGSearchInput Pydantic boundary."""

    def test_zero_top_k_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError) as exc_info:
            service.retrieve("apple pie recipe", top_k=0)
        assert exc_info.value.status_code == 400

    def test_negative_top_k_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError):
            service.retrieve("apple pie recipe", top_k=-1)

    def test_non_integer_top_k_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError):
            service.retrieve("apple pie recipe", top_k=2.5)  # type: ignore[arg-type]

    def test_negative_score_threshold_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError) as exc_info:
            service.retrieve("apple pie recipe", score_threshold=-0.1)
        assert exc_info.value.status_code == 400

    def test_score_threshold_above_one_raises_invalid_input_error(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        with pytest.raises(InvalidInputError):
            service.retrieve("apple pie recipe", score_threshold=1.1)

    def test_boundary_threshold_values_are_accepted(self) -> None:
        """0.0 and 1.0 are valid (inclusive) boundary values, not rejected."""
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        service.retrieve("apple pie recipe", score_threshold=0.0)
        service.retrieve("apple pie recipe", score_threshold=1.0)

    def test_valid_direct_call_still_works(self) -> None:
        """Regression guard: legitimate direct calls (bypassing the tool
        registry entirely) must be unaffected by the new validation."""
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        result = service.retrieve("apple pie recipe", top_k=2, score_threshold=0.5)
        assert result.has_evidence is True


# ---------------------------------------------------------------------------
# Metadata / source attribution
# ---------------------------------------------------------------------------


class TestSourceAttribution:
    def test_metadata_and_source_fields_preserved(self) -> None:
        service = _standard_service()
        service.index_documents(
            [
                _doc(
                    CAR_TEXT,
                    filename="manual.pdf",
                    source_type=SourceType.PDF,
                    extraction_method=ExtractionMethod.OCR,
                    metadata={"pages": [{"page_number": 1}], "total_pages": 1},
                )
            ]
        )

        # score_threshold=0.0 (the valid minimum) is used here purely to
        # disable relevance filtering for this metadata-preservation check;
        # -1.0 was previously used for this but is no longer a valid
        # score_threshold per the new service-level validation.
        result = service.retrieve("space travel and astronomy", score_threshold=0.0)

        chunk = result.results[0]
        assert chunk.filename == "manual.pdf"
        assert chunk.source_type == "pdf"
        assert chunk.extraction_method == "ocr"
        assert chunk.metadata["total_pages"] == 1
        assert chunk.metadata["pages"][0]["page_number"] == 1

    def test_multiple_documents_correctly_attributed(self) -> None:
        service = _standard_service()
        doc_recipe = _doc(RECIPE_TEXT, filename="recipe.txt")
        doc_car = _doc(CAR_TEXT, filename="car.pdf", source_type=SourceType.PDF)
        service.index_documents([doc_recipe, doc_car])

        result = service.retrieve("apple pie recipe", score_threshold=0.5)

        assert len(result.results) == 1
        assert result.results[0].document_id == doc_recipe.id
        assert result.results[0].filename == "recipe.txt"


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


class TestOrdering:
    def test_results_ranked_by_score_descending(self) -> None:
        service = _standard_service()
        service.index_documents(
            [_doc(RECIPE_TEXT, filename="recipe.txt"), _doc(BREAD_TEXT, filename="bread.txt")]
        )

        result = service.retrieve("apple pie recipe", score_threshold=0.0)

        scores = [r.score for r in result.results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_near_duplicate_chunks_deduplicated(self) -> None:
        text_a = "The quarterly revenue increased significantly this year."
        text_b = "The quarterly revenue increased significantly this year period."

        embedding_service = _scripted_embedding_service(
            {
                text_a: [1.0, 0.0, 0.0, 0.0],
                text_b: [0.999, 0.001, 0.0, 0.0],
                "revenue growth": [1.0, 0.0, 0.0, 0.0],
            }
        )
        service = RAGService(embedding_service=embedding_service, vector_store=FAISSVectorStore(dimension=DIM))
        service.index_documents([_doc(text_a, filename="a.txt"), _doc(text_b, filename="b.txt")])

        result = service.retrieve("revenue growth", score_threshold=0.0)

        assert len(result.results) == 1
        # The higher-scoring (exact match) copy must be the one kept.
        assert result.results[0].content == text_a

    def test_distinct_chunks_not_deduplicated(self) -> None:
        service = _standard_service()
        service.index_documents(
            [_doc(RECIPE_TEXT, filename="recipe.txt"), _doc(BREAD_TEXT, filename="bread.txt")]
        )
        result = service.retrieve("apple pie recipe", score_threshold=0.0)
        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    def test_clear_resets_index(self) -> None:
        service = _standard_service()
        service.index_documents([_doc(RECIPE_TEXT)])
        assert service.indexed_chunk_count == 1
        service.clear()
        assert service.indexed_chunk_count == 0
        result = service.retrieve("apple pie recipe")
        assert result.has_evidence is False


# ---------------------------------------------------------------------------
# Optional / manual integration test with the real embedding model
# ---------------------------------------------------------------------------
#
# To run it manually:
#   OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 pytest tests/test_rag_service.py -k RealModel -v
#
_RUN_INTEGRATION = os.environ.get("OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason=(
        "Real-model RAG integration test skipped by default (requires downloading "
        "sentence-transformers weights). Set OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 to run."
    ),
)
class TestRealModelIntegration:
    def test_real_embeddings_separate_relevant_from_irrelevant(self) -> None:
        service = RAGService()
        service.index_documents(
            [
                _doc(
                    "The company's quarterly revenue grew by twelve percent, driven by strong demand.",
                    filename="earnings.txt",
                ),
                _doc(
                    "The recipe calls for two cups of flour, a teaspoon of salt, and fresh yeast.",
                    filename="recipe.txt",
                ),
            ]
        )

        result = service.retrieve("What was the company's financial performance?")

        assert result.has_evidence is True
        assert result.results[0].filename == "earnings.txt"
