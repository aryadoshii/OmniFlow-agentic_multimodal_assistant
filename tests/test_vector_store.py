"""Tests for FAISSVectorStore — indexing, search ordering, top-k, and edge cases."""

import numpy as np
import pytest

from omniflow.exceptions import InvalidInputError, RAGRetrievalError
from omniflow.rag.vector_store import FAISSVectorStore


def _unit_vec(dim: int = 384, seed: int = 0) -> np.ndarray:
    """Returns a deterministic L2-normalized float32 vector."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).reshape(1, -1)


class TestInit:
    def test_starts_empty(self):
        store = FAISSVectorStore(dimension=384)
        assert store.count() == 0

    def test_custom_dimension(self):
        store = FAISSVectorStore(dimension=128)
        assert store._dimension == 128


class TestAddTexts:
    def test_add_returns_string_ids(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:2]
        ids = store.add_texts(["hello", "world"], embeddings=vecs)
        assert ids == ["0", "1"]

    def test_count_increases(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:3]
        store.add_texts(["a", "b", "c"], embeddings=vecs)
        assert store.count() == 3

    def test_empty_texts_returns_empty(self):
        store = FAISSVectorStore(dimension=4)
        ids = store.add_texts([])
        assert ids == []
        assert store.count() == 0

    def test_metadata_stored(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:1]
        store.add_texts(["text"], [{"source": "doc.pdf"}], embeddings=vecs)
        assert store._store[0]["metadata"]["source"] == "doc.pdf"

    def test_metadata_length_mismatch_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:2]
        with pytest.raises(InvalidInputError) as exc_info:
            store.add_texts(["a", "b"], [{"only": "one"}], embeddings=vecs)
        assert exc_info.value.status_code == 400

    def test_non_dict_metadata_entry_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:1]
        with pytest.raises(InvalidInputError):
            store.add_texts(["a"], ["not a dict"], embeddings=vecs)  # type: ignore[list-item]

    def test_embeddings_row_count_mismatch_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=4)
        vecs = np.eye(4, dtype=np.float32)[:1]  # only 1 row for 2 texts
        with pytest.raises(InvalidInputError):
            store.add_texts(["a", "b"], embeddings=vecs)

    def test_embedding_dimension_mismatch_raises_rag_retrieval_error(self):
        store = FAISSVectorStore(dimension=384)
        wrong_dim_vecs = np.zeros((1, 128), dtype=np.float32)
        with pytest.raises(RAGRetrievalError) as exc_info:
            store.add_texts(["a"], embeddings=wrong_dim_vecs)
        assert exc_info.value.error_code == "RAG_RETRIEVAL_ERROR"
        assert store.count() == 0  # index must remain untouched on failure

    def test_add_texts_without_embeddings_stores_but_does_not_index(self):
        """Text/metadata can be stored without vectors; count() stays 0 since
        nothing was actually added to the FAISS index itself."""
        store = FAISSVectorStore(dimension=4)
        ids = store.add_texts(["a", "b"])
        assert ids == ["0", "1"]
        assert store.count() == 0


class TestSearch:
    def test_empty_index_returns_empty(self):
        store = FAISSVectorStore(dimension=384)
        result = store.search(_unit_vec(), top_k=4)
        assert result == []

    def test_search_returns_correct_shape(self):
        store = FAISSVectorStore(dimension=384)
        vecs = np.vstack([_unit_vec(seed=i) for i in range(5)])
        texts = [f"doc {i}" for i in range(5)]
        store.add_texts(texts, embeddings=vecs)

        results = store.search(_unit_vec(seed=0), top_k=3)
        assert len(results) == 3

    def test_top_k_capped_at_index_size(self):
        store = FAISSVectorStore(dimension=384)
        vecs = np.vstack([_unit_vec(seed=i) for i in range(2)])
        store.add_texts(["a", "b"], embeddings=vecs)
        results = store.search(_unit_vec(seed=0), top_k=10)
        assert len(results) == 2

    def test_self_query_highest_score(self):
        """Querying with an exact stored vector should return it as the top result."""
        store = FAISSVectorStore(dimension=384)
        vec = _unit_vec(seed=42)
        other_vecs = np.vstack([_unit_vec(seed=i) for i in range(5)])
        all_vecs = np.vstack([vec, other_vecs])
        store.add_texts(["exact"] + [f"other {i}" for i in range(5)], embeddings=all_vecs)

        results = store.search(vec, top_k=1)
        assert results[0]["text"] == "exact"
        assert results[0]["score"] > 0.99

    def test_scores_descending(self):
        store = FAISSVectorStore(dimension=384)
        vecs = np.vstack([_unit_vec(seed=i) for i in range(4)])
        store.add_texts([f"doc {i}" for i in range(4)], embeddings=vecs)
        results = store.search(_unit_vec(seed=0), top_k=4)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_result_has_expected_keys(self):
        store = FAISSVectorStore(dimension=384)
        vecs = _unit_vec(seed=1)
        store.add_texts(["hello"], [{"k": "v"}], embeddings=vecs)
        results = store.search(_unit_vec(seed=1), top_k=1)
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "score" in results[0]

    def test_string_query_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=4)
        with pytest.raises(InvalidInputError, match="pre-embedded"):
            store.search("plain text query")

    def test_zero_top_k_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=384)
        store.add_texts(["a"], embeddings=_unit_vec())
        with pytest.raises(InvalidInputError):
            store.search(_unit_vec(), top_k=0)

    def test_negative_top_k_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=384)
        store.add_texts(["a"], embeddings=_unit_vec())
        with pytest.raises(InvalidInputError):
            store.search(_unit_vec(), top_k=-1)

    def test_default_top_k_comes_from_settings(self):
        """With top_k omitted, the number of returned results is bounded by
        settings.rag_top_k (default 4), not an unlimited/arbitrary count."""
        store = FAISSVectorStore(dimension=384)
        vecs = np.vstack([_unit_vec(seed=i) for i in range(10)])
        store.add_texts([f"doc {i}" for i in range(10)], embeddings=vecs)

        results = store.search(_unit_vec(seed=0))
        assert len(results) == 4

    def test_query_dimension_mismatch_raises_rag_retrieval_error(self):
        store = FAISSVectorStore(dimension=384)
        store.add_texts(["a"], embeddings=_unit_vec())
        wrong_dim_query = np.zeros((1, 128), dtype=np.float32)
        with pytest.raises(RAGRetrievalError) as exc_info:
            store.search(wrong_dim_query, top_k=1)
        assert exc_info.value.error_code == "RAG_RETRIEVAL_ERROR"

    def test_empty_query_array_raises_invalid_input_error(self):
        store = FAISSVectorStore(dimension=384)
        store.add_texts(["a"], embeddings=_unit_vec())
        with pytest.raises(InvalidInputError):
            store.search(np.array([], dtype=np.float32), top_k=1)

    def test_search_on_empty_index_skips_dimension_check(self):
        """An empty index returns [] immediately without validating query shape,
        since there is nothing to compare against yet."""
        store = FAISSVectorStore(dimension=384)
        result = store.search(np.zeros((1, 128), dtype=np.float32), top_k=4)
        assert result == []


class TestClear:
    def test_clear_resets_index(self):
        store = FAISSVectorStore(dimension=384)
        vecs = np.vstack([_unit_vec(seed=i) for i in range(3)])
        store.add_texts(["a", "b", "c"], embeddings=vecs)
        assert store.count() == 3
        store.clear()
        assert store.count() == 0
        assert len(store._store) == 0

    def test_search_after_clear_returns_empty(self):
        store = FAISSVectorStore(dimension=384)
        vecs = _unit_vec(seed=0)
        store.add_texts(["x"], embeddings=vecs)
        store.clear()
        results = store.search(_unit_vec(seed=0), top_k=4)
        assert results == []
