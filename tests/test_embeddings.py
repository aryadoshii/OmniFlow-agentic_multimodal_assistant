"""Tests for EmbeddingService (Phase 3.3).

All tests in the default suite mock the SentenceTransformer boundary
directly -- no model is downloaded and no network access occurs. A
separate, opt-in integration test at the bottom of this file exercises the
real model and is skipped unless explicitly enabled (see
TestRealModelIntegration).
"""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.exceptions import EmbeddingGenerationError, InvalidInputError
from backend.rag.embeddings import EmbeddingService


def _service_with_fake_model(dimension: int = 384, n: int = 1) -> EmbeddingService:
    """Builds an EmbeddingService with a pre-injected mock model (no loading)."""
    service = EmbeddingService()
    fake_model = MagicMock()
    fake_model.encode.return_value = np.zeros((n, dimension), dtype=np.float32)
    fake_model.get_sentence_embedding_dimension.return_value = dimension
    fake_model.get_embedding_dimension.return_value = dimension
    service._model = fake_model
    return service


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------


class TestLazyLoading:
    def test_not_loaded_on_instantiation(self) -> None:
        service = EmbeddingService()
        assert not service.is_loaded

    def test_construction_does_not_import_sentence_transformers(self) -> None:
        """Constructing the service must not trigger the heavy ML import."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # If EmbeddingService.__init__ imported sentence_transformers eagerly,
            # this would raise ImportError immediately.
            service = EmbeddingService()
            assert not service.is_loaded

    def test_is_loaded_after_embed(self) -> None:
        service = _service_with_fake_model()
        service.embed_texts(["hello"])
        assert service.is_loaded

    def test_model_loaded_only_once_and_reused(self) -> None:
        service = _service_with_fake_model(n=1)
        fake_model = service._model
        service.embed_texts(["first"])
        service.embed_texts(["second"])
        assert service._model is fake_model
        assert fake_model.encode.call_count == 2

    def test_get_model_raises_embedding_generation_error_when_not_installed(self) -> None:
        service = EmbeddingService()
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            with pytest.raises(EmbeddingGenerationError) as exc_info:
                service._get_model()
        assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"

    def test_model_load_failure_wrapped_as_embedding_generation_error(self) -> None:
        """Simulates the real library being importable but failing to construct
        the model (e.g. a corrupt cache or unknown model name), without
        triggering a real import of the heavy sentence_transformers package."""
        service = EmbeddingService()
        fake_module = MagicMock()
        fake_module.SentenceTransformer.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            with pytest.raises(EmbeddingGenerationError):
                service._get_model()


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


class TestEmbedTexts:
    def test_output_shape(self) -> None:
        service = _service_with_fake_model(n=3)
        result = service.embed_texts(["a", "b", "c"])
        assert result.shape == (3, 384)

    def test_output_dtype_float32(self) -> None:
        service = _service_with_fake_model()
        result = service.embed_texts(["text"])
        assert result.dtype == np.float32

    def test_empty_list_raises_invalid_input_error(self) -> None:
        service = _service_with_fake_model()
        with pytest.raises(InvalidInputError) as exc_info:
            service.embed_texts([])
        assert exc_info.value.status_code == 400

    def test_all_blank_texts_raise_invalid_input_error(self) -> None:
        service = _service_with_fake_model()
        with pytest.raises(InvalidInputError):
            service.embed_texts(["", "   ", "\t"])

    def test_mixed_blank_and_nonblank_texts_pass_through(self) -> None:
        """A batch with some blank entries is not fully empty and should proceed."""
        service = _service_with_fake_model(n=2)
        result = service.embed_texts(["", "real text"])
        assert result.shape == (2, 384)

    def test_batch_size_and_normalization_kwargs_passed(self) -> None:
        service = _service_with_fake_model(n=2)
        service.embed_texts(["a", "b"])
        call_kwargs = service._model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 32
        assert call_kwargs.get("normalize_embeddings") is True
        assert call_kwargs.get("convert_to_numpy") is True

    def test_custom_batch_size_used(self) -> None:
        service = EmbeddingService(batch_size=8)
        fake_model = MagicMock()
        fake_model.encode.return_value = np.zeros((1, 384), dtype=np.float32)
        service._model = fake_model

        service.embed_texts(["a"])
        assert fake_model.encode.call_args[1]["batch_size"] == 8

    def test_encode_failure_wrapped_as_embedding_generation_error(self) -> None:
        service = EmbeddingService()
        fake_model = MagicMock()
        fake_model.encode.side_effect = RuntimeError("device error")
        service._model = fake_model

        with pytest.raises(EmbeddingGenerationError) as exc_info:
            service.embed_texts(["a"])
        assert exc_info.value.error_code == "EMBEDDING_GENERATION_ERROR"
        assert "text_count" in exc_info.value.details


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


class TestEmbedQuery:
    def test_embed_query_returns_2d_array(self) -> None:
        service = _service_with_fake_model(n=1)
        result = service.embed_query("What is revenue?")
        assert result.shape == (1, 384)

    def test_empty_query_raises_invalid_input_error(self) -> None:
        service = _service_with_fake_model()
        with pytest.raises(InvalidInputError):
            service.embed_query("")

    def test_whitespace_only_query_raises_invalid_input_error(self) -> None:
        service = _service_with_fake_model()
        with pytest.raises(InvalidInputError):
            service.embed_query("   ")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class TestVectorNormalization:
    def test_l2_norm_close_to_1(self) -> None:
        """When normalize_embeddings=True the norms should be ~1.0."""
        vecs = np.random.rand(4, 384).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = (vecs / norms).astype(np.float32)

        service = EmbeddingService()
        fake_model = MagicMock()
        fake_model.encode.return_value = normalized
        service._model = fake_model

        result = service.embed_texts(["a", "b", "c", "d"])
        computed_norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(computed_norms, np.ones(4), atol=1e-5)


# ---------------------------------------------------------------------------
# Dimension / configuration
# ---------------------------------------------------------------------------


class TestDimension:
    def test_dimension_before_load_returns_default_constant(self) -> None:
        service = EmbeddingService()
        assert not service.is_loaded
        assert service.dimension == EmbeddingService.DIMENSION == 384

    def test_dimension_after_load_reflects_actual_model(self) -> None:
        service = _service_with_fake_model(dimension=768)
        assert service.dimension == 768

    def test_dimension_property_does_not_force_a_load(self) -> None:
        service = EmbeddingService()
        with patch("builtins.__import__", side_effect=AssertionError("must not import")):
            # Accessing .dimension before any embed call must never attempt
            # to import sentence_transformers.
            assert service.dimension == 384
        assert not service.is_loaded


class TestConfiguration:
    def test_defaults_come_from_settings(self) -> None:
        service = EmbeddingService()
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_explicit_overrides_take_precedence(self) -> None:
        service = EmbeddingService(model_name="custom/model", device="cuda", batch_size=4)
        assert service.model_name == "custom/model"
        assert service._device == "cuda"
        assert service._batch_size == 4


# ---------------------------------------------------------------------------
# Optional / manual integration test with the real model
# ---------------------------------------------------------------------------
#
# This test downloads and loads the actual sentence-transformers model and
# performs real CPU inference. It is skipped by default so the standard test
# suite (`pytest`) stays fast, deterministic, and network-free.
#
# To run it manually:
#   OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 pytest tests/test_embeddings.py -k RealModel -v
#
_RUN_INTEGRATION = os.environ.get("OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS") == "1"


@pytest.mark.skipif(
    not _RUN_INTEGRATION,
    reason=(
        "Real-model integration test skipped by default (requires downloading "
        "sentence-transformers weights). Set OMNIFLOW_RUN_MODEL_INTEGRATION_TESTS=1 to run."
    ),
)
class TestRealModelIntegration:
    def test_real_model_produces_normalized_embeddings(self) -> None:
        service = EmbeddingService()
        vectors = service.embed_texts(["OmniFlow is a multimodal assistant.", "Hello world."])

        assert vectors.shape == (2, service.dimension)
        assert vectors.dtype == np.float32

        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-3)
        assert service.is_loaded
