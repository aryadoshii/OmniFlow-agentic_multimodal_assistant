"""Lazy-loaded local CPU embedding service using sentence-transformers.

The SentenceTransformer model is NOT imported or loaded at module import
time, nor at EmbeddingService construction time. Weights are loaded only on
the first call to embed_texts()/embed_query(), then cached and reused for
the lifetime of the service instance. This keeps application startup, and
any import of this module, free from heavy ML library initialization.

This is the only embedding implementation in OmniFlow -- do not create a
second one elsewhere; extend this service if new embedding needs arise.
"""

from __future__ import annotations

import logging

import numpy as np

from omniflow.config import get_settings
from omniflow.exceptions import EmbeddingGenerationError, InvalidInputError

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Wraps a local sentence-transformers model for L2-normalized dense embeddings.

    Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions).
    Chosen for CPU-friendly inference speed and a small footprint (~80MB),
    appropriate for a free-tier deployment target, at an accepted quality
    tradeoff versus larger models (e.g. all-mpnet-base-v2, ~420MB/768-dim).

    Output vectors are L2-normalized so cosine similarity reduces to a plain
    inner product -- compatible with FAISS's IndexFlatIP used elsewhere in
    the RAG layer.

    Usage::

        service = EmbeddingService()
        vectors = service.embed_texts(["hello world", "another sentence"])
        query_vector = service.embed_query("hello")
    """

    DIMENSION = 384
    """Known embedding dimension for the default model, valid before any load."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self._model_name = model_name or settings.embedding_model_name
        self._device = device or settings.embedding_device
        self._batch_size = batch_size or settings.embedding_batch_size
        self._model = None  # Loaded lazily on first use.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Encodes a batch of strings into L2-normalized float32 embeddings.

        Args:
            texts: Non-empty list of strings to embed. Must contain at least
                   one non-blank string.

        Returns:
            A (len(texts), dimension) float32 numpy array of L2-normalized vectors.

        Raises:
            InvalidInputError: If texts is empty or contains only blank strings.
            EmbeddingGenerationError: If the embedding backend is unavailable,
                                      fails to load, or encoding fails.
        """
        if not texts:
            raise InvalidInputError("texts must be a non-empty list of strings.")
        if all(not text or not text.strip() for text in texts):
            raise InvalidInputError(
                "texts must contain at least one non-empty string."
            )

        model = self._get_model()

        try:
            embeddings = model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as exc:
            logger.error("Embedding generation failed for %d text(s): %s", len(texts), exc)
            raise EmbeddingGenerationError(
                f"Failed to generate embeddings: {type(exc).__name__}.",
                details={"text_count": str(len(texts)), "error_type": type(exc).__name__},
            ) from exc

        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Encodes a single query string into an L2-normalized float32 embedding.

        Args:
            query: The search query string. Must be non-empty.

        Returns:
            A (1, dimension) float32 numpy array.

        Raises:
            InvalidInputError: If query is empty or blank.
            EmbeddingGenerationError: If the embedding backend is unavailable,
                                      fails to load, or encoding fails.
        """
        if not query or not query.strip():
            raise InvalidInputError("query must be a non-empty string.")
        return self.embed_texts([query])

    @property
    def dimension(self) -> int:
        """Returns the embedding dimension.

        Returns the actual dimension of the loaded model if it has already
        been loaded, otherwise the known dimension for the configured
        default model -- without forcing a model load just to answer this.
        """
        if self._model is not None:
            # Prefer the newer method name (get_sentence_embedding_dimension is
            # deprecated as of sentence-transformers >= 5.x) while remaining
            # compatible with older installed versions that lack it.
            if hasattr(self._model, "get_embedding_dimension"):
                return self._model.get_embedding_dimension()
            return self._model.get_sentence_embedding_dimension()
        return self.DIMENSION

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model has been loaded into memory."""
        return self._model is not None

    @property
    def model_name(self) -> str:
        """Returns the configured model name."""
        return self._model_name

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazily loads and caches the SentenceTransformer model.

        Raises:
            EmbeddingGenerationError: If sentence-transformers is not
                                      installed or the model fails to load.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise EmbeddingGenerationError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install sentence-transformers",
                    details={"model_name": self._model_name},
                ) from exc

            logger.info(
                "Loading embedding model '%s' on device '%s'.",
                self._model_name,
                self._device,
            )
            try:
                self._model = SentenceTransformer(
                    self._model_name, device=self._device
                )
            except Exception as exc:
                logger.error(
                    "Failed to load embedding model '%s': %s", self._model_name, exc
                )
                raise EmbeddingGenerationError(
                    f"Failed to load embedding model '{self._model_name}': {type(exc).__name__}.",
                    details={
                        "model_name": self._model_name,
                        "device": self._device,
                        "error_type": type(exc).__name__,
                    },
                ) from exc
            logger.info("Embedding model loaded successfully.")
        return self._model
