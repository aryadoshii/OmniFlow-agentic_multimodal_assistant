"""In-memory FAISS vector store implementation for OmniFlow RAG.

Uses IndexFlatIP (inner-product) over L2-normalized vectors, which is
equivalent to cosine similarity while keeping index construction and
querying simple, exact (no approximation), and dependency-light --
appropriate for the corpus sizes this project targets. See the Phase 3.4
report for the full index-choice rationale.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from omniflow.config import get_settings
from omniflow.exceptions import InvalidInputError, RAGRetrievalError
from omniflow.rag.base import BaseVectorStore

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """Purely in-memory FAISS index for semantic similarity search.

    Stores chunk text and metadata in a parallel Python dict so that
    FAISS integer IDs can be resolved to the original content at search time.
    This dict *is* the metadata persistence layer for this store: everything
    lives in-process, in memory, for the lifetime of the instance. No disk
    or database persistence is implemented (see the Phase 3.4 report).

    Usage::

        store = FAISSVectorStore(dimension=384)
        ids = store.add_texts(["hello world"], [{"source": "doc.txt"}], embeddings=vecs)
        results = store.search(query_vector, top_k=4)
    """

    def __init__(self, dimension: int = 384) -> None:
        try:
            import faiss  # noqa: PLC0415
        except ImportError as exc:
            raise RAGRetrievalError(
                "faiss-cpu is not installed. Install it with: pip install faiss-cpu",
                details={"dependency": "faiss-cpu"},
            ) from exc
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)
        # Maps FAISS integer IDs (0-based sequential) → chunk payload
        self._store: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # BaseVectorStore interface
    # ------------------------------------------------------------------

    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> list[str]:
        """Indexes text strings with optional metadata and pre-computed embeddings.

        Args:
            texts: Plain-text strings to index.
            metadatas: Optional per-text metadata dicts. Defaults to empty dicts.
                       If provided, must have exactly one entry per text.
            embeddings: Optional pre-computed (N, dimension) float32 array. If
                        None, texts are stored but NOT embedded -- callers are
                        expected to always supply vectors in normal use.

        Returns:
            List of string IDs (matching FAISS 0-based integer IDs cast to str).

        Raises:
            InvalidInputError: If metadatas is provided with a length that does
                                not match texts, if a metadata entry is not a
                                dict, or if the embeddings row count does not
                                match the number of texts.
            RAGRetrievalError: If the embeddings' dimensionality does not match
                                this store's configured dimension, or if FAISS
                                indexing otherwise fails.
        """
        if not texts:
            return []

        n = len(texts)

        if metadatas is not None:
            if len(metadatas) != n:
                raise InvalidInputError(
                    f"metadatas length ({len(metadatas)}) must match texts length ({n}).",
                    details={"texts_count": str(n), "metadatas_count": str(len(metadatas))},
                )
            for meta in metadatas:
                if not isinstance(meta, dict):
                    raise InvalidInputError(
                        f"Each metadata entry must be a dict, got {type(meta).__name__}.",
                        details={"invalid_type": type(meta).__name__},
                    )
        else:
            metadatas = [{} for _ in range(n)]

        if embeddings is not None:
            vecs = np.asarray(embeddings, dtype=np.float32)
            if vecs.ndim == 1:
                vecs = vecs.reshape(1, -1)

            if vecs.shape[0] != n:
                raise InvalidInputError(
                    f"embeddings row count ({vecs.shape[0]}) must match texts length ({n}).",
                    details={"texts_count": str(n), "embeddings_rows": str(vecs.shape[0])},
                )
            if vecs.shape[1] != self._dimension:
                raise RAGRetrievalError(
                    f"Embedding dimension ({vecs.shape[1]}) does not match "
                    f"this store's configured dimension ({self._dimension}).",
                    details={
                        "expected_dimension": str(self._dimension),
                        "actual_dimension": str(vecs.shape[1]),
                    },
                )

            try:
                self._index.add(vecs)
            except Exception as exc:
                raise RAGRetrievalError(
                    f"FAISS indexing failed: {type(exc).__name__}.",
                    details={"error_type": type(exc).__name__},
                ) from exc

        start_id = len(self._store)
        ids: list[str] = []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            faiss_id = start_id + i
            self._store[faiss_id] = {"text": text, "metadata": meta}
            ids.append(str(faiss_id))

        logger.debug("Indexed %d chunks; total index size: %d.", n, len(self._store))
        return ids

    def search(
        self,
        query: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Performs inner-product (cosine) similarity search.

        Args:
            query: A pre-computed (1, dimension) or (dimension,) float32
                   numpy array. Raw strings are rejected -- callers must
                   embed the query first (e.g. via EmbeddingService).
            top_k: Number of results to return. Defaults to
                   ``settings.rag_top_k`` when not provided. Automatically
                   capped at the number of vectors currently indexed.

        Returns:
            Ordered list of dicts with keys ``text``, ``metadata``, and
            ``score``, highest score first. Returns [] if the index is empty.

        Raises:
            InvalidInputError: If query is a string, an empty array, or
                                top_k is not a positive integer.
            RAGRetrievalError: If the query dimension does not match this
                                store's configured dimension, or FAISS search
                                otherwise fails.
        """
        if isinstance(query, str):
            raise InvalidInputError(
                "FAISSVectorStore.search() requires a pre-embedded numpy vector, "
                "not a raw string. Embed the query first (e.g. via EmbeddingService)."
            )

        if top_k is None:
            top_k = get_settings().rag_top_k
        if top_k <= 0:
            raise InvalidInputError(
                f"top_k must be a positive integer, got {top_k}.",
                details={"top_k": str(top_k)},
            )

        if self._index.ntotal == 0:
            return []

        vec = np.asarray(query, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        if vec.size == 0:
            raise InvalidInputError("Query embedding must not be empty.")

        if vec.shape[1] != self._dimension:
            raise RAGRetrievalError(
                f"Query embedding dimension ({vec.shape[1]}) does not match "
                f"this store's configured dimension ({self._dimension}).",
                details={
                    "expected_dimension": str(self._dimension),
                    "actual_dimension": str(vec.shape[1]),
                },
            )

        k = min(top_k, self._index.ntotal)

        try:
            scores, indices = self._index.search(vec, k)
        except Exception as exc:
            raise RAGRetrievalError(
                f"FAISS search failed: {type(exc).__name__}.",
                details={"error_type": type(exc).__name__},
            ) from exc

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            payload = self._store.get(int(idx))
            if payload is None:
                continue
            results.append(
                {
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                    "score": float(score),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Resets the FAISS index and the in-memory store."""
        import faiss  # noqa: PLC0415
        self._index = faiss.IndexFlatIP(self._dimension)
        self._store.clear()
        logger.debug("FAISSVectorStore cleared.")

    def count(self) -> int:
        """Returns the number of vectors currently in the index."""
        return self._index.ntotal
