"""Abstract base interface for local vector store and retrieval."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseVectorStore(ABC):
    """Minimal interface for local vector storage and similarity search."""

    @abstractmethod
    def add_texts(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
        embeddings: np.ndarray | None = None,
    ) -> list[str]:
        """Indexes text chunks with optional metadata and pre-computed embeddings."""
        pass

    @abstractmethod
    def search(
        self, query: np.ndarray, top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Performs similarity search against a pre-embedded query vector, returning top-k matching chunks."""
        pass
