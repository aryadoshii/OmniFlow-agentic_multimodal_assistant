"""Abstract base interface for local vector store and retrieval."""

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Minimal interface for local vector storage and similarity search."""

    @abstractmethod
    def add_texts(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> list[str]:
        """Indexes text chunks with optional metadata."""
        pass

    @abstractmethod
    def search(
        self, query: str, top_k: int = 4
    ) -> list[dict[str, Any]]:
        """Performs similarity search returning top-k matching chunks."""
        pass
