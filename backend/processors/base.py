"""Abstract base interface for multimodal input processors."""

from abc import ABC, abstractmethod
from backend.models.document import NormalizedDocument


class BaseProcessor(ABC):
    """Minimal interface for modality-specific ingestion processors."""

    @abstractmethod
    def can_process(self, mime_type: str, filename: str | None = None) -> bool:
        """Determines whether this processor supports the provided MIME type or filename."""
        pass

    @abstractmethod
    def process(
        self, file_bytes: bytes, filename: str, mime_type: str
    ) -> NormalizedDocument:
        """Extracts content from raw bytes and returns a NormalizedDocument."""
        pass
