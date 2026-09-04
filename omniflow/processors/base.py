"""Abstract base interface for multimodal input processors."""

from abc import ABC, abstractmethod
from omniflow.models.document import NormalizedDocument


class BaseProcessor(ABC):
    """Minimal interface for modality-specific ingestion processors."""

    @abstractmethod
    def can_process(self, mime_type: str) -> bool:
        """Determines whether this processor supports the provided MIME type."""
        pass

    @abstractmethod
    def process(
        self, file_bytes: bytes, filename: str, mime_type: str
    ) -> NormalizedDocument:
        """Extracts content from raw bytes and returns a NormalizedDocument."""
        pass
