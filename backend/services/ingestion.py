"""Ingestion service coordinating file validation and deterministic processor selection."""

import logging
from typing import Sequence
from backend.config import get_settings
from backend.exceptions import UnsupportedFileError
from backend.models.document import NormalizedDocument
from backend.processors.audio_processor import AudioProcessor
from backend.processors.base import BaseProcessor
from backend.processors.image_processor import ImageProcessor
from backend.processors.pdf_processor import PDFProcessor
from backend.processors.text_processor import TextProcessor
from backend.services.file_validator import validate_upload

logger = logging.getLogger(__name__)


class IngestionService:
    """Orchestrates input validation, deterministic processor dispatch, and document normalization."""

    def __init__(
        self,
        processors: Sequence[BaseProcessor] | None = None,
    ) -> None:
        self.settings = get_settings()
        self.processors: list[BaseProcessor] = list(
            processors
            if processors is not None
            else [
                TextProcessor(),
                PDFProcessor(),
                ImageProcessor(),
                AudioProcessor(),
            ]
        )

    def process_text(self, text: str) -> NormalizedDocument:
        """Normalizes direct text input."""
        text_proc = next(
            (p for p in self.processors if isinstance(p, TextProcessor)),
            TextProcessor(),
        )
        return text_proc.process(text.encode("utf-8"), filename="user_input.txt")

    def process_file(
        self,
        content: bytes,
        filename: str,
        declared_mime_type: str | None = None,
    ) -> NormalizedDocument:
        """Validates upload and executes matching processor."""
        canonical_mime = validate_upload(
            filename=filename,
            content=content,
            declared_mime_type=declared_mime_type,
            max_size_mb=self.settings.max_upload_size_mb,
        )

        for processor in self.processors:
            if processor.can_process(canonical_mime, filename=filename):
                logger.debug(
                    "Selected processor %s for file '%s' (%s)",
                    processor.__class__.__name__,
                    filename,
                    canonical_mime,
                )
                return processor.process(content, filename=filename, mime_type=canonical_mime)

        raise UnsupportedFileError(
            f"No suitable processor found for file '{filename}' with MIME type '{canonical_mime}'.",
            details={"filename": filename, "mime_type": canonical_mime},
        )

    def ingest_inputs(
        self,
        text: str | None = None,
        files: Sequence[tuple[bytes, str, str | None]] | None = None,
    ) -> list[NormalizedDocument]:
        """Processes optional text and multiple uploaded files in sequential order."""
        documents: list[NormalizedDocument] = []

        # 1. Process text input if provided
        if text and text.strip():
            documents.append(self.process_text(text))

        # 2. Process file uploads preserving ordering
        if files:
            for content, filename, declared_mime in files:
                doc = self.process_file(content, filename, declared_mime)
                documents.append(doc)

        return documents
