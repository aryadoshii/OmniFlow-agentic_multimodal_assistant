"""Plain text processor for direct text inputs and text files."""

import logging
from pathlib import Path
from omniflow.exceptions import InvalidInputError
from omniflow.models.document import (
    ExtractionMethod,
    NormalizedDocument,
    SourceType,
)
from omniflow.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


class TextProcessor(BaseProcessor):
    """Processes plain text inputs into normalized documents."""

    def can_process(self, mime_type: str, filename: str | None = None) -> bool:
        """Checks if input is text/plain or has a .txt extension."""
        if mime_type == "text/plain":
            return True
        if filename:
            suffix = Path(filename).suffix.lower()
            return suffix == ".txt"
        return False

    def process(
        self, file_bytes: bytes, filename: str = "input.txt", mime_type: str = "text/plain"
    ) -> NormalizedDocument:
        """Validates, decodes, and normalizes plain text content."""
        if not file_bytes:
            raise InvalidInputError(
                f"Text input '{filename}' is empty.",
                details={"filename": filename},
            )

        # Attempt decoding with UTF-8 first, fallback to latin-1
        try:
            raw_text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = file_bytes.decode("latin-1")
            except Exception as exc:
                raise InvalidInputError(
                    f"Unable to decode text file '{filename}': {exc}",
                    details={"filename": filename},
                ) from exc

        # Normalize obvious repeated empty lines or trailing spaces while preserving structure
        lines = [line.rstrip() for line in raw_text.splitlines()]
        cleaned_text = "\n".join(lines).strip()

        if not cleaned_text:
            raise InvalidInputError(
                f"Text input '{filename}' contains only whitespace.",
                details={"filename": filename},
            )

        logger.info(
            "Processed text document '%s' (%d characters)", filename, len(cleaned_text)
        )

        return NormalizedDocument(
            filename=filename,
            source_type=SourceType.TEXT,
            mime_type="text/plain",
            content=cleaned_text,
            extraction_method=ExtractionMethod.DIRECT_INPUT,
            confidence=1.0,
            metadata={
                "character_count": len(cleaned_text),
                "line_count": len(lines),
            },
        )
