"""Ingestion service coordinating file validation and deterministic processor selection."""

import logging
import re
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
from backend.tools.youtube import extract_video_id

logger = logging.getLogger(__name__)

# Generic (not YouTube-specific) URL-token finder: matches any http(s) URL
# shape in free text, stopping at whitespace or a common enclosing/trailing
# punctuation character. This only identifies "URL-shaped text" -- it makes
# no judgment about whether a candidate is actually a YouTube URL.
_URL_TOKEN_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")

# Trailing punctuation commonly following a URL in prose (e.g. "...watch?v=abc123."
# or "(see youtu.be/abc123)") that is not part of the URL itself.
_TRAILING_PUNCTUATION = ").,;:!?]'\""


def _extract_youtube_urls(content: str) -> list[str]:
    """Deterministically finds YouTube URLs embedded in document text.

    A backstop for understand_intent's LLM-based URL spotting (see
    backend/agents/intent.py) -- guarantees a YouTube link embedded in a
    document's content is never missed due to an LLM error, retry
    exhaustion, or the URL simply falling past what the LLM's attention
    picks up in a long document.

    Deliberately reuses backend.tools.youtube.extract_video_id as the sole
    authority on what counts as a valid YouTube URL shape (watch, youtu.be,
    shorts, embed) rather than reimplementing that hostname/path logic here
    -- the two can never drift out of sync. This function only supplies the
    generic "find URL-shaped tokens in text" half of the job; every
    candidate is still validated by extract_video_id before being kept.

    Returns the original matched URL strings (not video ids), deduplicated,
    in the order first encountered.
    """
    seen: set[str] = set()
    detected: list[str] = []
    for match in _URL_TOKEN_PATTERN.finditer(content):
        candidate = match.group(0).rstrip(_TRAILING_PUNCTUATION)
        if not candidate or candidate in seen:
            continue
        try:
            extract_video_id(candidate)
        except ValueError:
            continue
        seen.add(candidate)
        detected.append(candidate)
    return detected


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
        doc = text_proc.process(text.encode("utf-8"), filename="user_input.txt")
        doc.detected_urls = _extract_youtube_urls(doc.content)
        return doc

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
                doc = processor.process(content, filename=filename, mime_type=canonical_mime)
                doc.detected_urls = _extract_youtube_urls(doc.content)
                return doc

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
