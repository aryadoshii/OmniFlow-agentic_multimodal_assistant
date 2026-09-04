"""Audio processor for WAV, MP3, and M4A files using lazy-loaded faster-whisper."""

import logging
from pathlib import Path
from omniflow.exceptions import ProcessingFailureError, UnsupportedFileError
from omniflow.models.document import (
    ExtractionMethod,
    NormalizedDocument,
    SourceType,
)
from omniflow.processors.base import BaseProcessor
from omniflow.services.temp_manager import temporary_upload_file
from omniflow.services.whisper_service import WhisperService

logger = logging.getLogger(__name__)

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".m4a"}
SUPPORTED_AUDIO_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
}


class AudioProcessor(BaseProcessor):
    """Processes audio recordings into transcribed text using local Whisper STT."""

    def __init__(self, whisper_service: WhisperService | None = None) -> None:
        self.whisper_service = whisper_service or WhisperService()

    def can_process(self, mime_type: str, filename: str | None = None) -> bool:
        """Determines if the file format is supported audio."""
        if mime_type in SUPPORTED_AUDIO_MIMES:
            return True
        if filename:
            suffix = Path(filename).suffix.lower()
            return suffix in SUPPORTED_AUDIO_EXTS
        return False

    def process(
        self, file_bytes: bytes, filename: str, mime_type: str
    ) -> NormalizedDocument:
        """Transcribes audio content safely via temporary file handling and Whisper STT."""
        if not file_bytes:
            raise ProcessingFailureError(
                f"Audio file '{filename}' is empty.",
                details={"filename": filename},
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTS and mime_type not in SUPPORTED_AUDIO_MIMES:
            raise UnsupportedFileError(
                f"Unsupported audio format for '{filename}'. Supported extensions: {sorted(SUPPORTED_AUDIO_EXTS)}",
                details={"filename": filename},
            )

        # Write to temporary file for Whisper ingestion and ensure cleanup
        with temporary_upload_file(file_bytes, filename) as temp_audio_path:
            transcript, confidence, stt_metadata = self.whisper_service.transcribe(
                temp_audio_path
            )

        warnings: list[str] = []
        if not transcript:
            warnings.append(f"No audible speech recognized in audio file '{filename}'.")

        logger.info(
            "Transcribed audio '%s' (duration: %ss, words: %d)",
            filename,
            stt_metadata.get("duration_seconds", "unknown"),
            len(transcript.split()) if transcript else 0,
        )

        canonical_mime = (
            "audio/wav" if suffix == ".wav"
            else "audio/mpeg" if suffix == ".mp3"
            else "audio/mp4"
        )

        return NormalizedDocument(
            filename=filename,
            source_type=SourceType.AUDIO,
            mime_type=canonical_mime,
            content=transcript,
            extraction_method=ExtractionMethod.SPEECH_TO_TEXT,
            confidence=confidence,
            metadata=stt_metadata,
            warnings=warnings,
        )
