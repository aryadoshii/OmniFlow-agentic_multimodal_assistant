"""Lazy-loaded speech-to-text service powered by faster-whisper."""

import logging
from pathlib import Path
from typing import Any
from omniflow.config import get_settings
from omniflow.exceptions import TranscriptionError

logger = logging.getLogger(__name__)


class WhisperService:
    """Service providing lazy-loaded speech-to-text transcription.

    Ensures faster-whisper is not loaded or downloaded on import or startup,
    and reuses a single model instance when audio processing is actually invoked.
    """

    _cached_model: Any = None

    def __init__(self) -> None:
        settings = get_settings()
        self.model_size = settings.whisper_model_size
        self.device = settings.whisper_device
        self.compute_type = settings.whisper_compute_type

    def _get_model(self) -> Any:
        """Lazily imports and instantiates the WhisperModel on first audio request."""
        if WhisperService._cached_model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                logger.error("faster-whisper package is not available: %s", exc)
                raise TranscriptionError(
                    "Speech-to-text backend is not available (faster-whisper not installed).",
                    details={"backend": "faster-whisper"},
                ) from exc

            logger.info(
                "Initializing faster-whisper model (size=%s, device=%s, compute_type=%s)",
                self.model_size,
                self.device,
                self.compute_type,
            )
            try:
                WhisperService._cached_model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                logger.error("Failed to initialize faster-whisper model: %s", exc)
                raise TranscriptionError(
                    f"Failed to initialize speech-to-text model '{self.model_size}': {exc}",
                    details={"model_size": self.model_size, "error": str(exc)},
                ) from exc

        return WhisperService._cached_model

    def transcribe(
        self, audio_path: Path
    ) -> tuple[str, float | None, dict[str, Any]]:
        """Transcribes an audio file into text, returning (transcript, confidence, metadata)."""
        model = self._get_model()

        try:
            segments, info = model.transcribe(
                str(audio_path),
                beam_size=5,
            )

            segment_texts = [segment.text.strip() for segment in segments if segment.text]
            transcript = " ".join(segment_texts).strip()

            metadata: dict[str, Any] = {
                "stt_engine": "faster-whisper",
                "model_size": self.model_size,
                "duration_seconds": round(getattr(info, "duration", 0.0), 2),
                "language": getattr(info, "language", "unknown"),
                "language_probability": round(
                    getattr(info, "language_probability", 0.0), 4
                ),
            }

            return transcript, None, metadata

        except Exception as exc:
            logger.error("Transcription failed for audio %s: %s", audio_path, exc)
            raise TranscriptionError(
                f"Audio transcription failed: {exc}",
                details={"audio_file": audio_path.name, "error": str(exc)},
            ) from exc
