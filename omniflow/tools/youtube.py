"""YouTube transcript retrieval tool for OmniFlow.

Deterministically extracts a canonical video ID from a YouTube URL (no
network access during validation), retrieves publicly available captions
via the free, keyless ``youtube-transcript-api`` library, and normalizes
the result into structured Pydantic output conforming to the Phase 3.1
BaseTool contract (structured input in, structured output out).

This module has no knowledge of LLMs, prompts, agents, planning, or
orchestration frameworks.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from pydantic import BaseModel, Field, field_validator

from omniflow.config import get_settings
from omniflow.exceptions import (
    ExternalProviderError,
    TranscriptUnavailableError,
)
from omniflow.tools.base import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level optional import so the app still starts without the package
# installed, and so tests can patch this module's own attributes to simulate
# provider behavior without any network access.
# ---------------------------------------------------------------------------
try:
    from youtube_transcript_api import (
        AgeRestricted,
        InvalidVideoId,
        NoTranscriptFound,
        NotTranslatable,
        TranscriptsDisabled,
        TranslationLanguageNotAvailable,
        VideoUnavailable,
        VideoUnplayable,
        YouTubeTranscriptApi,
        YouTubeTranscriptApiException,
    )
except ImportError:  # pragma: no cover - exercised via a dedicated test
    YouTubeTranscriptApi = None  # type: ignore[assignment,misc]
    YouTubeTranscriptApiException = Exception  # type: ignore[assignment,misc]
    AgeRestricted = None  # type: ignore[assignment,misc]
    InvalidVideoId = None  # type: ignore[assignment,misc]
    NoTranscriptFound = None  # type: ignore[assignment,misc]
    NotTranslatable = None  # type: ignore[assignment,misc]
    TranscriptsDisabled = None  # type: ignore[assignment,misc]
    TranslationLanguageNotAvailable = None  # type: ignore[assignment,misc]
    VideoUnavailable = None  # type: ignore[assignment,misc]
    VideoUnplayable = None  # type: ignore[assignment,misc]

# Exceptions meaning "the video/request was valid, but no transcript exists
# for it" -- distinct from a network/provider-side failure.
_UNAVAILABLE_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
    exc
    for exc in (
        TranscriptsDisabled,
        NoTranscriptFound,
        VideoUnavailable,
        VideoUnplayable,
        AgeRestricted,
        NotTranslatable,
        TranslationLanguageNotAvailable,
        InvalidVideoId,
    )
    if exc is not None
)

_VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")
_WATCH_HOSTNAMES = frozenset(
    {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
)
_SHORT_HOSTNAMES = frozenset({"youtu.be", "www.youtu.be"})


def extract_video_id(url: str) -> str:
    """Parses a URL and deterministically extracts a canonical YouTube video ID.

    Performs no network I/O -- purely structural validation of the URL
    (scheme, hostname, path/query) against known YouTube URL shapes.

    Args:
        url: A candidate YouTube URL.

    Returns:
        The 11-character canonical video ID.

    Raises:
        ValueError: If the URL is empty, malformed, not a recognized YouTube
                    hostname, or does not resolve to a well-formed video ID.
    """
    if not url or not url.strip():
        raise ValueError("URL must not be empty.")

    candidate = url.strip()

    try:
        parsed = urlparse(candidate)
    except ValueError as exc:
        raise ValueError(f"URL could not be parsed: {exc}") from exc

    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use the http or https scheme.")

    hostname = (parsed.hostname or "").lower()
    video_id: str | None = None

    if hostname in _SHORT_HOSTNAMES:
        video_id = parsed.path.lstrip("/").split("/")[0] or None
    elif hostname in _WATCH_HOSTNAMES:
        if parsed.path == "/watch":
            values = parse_qs(parsed.query).get("v")
            video_id = values[0] if values else None
        elif parsed.path.startswith("/shorts/"):
            video_id = parsed.path[len("/shorts/") :].split("/")[0] or None
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path[len("/embed/") :].split("/")[0] or None

    if not video_id or not _VIDEO_ID_PATTERN.match(video_id):
        raise ValueError(
            f"'{candidate}' is not a recognized YouTube video URL. "
            "Supported formats: watch, youtu.be, shorts, and embed URLs."
        )

    return video_id


class YouTubeTranscriptInput(BaseModel):
    """Structured input for YouTubeTranscriptTool.

    URL validation is purely structural (no network access): it rejects
    empty, malformed, and non-YouTube URLs before any transcript retrieval
    is attempted.
    """

    url: str = Field(
        ...,
        min_length=1,
        description="A YouTube video URL (watch, youtu.be, shorts, or embed format).",
    )
    languages: list[str] = Field(
        default_factory=lambda: ["en"],
        description="Preferred transcript language codes, in priority order.",
    )

    @field_validator("url")
    @classmethod
    def _validate_youtube_url(cls, value: str) -> str:
        extract_video_id(value)  # raises ValueError -> wrapped as a pydantic ValidationError
        return value.strip()


class TranscriptSegment(BaseModel):
    """A single timed transcript snippet, preserved for downstream chunking."""

    text: str = Field(..., description="Snippet text as supplied by the provider.")
    start_seconds: float = Field(
        ..., ge=0.0, description="Snippet start offset in seconds."
    )
    duration_seconds: float = Field(
        ..., ge=0.0, description="Snippet on-screen duration in seconds."
    )


class YouTubeTranscriptOutput(BaseModel):
    """Structured, normalized transcript result for a single YouTube video."""

    video_id: str = Field(..., description="Canonical 11-character YouTube video ID.")
    source_url: str = Field(..., description="Original input URL, echoed back for traceability.")
    transcript: str = Field(
        default="", description="Normalized, combined transcript text."
    )
    language: str | None = Field(
        default=None, description="Human-readable transcript language, if supplied by the provider."
    )
    language_code: str | None = Field(
        default=None, description="Transcript language code (e.g. 'en'), if supplied by the provider."
    )
    is_generated: bool | None = Field(
        default=None,
        description="True if the transcript is auto-generated captions rather than manually created, if known.",
    )
    segment_count: int = Field(
        default=0, description="Number of transcript segments retained after normalization."
    )
    total_duration_seconds: float | None = Field(
        default=None,
        description="Duration spanned by retained segments in seconds, derived from segment timing.",
    )
    segments: list[TranscriptSegment] = Field(
        default_factory=list,
        description="Timed transcript segments, preserved for future chunking/citation use.",
    )
    provider: str = Field(
        default="youtube_transcript_api",
        description="Identifier of the extraction mechanism used.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Safe, non-sensitive metadata bundle for downstream ingestion/RAG use.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings encountered during retrieval or normalization.",
    )


def _normalize_segments(
    raw_segments: list[Any],
) -> tuple[str, list[TranscriptSegment], list[str]]:
    """Deterministically cleans and merges raw provider snippets.

    Strips whitespace, drops empty snippets, collapses immediately-repeated
    duplicate snippets, and joins the remaining text with single spaces.
    Does not rewrite, summarize, translate, or otherwise alter the
    semantic content of the transcript.

    Returns:
        A tuple of (combined_transcript, cleaned_segments, warnings).
    """
    warnings: list[str] = []
    cleaned: list[TranscriptSegment] = []
    previous_text: str | None = None
    duplicate_count = 0
    empty_count = 0

    for snippet in raw_segments:
        text = getattr(snippet, "text", "") or ""
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            empty_count += 1
            continue

        if previous_text is not None and text == previous_text:
            duplicate_count += 1
            continue

        cleaned.append(
            TranscriptSegment(
                text=text,
                start_seconds=float(getattr(snippet, "start", 0.0)),
                duration_seconds=float(getattr(snippet, "duration", 0.0)),
            )
        )
        previous_text = text

    if empty_count:
        warnings.append(f"Skipped {empty_count} empty transcript segment(s).")
    if duplicate_count:
        warnings.append(f"Skipped {duplicate_count} consecutive duplicate segment(s).")

    transcript = re.sub(r"\s+", " ", " ".join(seg.text for seg in cleaned)).strip()

    if not transcript:
        warnings.append("No transcript text was available for this video.")

    return transcript, cleaned, warnings


def _apply_length_limit(
    transcript: str,
    segments: list[TranscriptSegment],
    max_chars: int,
) -> tuple[str, list[TranscriptSegment], list[str]]:
    """Truncates transcript/segments to a configured maximum character budget.

    Bounds memory usage for very long transcripts (e.g. multi-hour videos)
    while keeping the returned segments consistent with the returned text.
    """
    if len(transcript) <= max_chars:
        return transcript, segments, []

    truncated_transcript = transcript[:max_chars].rstrip()

    kept_segments: list[TranscriptSegment] = []
    running_len = 0
    for seg in segments:
        added_len = len(seg.text) + 1
        if running_len + added_len > max_chars:
            break
        kept_segments.append(seg)
        running_len += added_len

    warning = (
        f"Transcript truncated to {max_chars} characters "
        f"(original length {len(transcript)})."
    )
    return truncated_transcript, kept_segments, [warning]


class YouTubeTranscriptTool(BaseTool):
    """Retrieves and normalizes the transcript for a public YouTube video.

    Uses ``youtube-transcript-api`` (no API key, no paid access required) to
    fetch publicly available captions. Performs deterministic URL validation,
    transcript normalization, and defensive length limiting. No LLM calls
    are made anywhere in this tool.
    """

    @property
    def name(self) -> str:
        return "youtube_transcript"

    @property
    def description(self) -> str:
        return (
            "Retrieves and normalizes the transcript/captions for a public YouTube "
            "video given its URL. Accepts watch, youtu.be, shorts, and embed URL "
            "formats. Does not require a paid API key."
        )

    @property
    def input_model(self) -> type[BaseModel]:
        return YouTubeTranscriptInput

    def run(self, tool_input: YouTubeTranscriptInput) -> YouTubeTranscriptOutput:
        """Fetches and normalizes a transcript.

        Raises:
            ExternalProviderError: If the provider library is unavailable, or
                                    the request fails for provider/network reasons.
            TranscriptUnavailableError: If the video is valid but no transcript
                                         exists for it (disabled, missing, etc.).
        """
        settings = get_settings()
        video_id = extract_video_id(tool_input.url)

        if YouTubeTranscriptApi is None:
            raise ExternalProviderError(
                "youtube-transcript-api is not installed; cannot retrieve transcripts.",
                details={"tool": self.name, "video_id": video_id},
            )

        logger.debug("Fetching transcript for video_id='%s'.", video_id)

        try:
            fetched = YouTubeTranscriptApi().fetch(
                video_id, languages=tool_input.languages
            )
        except _UNAVAILABLE_EXCEPTIONS as exc:
            raise TranscriptUnavailableError(
                f"No transcript is available for video '{video_id}': {type(exc).__name__}.",
                details={"video_id": video_id, "reason": type(exc).__name__},
            ) from exc
        except YouTubeTranscriptApiException as exc:
            raise ExternalProviderError(
                f"Failed to retrieve transcript for video '{video_id}': {type(exc).__name__}.",
                details={"video_id": video_id, "error_type": type(exc).__name__},
            ) from exc

        transcript, segments, normalize_warnings = _normalize_segments(list(fetched))
        transcript, segments, limit_warnings = _apply_length_limit(
            transcript, segments, settings.youtube_transcript_max_chars
        )

        total_duration = (
            max(seg.start_seconds + seg.duration_seconds for seg in segments)
            if segments
            else None
        )

        language_code = getattr(fetched, "language_code", None)
        is_generated = getattr(fetched, "is_generated", None)

        output = YouTubeTranscriptOutput(
            video_id=video_id,
            source_url=tool_input.url,
            transcript=transcript,
            language=getattr(fetched, "language", None),
            language_code=language_code,
            is_generated=is_generated,
            segment_count=len(segments),
            total_duration_seconds=total_duration,
            segments=segments,
            provider="youtube_transcript_api",
            metadata={
                "video_id": video_id,
                "language_code": language_code,
                "is_generated": is_generated,
                "segment_count": len(segments),
                "provider": "youtube_transcript_api",
            },
            warnings=normalize_warnings + limit_warnings,
        )

        logger.info(
            "Retrieved transcript for video '%s': %d segments, %d chars.",
            video_id,
            len(segments),
            len(transcript),
        )
        return output
