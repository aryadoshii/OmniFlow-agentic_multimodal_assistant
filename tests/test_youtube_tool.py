"""Tests for the YouTube Transcript Tool (Phase 3.2).

All tests mock only the external provider boundary (YouTubeTranscriptApi's
fetch() call) -- URL validation, normalization, truncation, and exception
mapping are exercised as real code paths. No network access occurs.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from backend.exceptions import (
    ExternalProviderError,
    InvalidInputError,
    TranscriptUnavailableError,
)
from backend.tools.registry import ToolRegistry
from backend.tools.youtube import (
    TranscriptSegment,
    YouTubeTranscriptInput,
    YouTubeTranscriptOutput,
    YouTubeTranscriptTool,
    extract_video_id,
)

VALID_VIDEO_ID = "dQw4w9WgXcQ"


# ---------------------------------------------------------------------------
# Lightweight fakes for the provider boundary (not the library's own classes,
# so tests stay decoupled from its internal module layout).
# ---------------------------------------------------------------------------


@dataclass
class _FakeSnippet:
    text: str
    start: float
    duration: float


class _FakeFetchedTranscript:
    def __init__(
        self,
        snippets: list[_FakeSnippet],
        language: str = "English",
        language_code: str = "en",
        is_generated: bool = False,
    ) -> None:
        self.snippets = snippets
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated

    def __iter__(self):
        return iter(self.snippets)

    def __len__(self) -> int:
        return len(self.snippets)


def _mock_api_returning(fetched_transcript: _FakeFetchedTranscript):
    """Builds a mock replacing YouTubeTranscriptApi() so .fetch() returns a fixed value."""
    mock_instance = MagicMock()
    mock_instance.fetch.return_value = fetched_transcript
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


def _mock_api_raising(exc: BaseException):
    """Builds a mock replacing YouTubeTranscriptApi() so .fetch() raises the given exception."""
    mock_instance = MagicMock()
    mock_instance.fetch.side_effect = exc
    mock_cls = MagicMock(return_value=mock_instance)
    return mock_cls


# ---------------------------------------------------------------------------
# URL validation / video ID extraction
# ---------------------------------------------------------------------------


class TestExtractVideoId:
    def test_standard_watch_url(self) -> None:
        assert (
            extract_video_id(f"https://www.youtube.com/watch?v={VALID_VIDEO_ID}")
            == VALID_VIDEO_ID
        )

    def test_watch_url_with_extra_query_params(self) -> None:
        url = f"https://www.youtube.com/watch?list=PL123&v={VALID_VIDEO_ID}&t=30s"
        assert extract_video_id(url) == VALID_VIDEO_ID

    def test_short_youtu_be_url(self) -> None:
        assert extract_video_id(f"https://youtu.be/{VALID_VIDEO_ID}") == VALID_VIDEO_ID

    def test_shorts_url(self) -> None:
        assert (
            extract_video_id(f"https://www.youtube.com/shorts/{VALID_VIDEO_ID}")
            == VALID_VIDEO_ID
        )

    def test_embed_url(self) -> None:
        assert (
            extract_video_id(f"https://www.youtube.com/embed/{VALID_VIDEO_ID}")
            == VALID_VIDEO_ID
        )

    def test_mobile_hostname(self) -> None:
        assert (
            extract_video_id(f"https://m.youtube.com/watch?v={VALID_VIDEO_ID}")
            == VALID_VIDEO_ID
        )

    def test_empty_url_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            extract_video_id("")

    def test_whitespace_only_url_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            extract_video_id("   ")

    def test_non_youtube_url_raises(self) -> None:
        with pytest.raises(ValueError, match="not a recognized YouTube"):
            extract_video_id("https://example.com/watch?v=dQw4w9WgXcQ")

    def test_lookalike_domain_is_rejected(self) -> None:
        """A hostname merely containing 'youtube.com' must not be accepted."""
        with pytest.raises(ValueError, match="not a recognized YouTube"):
            extract_video_id("https://youtube.com.evil.example/watch?v=dQw4w9WgXcQ")

    def test_malformed_url_raises(self) -> None:
        with pytest.raises(ValueError):
            extract_video_id("not a url at all")

    def test_non_http_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="http or https"):
            extract_video_id(f"javascript:alert(1)//{VALID_VIDEO_ID}")

    def test_watch_url_without_video_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not a recognized YouTube"):
            extract_video_id("https://www.youtube.com/watch?list=PL123")

    def test_wrong_length_id_raises(self) -> None:
        with pytest.raises(ValueError, match="not a recognized YouTube"):
            extract_video_id("https://youtu.be/short")


# ---------------------------------------------------------------------------
# YouTubeTranscriptInput validation
# ---------------------------------------------------------------------------


class TestYouTubeTranscriptInput:
    def test_valid_url_constructs(self) -> None:
        model = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")
        assert model.languages == ["en"]

    def test_invalid_url_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            YouTubeTranscriptInput(url="https://example.com/not-youtube")

    def test_empty_url_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            YouTubeTranscriptInput(url="")

    def test_custom_languages_preserved(self) -> None:
        model = YouTubeTranscriptInput(
            url=f"https://youtu.be/{VALID_VIDEO_ID}", languages=["es", "en"]
        )
        assert model.languages == ["es", "en"]


# ---------------------------------------------------------------------------
# Successful retrieval + normalization
# ---------------------------------------------------------------------------


class TestSuccessfulRetrieval:
    def test_single_segment_transcript(self) -> None:
        fake = _FakeFetchedTranscript([_FakeSnippet("Hello world.", 0.0, 2.0)])
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert isinstance(output, YouTubeTranscriptOutput)
        assert output.video_id == VALID_VIDEO_ID
        assert output.source_url == tool_input.url
        assert output.transcript == "Hello world."
        assert output.segment_count == 1
        assert output.language_code == "en"
        assert output.is_generated is False

    def test_multiple_segments_combined(self) -> None:
        fake = _FakeFetchedTranscript(
            [
                _FakeSnippet("This is the first part.", 0.0, 2.0),
                _FakeSnippet("This is the second part.", 2.0, 2.5),
                _FakeSnippet("And a third.", 4.5, 1.5),
            ]
        )
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert output.segment_count == 3
        assert output.transcript == (
            "This is the first part. This is the second part. And a third."
        )
        assert output.total_duration_seconds == pytest.approx(6.0)

    def test_empty_transcript_produces_warning(self) -> None:
        fake = _FakeFetchedTranscript([_FakeSnippet("", 0.0, 1.0), _FakeSnippet("   ", 1.0, 1.0)])
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert output.transcript == ""
        assert output.segment_count == 0
        assert any("No transcript text" in w for w in output.warnings)

    def test_duplicate_and_empty_segments_normalized(self) -> None:
        fake = _FakeFetchedTranscript(
            [
                _FakeSnippet("Repeated line.", 0.0, 1.0),
                _FakeSnippet("Repeated line.", 1.0, 1.0),  # exact consecutive duplicate
                _FakeSnippet("", 2.0, 1.0),  # empty
                _FakeSnippet("Final line.", 3.0, 1.0),
            ]
        )
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert output.transcript == "Repeated line. Final line."
        assert output.segment_count == 2
        assert any("duplicate segment" in w for w in output.warnings)
        assert any("empty transcript segment" in w for w in output.warnings)

    def test_timestamps_and_metadata_preserved(self) -> None:
        fake = _FakeFetchedTranscript(
            [_FakeSnippet("Timed line.", 12.5, 3.25)],
            language="Spanish",
            language_code="es",
            is_generated=True,
        )
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert len(output.segments) == 1
        segment = output.segments[0]
        assert isinstance(segment, TranscriptSegment)
        assert segment.start_seconds == 12.5
        assert segment.duration_seconds == 3.25
        assert output.language == "Spanish"
        assert output.metadata["language_code"] == "es"
        assert output.metadata["is_generated"] is True
        assert output.metadata["video_id"] == VALID_VIDEO_ID

    def test_transcript_length_limit_truncates(self) -> None:
        long_text = "word " * 100  # 500 chars
        fake = _FakeFetchedTranscript(
            [_FakeSnippet(long_text, float(i), 1.0) for i in range(5)]
        )
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)), \
             patch("backend.tools.youtube.get_settings") as mock_settings:
            mock_settings.return_value.youtube_transcript_max_chars = 100
            output = tool.run(tool_input)

        assert len(output.transcript) <= 100
        assert any("truncated" in w for w in output.warnings)


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


class TestFailureHandling:
    def test_transcripts_disabled_raises_transcript_unavailable(self) -> None:
        class FakeTranscriptsDisabled(Exception):
            pass

        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch(
            "backend.tools.youtube._UNAVAILABLE_EXCEPTIONS", (FakeTranscriptsDisabled,)
        ), patch(
            "backend.tools.youtube.YouTubeTranscriptApi",
            _mock_api_raising(FakeTranscriptsDisabled("disabled")),
        ):
            with pytest.raises(TranscriptUnavailableError) as exc_info:
                tool.run(tool_input)

        assert exc_info.value.error_code == "TRANSCRIPT_UNAVAILABLE"
        assert exc_info.value.details["video_id"] == VALID_VIDEO_ID

    def test_provider_failure_raises_external_provider_error(self) -> None:
        class FakeProviderException(Exception):
            pass

        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch(
            "backend.tools.youtube.YouTubeTranscriptApiException", FakeProviderException
        ), patch(
            "backend.tools.youtube.YouTubeTranscriptApi",
            _mock_api_raising(FakeProviderException("blocked")),
        ):
            with pytest.raises(ExternalProviderError) as exc_info:
                tool.run(tool_input)

        assert exc_info.value.error_code == "EXTERNAL_PROVIDER_ERROR"
        assert exc_info.value.details["video_id"] == VALID_VIDEO_ID

    def test_library_not_installed_raises_external_provider_error(self) -> None:
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", None):
            with pytest.raises(ExternalProviderError, match="not installed"):
                tool.run(tool_input)

    def test_invalid_url_never_reaches_provider(self) -> None:
        """Malformed/non-YouTube input must fail before any provider call is attempted."""
        with pytest.raises(ValidationError):
            YouTubeTranscriptInput(url="https://example.com/nope")
        # No tool.run() call is possible without a valid input model instance,
        # which is exactly the point: invalid input never reaches retrieval.


# ---------------------------------------------------------------------------
# Structured output guarantees
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    def test_output_is_pydantic_basemodel(self) -> None:
        fake = _FakeFetchedTranscript([_FakeSnippet("Hi.", 0.0, 1.0)])
        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output = tool.run(tool_input)

        assert isinstance(output, BaseModel)
        assert isinstance(output, YouTubeTranscriptOutput)

    def test_input_model_property_returns_correct_type(self) -> None:
        tool = YouTubeTranscriptTool()
        assert tool.input_model is YouTubeTranscriptInput

    def test_tool_metadata(self) -> None:
        tool = YouTubeTranscriptTool()
        assert tool.name == "youtube_transcript"
        assert "transcript" in tool.description.lower()


# ---------------------------------------------------------------------------
# No LLM/orchestration coupling
# ---------------------------------------------------------------------------


class TestNoOrchestrationCoupling:
    def test_module_has_no_forbidden_imports(self) -> None:
        import backend.tools.youtube as youtube_module

        with open(youtube_module.__file__, encoding="utf-8") as f:
            content = f.read().lower()
        forbidden = ["langgraph", "openai", "google.generativeai", "langchain"]
        for term in forbidden:
            assert term not in content, f"Forbidden dependency '{term}' referenced in youtube.py."


# ---------------------------------------------------------------------------
# Registry integration (Phase 3.1 contract)
# ---------------------------------------------------------------------------


class TestRegistryIntegration:
    def test_execute_via_registry_success(self) -> None:
        fake = _FakeFetchedTranscript([_FakeSnippet("Registry works.", 0.0, 1.0)])
        registry = ToolRegistry()
        registry.register(YouTubeTranscriptTool())

        with patch("backend.tools.youtube.YouTubeTranscriptApi", _mock_api_returning(fake)):
            output, trace = registry.execute(
                "youtube_transcript", url=f"https://youtu.be/{VALID_VIDEO_ID}"
            )

        assert isinstance(output, YouTubeTranscriptOutput)
        assert output.transcript == "Registry works."
        assert trace.status == "success"
        assert trace.tool_name == "youtube_transcript"

    def test_execute_via_registry_invalid_url_raises_invalid_input_error(self) -> None:
        registry = ToolRegistry()
        registry.register(YouTubeTranscriptTool())

        with pytest.raises(InvalidInputError):
            registry.execute("youtube_transcript", url="https://example.com/not-youtube")

    def test_execute_via_registry_preserves_specific_domain_error_type(self) -> None:
        """As of Phase 3.6, ToolRegistry.execute() passes through ANY
        OmniFlowException a tool raises unmodified (not just
        ToolExecutionError) -- fixing the Phase 3.1/3.2 registry limitation
        where TranscriptUnavailableError etc. were flattened into a generic
        ToolExecutionError, losing their real status/error code."""
        class FakeTranscriptsDisabled(Exception):
            pass

        registry = ToolRegistry()
        registry.register(YouTubeTranscriptTool())

        with patch(
            "backend.tools.youtube._UNAVAILABLE_EXCEPTIONS", (FakeTranscriptsDisabled,)
        ), patch(
            "backend.tools.youtube.YouTubeTranscriptApi",
            _mock_api_raising(FakeTranscriptsDisabled("disabled")),
        ):
            with pytest.raises(TranscriptUnavailableError) as exc_info:
                registry.execute(
                    "youtube_transcript", url=f"https://youtu.be/{VALID_VIDEO_ID}"
                )

        assert exc_info.value.status_code == 422
        assert "duration_ms" in exc_info.value.details

    def test_direct_run_call_preserves_transcript_unavailable_type(self) -> None:
        """Calling tool.run() directly (bypassing the registry) preserves the
        specific TranscriptUnavailableError type and its 422 status code."""
        class FakeTranscriptsDisabled(Exception):
            pass

        tool = YouTubeTranscriptTool()
        tool_input = YouTubeTranscriptInput(url=f"https://youtu.be/{VALID_VIDEO_ID}")

        with patch(
            "backend.tools.youtube._UNAVAILABLE_EXCEPTIONS", (FakeTranscriptsDisabled,)
        ), patch(
            "backend.tools.youtube.YouTubeTranscriptApi",
            _mock_api_raising(FakeTranscriptsDisabled("disabled")),
        ):
            with pytest.raises(TranscriptUnavailableError) as exc_info:
                tool.run(tool_input)

        assert exc_info.value.status_code == 422
