"""Unit tests for multimodal processors (Text, Image, PDF, Audio)."""

import io
import wave
from unittest.mock import MagicMock
from PIL import Image
import pymupdf
import pytest
from omniflow.exceptions import (
    InvalidInputError,
    ProcessingFailureError,
    TranscriptionError,
    UnsupportedFileError,
)
from omniflow.models.document import ExtractionMethod, SourceType
from omniflow.processors.audio_processor import AudioProcessor
from omniflow.processors.image_processor import ImageProcessor
from omniflow.processors.pdf_processor import PDFProcessor
from omniflow.processors.text_processor import TextProcessor
from omniflow.services.ocr_service import OCRService
from omniflow.services.whisper_service import WhisperService


# ============================================================================
# Helpers & Fixtures
# ============================================================================


def create_sample_image_bytes(format: str = "PNG", text: str = "Test") -> bytes:
    """Generates an in-memory image byte string."""
    image = Image.new("RGB", (120, 60), color=(255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format=format)
    return buf.getvalue()


def create_sample_pdf_bytes(pages: list[str]) -> bytes:
    """Generates an in-memory native text PDF using PyMuPDF."""
    doc = pymupdf.open()
    for page_text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), page_text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_scanned_pdf_bytes() -> bytes:
    """Generates a PDF containing only an embedded image with no native text."""
    doc = pymupdf.open()
    page = doc.new_page()
    img_bytes = create_sample_image_bytes("PNG")
    rect = pymupdf.Rect(50, 50, 200, 200)
    page.insert_image(rect, stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_mixed_pdf_bytes() -> bytes:
    """Generates a PDF where page 1 has native text and page 2 is a scanned image."""
    doc = pymupdf.open()
    # Page 1: Native text
    page1 = doc.new_page()
    page1.insert_text(
        (50, 72), "This is page one with plenty of native text to exceed the threshold."
    )
    # Page 2: Image only (scanned)
    page2 = doc.new_page()
    img_bytes = create_sample_image_bytes("PNG")
    page2.insert_image(pymupdf.Rect(50, 50, 200, 200), stream=img_bytes)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_sample_wav_bytes(duration_sec: float = 0.5) -> bytes:
    """Generates a minimal valid WAV audio file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        num_frames = int(16000 * duration_sec)
        wav_file.writeframes(b"\x00\x00" * num_frames)
    return buf.getvalue()


# ============================================================================
# TextProcessor Tests
# ============================================================================


def test_text_processor_success() -> None:
    processor = TextProcessor()
    assert processor.can_process("text/plain", "doc.txt") is True
    assert processor.can_process("application/pdf", "doc.pdf") is False

    doc = processor.process(b"  Hello OmniFlow!\n\nSecond line.  ", "notes.txt")
    assert doc.source_type == SourceType.TEXT
    assert doc.extraction_method == ExtractionMethod.DIRECT_INPUT
    assert doc.content == "Hello OmniFlow!\n\nSecond line."
    assert doc.metadata["character_count"] > 0
    assert doc.confidence == 1.0


def test_text_processor_empty_rejection() -> None:
    processor = TextProcessor()
    with pytest.raises(InvalidInputError):
        processor.process(b"", "empty.txt")

    with pytest.raises(InvalidInputError):
        processor.process(b"   \n\n   \t  ", "spaces.txt")


# ============================================================================
# ImageProcessor Tests
# ============================================================================


def test_image_processor_can_process() -> None:
    processor = ImageProcessor()
    assert processor.can_process("image/png", "pic.png") is True
    assert processor.can_process("image/jpeg", "pic.jpg") is True
    assert processor.can_process("application/pdf", "pic.pdf") is False


def test_image_processor_valid_ocr() -> None:
    mock_ocr = MagicMock(spec=OCRService)
    mock_ocr.extract_text_with_confidence.return_value = (
        "Extracted Invoice Text",
        0.95,
        {"word_count": 3},
    )

    processor = ImageProcessor(ocr_service=mock_ocr)
    img_bytes = create_sample_image_bytes("PNG")
    doc = processor.process(img_bytes, "invoice.png", "image/png")

    assert doc.source_type == SourceType.IMAGE
    assert doc.extraction_method == ExtractionMethod.OCR
    assert doc.content == "Extracted Invoice Text"
    assert doc.confidence == 0.95
    assert doc.metadata["word_count"] == 3
    assert doc.warnings == []


def test_image_processor_empty_ocr_result() -> None:
    mock_ocr = MagicMock(spec=OCRService)
    mock_ocr.extract_text_with_confidence.return_value = ("", None, {"word_count": 0})

    processor = ImageProcessor(ocr_service=mock_ocr)
    img_bytes = create_sample_image_bytes("JPEG")
    doc = processor.process(img_bytes, "blank.jpg", "image/jpeg")

    assert doc.content == ""
    assert len(doc.warnings) == 1
    assert "No text recognized" in doc.warnings[0]


def test_image_processor_corrupt_image() -> None:
    processor = ImageProcessor()
    with pytest.raises(ProcessingFailureError):
        processor.process(b"not an image at all", "corrupt.png", "image/png")


# ============================================================================
# PDFProcessor Tests
# ============================================================================


def test_pdf_processor_native_text() -> None:
    processor = PDFProcessor()
    assert processor.can_process("application/pdf", "file.pdf") is True

    pdf_bytes = create_sample_pdf_bytes([
        "Page 1: Deep learning architectures and agentic workflows.",
        "Page 2: Vector retrieval and multimodal normalized documents.",
    ])

    doc = processor.process(pdf_bytes, "paper.pdf")
    assert doc.source_type == SourceType.PDF
    assert doc.extraction_method == ExtractionMethod.NATIVE_TEXT
    assert "Page 1: Deep learning" in doc.content
    assert "Page 2: Vector retrieval" in doc.content
    assert doc.metadata["total_pages"] == 2
    assert doc.metadata["native_pages"] == 2
    assert doc.metadata["ocr_pages"] == 0
    assert doc.confidence == 1.0


def test_pdf_processor_scanned_ocr_fallback() -> None:
    mock_ocr = MagicMock(spec=OCRService)
    mock_ocr.extract_text_with_confidence.return_value = (
        "Scanned Receipts Data",
        0.88,
        {"word_count": 3},
    )

    processor = PDFProcessor(ocr_service=mock_ocr)
    pdf_bytes = create_scanned_pdf_bytes()

    doc = processor.process(pdf_bytes, "scanned.pdf")
    assert doc.source_type == SourceType.PDF
    assert doc.extraction_method == ExtractionMethod.OCR
    assert doc.content == "Scanned Receipts Data"
    assert doc.metadata["total_pages"] == 1
    assert doc.metadata["ocr_pages"] == 1
    assert doc.metadata["native_pages"] == 0
    assert doc.confidence == 0.88


def test_pdf_processor_mixed_pages() -> None:
    mock_ocr = MagicMock(spec=OCRService)
    mock_ocr.extract_text_with_confidence.return_value = (
        "OCR text from page 2",
        0.90,
        {"word_count": 5},
    )

    processor = PDFProcessor(ocr_service=mock_ocr)
    pdf_bytes = create_mixed_pdf_bytes()

    doc = processor.process(pdf_bytes, "mixed.pdf")
    assert doc.source_type == SourceType.PDF
    assert doc.extraction_method == ExtractionMethod.MIXED
    assert "This is page one" in doc.content
    assert "OCR text from page 2" in doc.content
    assert doc.metadata["total_pages"] == 2
    assert doc.metadata["native_pages"] == 1
    assert doc.metadata["ocr_pages"] == 1


def test_pdf_processor_corrupt_file() -> None:
    processor = PDFProcessor()
    with pytest.raises(ProcessingFailureError):
        processor.process(b"not a valid pdf", "bad.pdf")


def test_pdf_processor_empty_bytes() -> None:
    processor = PDFProcessor()
    with pytest.raises(ProcessingFailureError):
        processor.process(b"", "empty.pdf")


# ============================================================================
# AudioProcessor Tests
# ============================================================================


def test_audio_processor_can_process() -> None:
    processor = AudioProcessor()
    assert processor.can_process("audio/wav", "recording.wav") is True
    assert processor.can_process("audio/mpeg", "recording.mp3") is True
    assert processor.can_process("audio/mp4", "recording.m4a") is True
    assert processor.can_process("image/png", "pic.png") is False


def test_audio_processor_transcription_success() -> None:
    mock_whisper = MagicMock(spec=WhisperService)
    mock_whisper.transcribe.return_value = (
        "Meeting summary discussing product launch schedule.",
        None,
        {"duration_seconds": 3.5, "language": "en"},
    )

    processor = AudioProcessor(whisper_service=mock_whisper)
    wav_bytes = create_sample_wav_bytes(0.5)

    doc = processor.process(wav_bytes, "meeting.wav", "audio/wav")
    assert doc.source_type == SourceType.AUDIO
    assert doc.extraction_method == ExtractionMethod.SPEECH_TO_TEXT
    assert doc.content == "Meeting summary discussing product launch schedule."
    assert doc.metadata["duration_seconds"] == 3.5
    assert doc.warnings == []


def test_audio_processor_transcription_failure() -> None:
    mock_whisper = MagicMock(spec=WhisperService)
    mock_whisper.transcribe.side_effect = TranscriptionError("Backend failed")

    processor = AudioProcessor(whisper_service=mock_whisper)
    wav_bytes = create_sample_wav_bytes(0.2)

    with pytest.raises(TranscriptionError):
        processor.process(wav_bytes, "call.wav", "audio/wav")


def test_audio_processor_unsupported_format() -> None:
    processor = AudioProcessor()
    with pytest.raises(UnsupportedFileError):
        processor.process(b"dummy data", "audio.flac", "audio/flac")
