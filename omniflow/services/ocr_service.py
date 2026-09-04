"""Reusable OCR service utilizing Tesseract."""

import logging
from typing import Any
from PIL import Image
import pytesseract
from omniflow.config import get_settings
from omniflow.exceptions import OCRProcessingError

logger = logging.getLogger(__name__)


class OCRService:
    """Service encapsulating Tesseract OCR operations, confidence calculation, and error handling."""

    def __init__(self) -> None:
        settings = get_settings()
        self.language = settings.ocr_language
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    def extract_text_with_confidence(
        self, image: Image.Image
    ) -> tuple[str, float | None, dict[str, Any]]:
        """Performs OCR on a PIL Image, returning (text, confidence, metadata)."""
        try:
            # Extract detailed OCR data to compute genuine aggregate confidence
            ocr_data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT,
            )

            # Filter valid word confidences (Tesseract uses -1 for layout/blocks)
            confidences: list[float] = []
            words: list[str] = []

            for i, word in enumerate(ocr_data.get("text", [])):
                cleaned_word = word.strip()
                if cleaned_word:
                    words.append(cleaned_word)
                    try:
                        conf_val = float(ocr_data["conf"][i])
                        if conf_val >= 0.0:
                            confidences.append(conf_val)
                    except (KeyError, IndexError, ValueError):
                        pass

            extracted_text = " ".join(words).strip()

            # Calculate average confidence normalized to [0.0, 1.0] if words were recognized
            confidence: float | None = None
            if confidences:
                confidence = round(sum(confidences) / len(confidences) / 100.0, 4)

            metadata: dict[str, Any] = {
                "ocr_engine": "tesseract",
                "ocr_language": self.language,
                "word_count": len(words),
            }

            return extracted_text, confidence, metadata

        except pytesseract.TesseractNotFoundError as exc:
            logger.error("Tesseract OCR binary not found: %s", exc)
            raise OCRProcessingError(
                "Tesseract OCR binary is not installed or not in PATH.",
                details={"dependency": "tesseract"},
            ) from exc
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract execution failed: %s", exc)
            raise OCRProcessingError(
                f"Tesseract OCR processing failed: {exc}",
                details={"error": str(exc)},
            ) from exc
        except Exception as exc:
            logger.error("Unexpected failure during OCR processing: %s", exc)
            raise OCRProcessingError(
                f"Failed to perform OCR on image: {exc}",
                details={"error": str(exc)},
            ) from exc
