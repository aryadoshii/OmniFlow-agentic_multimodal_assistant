"""Image processor for JPG/PNG files using Pillow and Tesseract OCR."""

import io
import logging
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from omniflow.exceptions import ProcessingFailureError
from omniflow.models.document import (
    ExtractionMethod,
    NormalizedDocument,
    SourceType,
)
from omniflow.processors.base import BaseProcessor
from omniflow.services.ocr_service import OCRService

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_MIMES = {"image/jpeg", "image/png"}
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ImageProcessor(BaseProcessor):
    """Ingests JPG, JPEG, and PNG images and extracts text using local OCR."""

    def __init__(self, ocr_service: OCRService | None = None) -> None:
        self.ocr_service = ocr_service or OCRService()

    def can_process(self, mime_type: str, filename: str | None = None) -> bool:
        """Determines if the file is a supported image format."""
        if mime_type in SUPPORTED_IMAGE_MIMES:
            return True
        if filename:
            suffix = Path(filename).suffix.lower()
            return suffix in SUPPORTED_IMAGE_EXTS
        return False

    def process(
        self, file_bytes: bytes, filename: str, mime_type: str
    ) -> NormalizedDocument:
        """Validates the image with Pillow, runs OCR, and builds NormalizedDocument."""
        if not file_bytes:
            raise ProcessingFailureError(
                f"Image file '{filename}' is empty.",
                details={"filename": filename},
            )

        # Validate image decoding with Pillow
        try:
            image = Image.open(io.BytesIO(file_bytes))
            image.verify()  # Verifies file integrity
            # Re-open image for processing after verify()
            image = Image.open(io.BytesIO(file_bytes))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            logger.error("Corrupt or invalid image file '%s': %s", filename, exc)
            raise ProcessingFailureError(
                f"Invalid or corrupted image file '{filename}': {exc}",
                details={"filename": filename, "error": str(exc)},
            ) from exc

        width, height = image.size
        img_format = image.format or "UNKNOWN"

        # Preprocessing: convert palette or alpha images to RGB for optimal OCR compatibility
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")

        # Execute OCR
        extracted_text, confidence, ocr_meta = (
            self.ocr_service.extract_text_with_confidence(image)
        )

        warnings: list[str] = []
        if not extracted_text:
            warnings.append(f"No text recognized in image '{filename}'.")

        logger.info(
            "Processed image '%s' (%dx%d %s): %d words extracted",
            filename,
            width,
            height,
            img_format,
            ocr_meta.get("word_count", 0),
        )

        metadata = {
            "width": width,
            "height": height,
            "format": img_format,
            **ocr_meta,
        }

        canonical_mime = "image/jpeg" if Path(filename).suffix.lower() in {".jpg", ".jpeg"} else "image/png"

        return NormalizedDocument(
            filename=filename,
            source_type=SourceType.IMAGE,
            mime_type=canonical_mime,
            content=extracted_text,
            extraction_method=ExtractionMethod.OCR,
            confidence=confidence,
            metadata=metadata,
            warnings=warnings,
        )
