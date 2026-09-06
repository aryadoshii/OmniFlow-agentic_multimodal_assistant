"""PDF processor utilizing PyMuPDF with per-page OCR fallback for scanned pages."""

import io
import logging
from pathlib import Path
from typing import Any
from PIL import Image
import pymupdf
from backend.config import get_settings
from backend.exceptions import InvalidInputError, ProcessingFailureError
from backend.models.document import (
    ExtractionMethod,
    NormalizedDocument,
    SourceType,
)
from backend.processors.base import BaseProcessor
from backend.services.ocr_service import OCRService

logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """Processes PDF documents using native text extraction with per-page OCR fallback."""

    def __init__(self, ocr_service: OCRService | None = None) -> None:
        self.ocr_service = ocr_service or OCRService()
        settings = get_settings()
        self.char_threshold = settings.pdf_native_text_char_threshold

    def can_process(self, mime_type: str, filename: str | None = None) -> bool:
        """Determines if the file is a PDF."""
        if mime_type == "application/pdf":
            return True
        if filename:
            suffix = Path(filename).suffix.lower()
            return suffix == ".pdf"
        return False

    def process(
        self, file_bytes: bytes, filename: str, mime_type: str = "application/pdf"
    ) -> NormalizedDocument:
        """Extracts text page-by-page, rendering to image and falling back to OCR when necessary."""
        if not file_bytes:
            raise ProcessingFailureError(
                f"PDF file '{filename}' is empty.",
                details={"filename": filename},
            )

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("Failed to parse PDF '%s': %s", filename, exc)
            raise ProcessingFailureError(
                f"Failed to parse PDF file '{filename}': {exc}",
                details={"filename": filename, "error": str(exc)},
            ) from exc

        try:
            # Check for encrypted or password-protected PDF
            if doc.is_encrypted:
                logger.warning("PDF '%s' is password-protected or encrypted", filename)
                raise ProcessingFailureError(
                    f"PDF document '{filename}' is password-protected or encrypted.",
                    details={"filename": filename, "encrypted": True},
                )

            total_pages = len(doc)
            if total_pages == 0:
                raise InvalidInputError(
                    f"PDF document '{filename}' contains 0 pages.",
                    details={"filename": filename},
                )

            pages_meta: list[dict[str, Any]] = []
            page_contents: list[str] = []
            warnings: list[str] = []
            native_page_count = 0
            ocr_page_count = 0
            all_confidences: list[float] = []

            for page_index in range(total_pages):
                page_num = page_index + 1
                page = doc[page_index]

                # 1. Attempt native text extraction
                native_text = page.get_text().strip()

                if len(native_text) >= self.char_threshold:
                    # Meaningful native text exists
                    native_page_count += 1
                    page_contents.append(native_text)
                    pages_meta.append({
                        "page_number": page_num,
                        "extraction_method": "native_text",
                        "character_count": len(native_text),
                    })
                else:
                    # Page is likely scanned or empty: render and run OCR fallback
                    try:
                        pix = page.get_pixmap(dpi=150)
                        img = Image.open(io.BytesIO(pix.tobytes("png")))
                        ocr_text, ocr_conf, ocr_details = (
                            self.ocr_service.extract_text_with_confidence(img)
                        )

                        if ocr_text:
                            ocr_page_count += 1
                            page_contents.append(ocr_text)
                            if ocr_conf is not None:
                                all_confidences.append(ocr_conf)
                            pages_meta.append({
                                "page_number": page_num,
                                "extraction_method": "ocr",
                                "character_count": len(ocr_text),
                                "confidence": ocr_conf,
                            })
                        else:
                            # Both native and OCR yielded no text for this page
                            if native_text:
                                page_contents.append(native_text)
                                pages_meta.append({
                                    "page_number": page_num,
                                    "extraction_method": "native_text",
                                    "character_count": len(native_text),
                                })
                            else:
                                warnings.append(f"Page {page_num} contains no extractable text.")
                                pages_meta.append({
                                    "page_number": page_num,
                                    "extraction_method": "none",
                                    "character_count": 0,
                                })

                    except Exception as err:
                        # Individual page OCR failure must not destroy rest of document
                        logger.warning(
                            "OCR fallback failed on PDF '%s' page %d: %s",
                            filename,
                            page_num,
                            err,
                        )
                        warnings.append(
                            f"OCR fallback failed on page {page_num}: {err}"
                        )
                        if native_text:
                            page_contents.append(native_text)
                        pages_meta.append({
                            "page_number": page_num,
                            "extraction_method": "failed_ocr",
                            "error": str(err),
                        })

            # Determine aggregate extraction method
            if ocr_page_count > 0 and native_page_count > 0:
                method = ExtractionMethod.MIXED
            elif ocr_page_count > 0:
                method = ExtractionMethod.OCR
            else:
                method = ExtractionMethod.NATIVE_TEXT

            combined_content = "\n\n".join(page_contents).strip()
            if not combined_content:
                warnings.append(f"No extractable text found in PDF '{filename}'.")

            avg_confidence: float | None = None
            if all_confidences:
                avg_confidence = round(
                    sum(all_confidences) / len(all_confidences), 4
                )
            elif method == ExtractionMethod.NATIVE_TEXT and combined_content:
                avg_confidence = 1.0

            metadata: dict[str, Any] = {
                "total_pages": total_pages,
                "native_pages": native_page_count,
                "ocr_pages": ocr_page_count,
                "pages": pages_meta,
            }

            logger.info(
                "Processed PDF '%s' (%d pages): %d native, %d OCR",
                filename,
                total_pages,
                native_page_count,
                ocr_page_count,
            )

            return NormalizedDocument(
                filename=filename,
                source_type=SourceType.PDF,
                mime_type="application/pdf",
                content=combined_content,
                extraction_method=method,
                confidence=avg_confidence,
                metadata=metadata,
                warnings=warnings,
            )

        finally:
            doc.close()
