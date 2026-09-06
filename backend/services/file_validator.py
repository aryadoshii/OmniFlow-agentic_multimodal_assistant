"""Centralized file validation and format detection."""

import logging
from pathlib import Path
from backend.exceptions import UnsupportedFileError, UploadValidationError

logger = logging.getLogger(__name__)

# Allowed file extensions mapped to canonical MIME types
EXTENSION_TO_MIME = {
    ".txt": "text/plain",
    ".pdf": "application/pdf",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
}

# Plausible MIME types accepted per extension
ALLOWED_MIMES = {
    "text/plain",
    "application/pdf",
    "image/jpeg",
    "image/png",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/x-m4a",
    "audio/m4a",
    "application/octet-stream",  # Generic browser fallback
}


def validate_upload(
    filename: str,
    content: bytes,
    declared_mime_type: str | None,
    max_size_mb: int,
) -> str:
    """Validates an uploaded file's size, extension, and content availability.

    Uses a practical combination of extension and MIME inspection without rejecting
    valid files solely because of browser MIME discrepancies. Returns a canonical MIME type.
    """
    if not filename:
        raise UploadValidationError("Filename cannot be empty.")

    file_size = len(content)
    if file_size == 0:
        raise UploadValidationError(
            f"Uploaded file '{filename}' is empty (0 bytes).",
            details={"filename": filename},
        )

    max_bytes = max_size_mb * 1024 * 1024
    if file_size > max_bytes:
        raise UploadValidationError(
            f"File '{filename}' size ({file_size / (1024 * 1024):.2f} MB) exceeds maximum allowed limit of {max_size_mb} MB.",
            details={"filename": filename, "size_bytes": file_size, "limit_mb": max_size_mb},
        )

    suffix = Path(filename).suffix.lower()

    # Check extension match
    if suffix in EXTENSION_TO_MIME:
        canonical_mime = EXTENSION_TO_MIME[suffix]
        # If the browser provided a plausible specific MIME type, log it
        if declared_mime_type and declared_mime_type in ALLOWED_MIMES:
            logger.debug(
                "Validated upload '%s' with ext '%s' and declared MIME '%s'",
                filename,
                suffix,
                declared_mime_type,
            )
        return canonical_mime

    # Check if declared MIME type matches a known format even if extension was unconventional
    mime_to_canonical = {
        "text/plain": "text/plain",
        "application/pdf": "application/pdf",
        "image/jpeg": "image/jpeg",
        "image/png": "image/png",
        "audio/wav": "audio/wav",
        "audio/x-wav": "audio/wav",
        "audio/mpeg": "audio/mpeg",
        "audio/mp3": "audio/mpeg",
        "audio/mp4": "audio/mp4",
        "audio/x-m4a": "audio/mp4",
        "audio/m4a": "audio/mp4",
    }

    if declared_mime_type and declared_mime_type in mime_to_canonical:
        return mime_to_canonical[declared_mime_type]

    raise UnsupportedFileError(
        f"File format for '{filename}' is not supported. Supported extensions: "
        f"{', '.join(sorted(EXTENSION_TO_MIME.keys()))}",
        details={
            "filename": filename,
            "extension": suffix,
            "declared_mime": declared_mime_type,
        },
    )
