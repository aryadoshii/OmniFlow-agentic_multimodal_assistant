"""Safe temporary file management for ephemeral upload processing."""

import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# Pattern to sanitize incoming filenames and prevent path traversal
_UNSAFE_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]")


def sanitize_filename(filename: str) -> str:
    """Strips directory traversal components and replaces unsafe characters."""
    base_name = Path(filename).name
    sanitized = _UNSAFE_CHARS_PATTERN.sub("_", base_name)
    if not sanitized or sanitized.startswith("."):
        sanitized = f"upload_{sanitized}".lstrip(".")
    return sanitized


@contextmanager
def temporary_upload_file(
    content: bytes, original_filename: str
) -> Generator[Path, None, None]:
    """Writes bytes to a secure temporary file outside the repository, guaranteeing cleanup."""
    sanitized_name = sanitize_filename(original_filename)
    suffix = Path(sanitized_name).suffix.lower()

    # Create temporary file in system temporary directory
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        prefix="omniflow_upload_",
        suffix=suffix,
    )
    temp_path = Path(temp_file.name)

    try:
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        yield temp_path
    finally:
        # Guarantee removal of temporary file even if exceptions occur during processing
        if temp_path.exists():
            try:
                os.remove(temp_path)
            except OSError as err:
                logger.warning(
                    "Failed to delete temporary file %s: %s", temp_path, err
                )
