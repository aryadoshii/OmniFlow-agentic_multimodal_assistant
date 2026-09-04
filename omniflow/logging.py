"""Centralized logging configuration and utilities for OmniFlow."""

import logging
import re
import sys
import time
from contextlib import contextmanager
from typing import Generator

# Regex patterns to prevent accidental logging of API keys and common credentials
_SENSITIVE_PATTERNS = [
    re.compile(r"(AIza[0-9A-Za-z-_]{35})"),                      # Google API keys
    re.compile(r"(bearer\s+[A-Za-z0-9\-\._~\+\/]+=*)", re.I),   # Bearer tokens
    re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([^'\"\s]+)", re.I),
]


class SanitizingFilter(logging.Filter):
    """Logging filter that scrubs sensitive patterns (keys, tokens) from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            sanitized = record.msg
            for pattern in _SENSITIVE_PATTERNS:
                sanitized = pattern.sub(r"\1***REDACTED***", sanitized)
            record.msg = sanitized
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """Configures centralized root logging with formatting and sanitization."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup is invoked multiple times
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SanitizingFilter())
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)
            handler.addFilter(SanitizingFilter())

    # Quiet overly chatty external libraries if any
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@contextmanager
def log_execution_time(
    logger: logging.Logger, operation_name: str
) -> Generator[None, None, None]:
    """Context manager to measure and log operation duration safely."""
    start_time = time.perf_counter()
    logger.debug("Started operation: %s", operation_name)
    try:
        yield
    except Exception as exc:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            "Failed operation: %s after %.2fms with error: %s",
            operation_name,
            duration_ms,
            str(exc),
        )
        raise
    else:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            "Completed operation: %s in %.2fms",
            operation_name,
            duration_ms,
        )
