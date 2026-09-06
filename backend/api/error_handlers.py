"""Global exception handlers mapping application errors to uniform JSON responses."""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.exceptions import OmniFlowException

logger = logging.getLogger(__name__)


async def omniflow_exception_handler(
    request: Request, exc: OmniFlowException
) -> JSONResponse:
    """Handles domain-specific OmniFlow exceptions with matching status codes."""
    logger.warning(
        "Application error on %s %s: [%s] %s",
        request.method,
        request.url.path,
        exc.error_code,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handles FastAPI/Pydantic request validation errors."""
    logger.warning(
        "Validation error on %s %s: %s",
        request.method,
        request.url.path,
        exc.errors(),
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request payload validation failed.",
                "details": {"errors": exc.errors()},
            }
        },
    )


async def global_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Catches unhandled exceptions, logs internal traceback, returns clean error."""
    logger.exception(
        "Unhandled server exception on %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred. Please contact support if the issue persists.",
                "details": {},
            }
        },
    )


def register_error_handlers(app: FastAPI) -> None:
    """Registers all exception handlers on the FastAPI application."""
    app.add_exception_handler(OmniFlowException, omniflow_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
