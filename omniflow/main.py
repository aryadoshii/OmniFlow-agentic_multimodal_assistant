"""OmniFlow FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI

from omniflow.api.error_handlers import register_error_handlers
from omniflow.api.routes import health, ingest, root
from omniflow.config import get_settings
from omniflow.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context managing startup and shutdown tasks."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    logger.info(
        "Starting %s v%s in %s environment",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app() -> FastAPI:
    """Application factory for OmniFlow."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Agentic Multimodal AI Assistant - Ingestion Gateway",
        lifespan=lifespan,
    )

    # Register centralized exception handlers
    register_error_handlers(app)

    # Register modular routers
    app.include_router(root.router)
    app.include_router(health.router)
    app.include_router(ingest.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "omniflow.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
