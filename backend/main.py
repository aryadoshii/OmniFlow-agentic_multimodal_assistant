"""OmniFlow FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.api.error_handlers import register_error_handlers
from backend.api.routes import agent, health, history, ingest, root
from backend.config import get_settings
from backend.logging import setup_logging
from backend.services.history_store import init_db as init_history_db

logger = logging.getLogger(__name__)

# Phase 7 (Render single-service deployment): the built frontend, if
# present, at the same fixed location the deployment Dockerfile copies it
# to (frontend/dist, relative to the repo root -- this file lives at
# <repo>/backend/main.py, so parent.parent is <repo>).
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context managing startup and shutdown tasks."""
    settings = get_settings()
    setup_logging(log_level=settings.log_level)
    init_history_db()
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
        description="Agentic Multimodal AI Assistant",
        lifespan=lifespan,
    )

    # Register centralized exception handlers
    register_error_handlers(app)

    # Register modular routers. /health, /ingest, /query are always the
    # backend's real API surface, unaffected by the frontend below.
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(agent.router)
    app.include_router(history.router)

    # Single-service deployment (Phase 7): only when explicitly running in
    # production AND the frontend has actually been built (the deployment
    # Dockerfile builds it and copies it to frontend/dist) does "/" serve
    # the built SPA instead of the JSON discovery response -- so local
    # dev/test runs (APP_ENV defaults to "development", and the test suite's
    # own settings always use "test") are completely unaffected regardless
    # of whether a stray local `npm run build` output happens to exist on
    # disk. This decision is made once, at app-construction time, exactly
    # like a real deployed container would (APP_ENV=production is part of
    # its actual startup environment) -- not per-request.
    if settings.is_production and _FRONTEND_DIST.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=_FRONTEND_DIST / "assets"),
            name="frontend-assets",
        )

        @app.get("/", include_in_schema=False)
        async def serve_frontend_index() -> FileResponse:
            return FileResponse(_FRONTEND_DIST / "index.html")
    else:
        app.include_router(root.router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.is_development,
    )
