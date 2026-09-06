"""Health check endpoint for OmniFlow."""

from typing import Literal
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings, get_settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Structured response schema for health checks."""

    status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    app_name: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Returns application health status, version, and running environment."""
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
    )
