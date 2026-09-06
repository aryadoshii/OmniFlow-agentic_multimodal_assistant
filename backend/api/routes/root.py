"""Root route establishing endpoint discovery and placeholder for future UI."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings, get_settings

router = APIRouter(tags=["Root"])


class RootResponse(BaseModel):
    """Metadata response for API root discovery."""

    message: str
    app_name: str
    version: str
    docs_url: str
    health_url: str


@router.get("/", response_model=RootResponse)
async def get_root(settings: Settings = Depends(get_settings)) -> RootResponse:
    """Returns root discovery information and endpoints."""
    return RootResponse(
        message="Welcome to OmniFlow - Agentic Multimodal AI Assistant API.",
        app_name=settings.app_name,
        version=settings.app_version,
        docs_url="/docs",
        health_url="/health",
    )
