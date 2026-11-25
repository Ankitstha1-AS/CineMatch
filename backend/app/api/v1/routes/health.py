"""Health and readiness endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Result of a liveness check."""

    status: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health() -> HealthResponse:
    """Report that the API process is up and which build is running."""

    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.environment,
    )
