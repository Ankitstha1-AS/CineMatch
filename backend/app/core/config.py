"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings.

    Values are read from environment variables first, then from a local ``.env``
    file (the repo-root file when running from ``backend/``). Never commit real
    secrets; copy ``.env.example`` to ``.env`` and fill it in.
    """

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CineMatch API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"

    database_url: str = "postgresql+psycopg://cinematch:cinematch@localhost:5432/cinematch"
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    # Auth settings are wired up in Phase 4; the dev default is not for production.
    secret_key: str = "insecure-development-key-change-me"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # TMDB is used for display metadata only, never for model training.
    tmdb_api_key: str | None = None


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
