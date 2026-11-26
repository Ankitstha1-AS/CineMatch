"""Versioned model registry (data store D4)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, Integer, String, false, func, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import json_type


class ModelVersion(Base):
    """One trained artifact plus its manifest, metrics and activation state."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
    manifest: Mapped[dict] = mapped_column(
        json_type(), nullable=False, default=dict, server_default=text("'{}'")
    )
    metrics: Mapped[dict] = mapped_column(
        json_type(), nullable=False, default=dict, server_default=text("'{}'")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # At most one active version at a time.
        Index(
            "uq_model_versions_one_active",
            "is_active",
            unique=True,
            postgresql_where=text("is_active"),
            sqlite_where=text("is_active = 1"),
        ),
    )
