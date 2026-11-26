"""Movie catalogue and cached display metadata (data store D2)."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, SmallInteger, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.types import json_type


class Movie(TimestampMixin, Base):
    """A film in the catalogue.

    ``genres`` and ``tags`` are JSON arrays used by the content-based model.
    ``poster_path``/``overview``/``tmdb_id`` are display-only TMDB data and are
    never used for training.
    """

    __tablename__ = "movies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Stable MovieLens id: joins catalogue rows to the offline model artifacts.
    ml_movie_id: Mapped[int | None] = mapped_column(Integer, unique=True)
    title: Mapped[str] = mapped_column(String(512), index=True, nullable=False)
    year: Mapped[int | None] = mapped_column(SmallInteger, index=True)
    genres: Mapped[list[str]] = mapped_column(
        json_type(), nullable=False, default=list, server_default=text("'[]'")
    )
    tags: Mapped[list[str]] = mapped_column(
        json_type(), nullable=False, default=list, server_default=text("'[]'")
    )
    tmdb_id: Mapped[int | None] = mapped_column(Integer, index=True)
    poster_path: Mapped[str | None] = mapped_column(String(255))
    overview: Mapped[str | None] = mapped_column(Text)
    metadata_cached_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
