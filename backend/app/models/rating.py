"""Ratings and reviews (data store D3)."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.types import big_int_pk_type


class Rating(TimestampMixin, Base):
    """One user's half-star rating of one movie.

    ``rated_at`` is the time the viewer gave the rating (used by the temporal
    split); ``created_at`` is when the row reached this database.
    """

    __tablename__ = "ratings"

    id: Mapped[int] = mapped_column(big_int_pk_type(), primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    rating: Mapped[Decimal] = mapped_column(Numeric(2, 1), nullable=False)
    review: Mapped[str | None] = mapped_column(Text)
    rated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="ck_ratings_value"),
        UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
        Index("ix_ratings_user_rated_at", "user_id", "rated_at"),
    )
