"""Thumbs up/down feedback events (data store D3)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.db.types import big_int_pk_type


class FeedbackKind(StrEnum):
    """Direction of a feedback event."""

    UP = "up"
    DOWN = "down"


class Feedback(Base):
    """A single thumbs up/down event; stored for the next retraining run."""

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(big_int_pk_type(), primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), index=True, nullable=False
    )
    kind: Mapped[FeedbackKind] = mapped_column(
        Enum(
            FeedbackKind,
            name="feedback_kind",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
