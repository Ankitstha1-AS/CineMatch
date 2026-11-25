"""User accounts and profiles (data store D1)."""

from __future__ import annotations

from enum import StrEnum

from sqlalchemy import Boolean, Enum, Integer, String, false, text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin
from app.db.types import json_type


class UserRole(StrEnum):
    """Authorisation roles."""

    USER = "user"
    ADMIN = "admin"


class User(TimestampMixin, Base):
    """A registered viewer."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(
            UserRole,
            name="user_role",
            values_callable=lambda enum: [member.value for member in enum],
        ),
        nullable=False,
        default=UserRole.USER,
        server_default=UserRole.USER.value,
    )
    favourite_genres: Mapped[list[str]] = mapped_column(
        json_type(), nullable=False, default=list, server_default=text("'[]'")
    )
    consent_to_study: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
