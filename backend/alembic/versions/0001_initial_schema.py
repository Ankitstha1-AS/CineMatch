"""Initial schema for CineMatch (D1-D5).

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-25

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    user_role = sa.Enum("user", "admin", name="user_role")
    feedback_kind = sa.Enum("up", "down", name="feedback_kind")

    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("display_name", sa.String(length=80), nullable=False),
        sa.Column("role", user_role, server_default="user", nullable=False),
        sa.Column(
            "favourite_genres",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "consent_to_study",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "movies",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ml_movie_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("year", sa.SmallInteger(), nullable=True),
        sa.Column(
            "genres",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column("tmdb_id", sa.Integer(), nullable=True),
        sa.Column("poster_path", sa.String(length=255), nullable=True),
        sa.Column("overview", sa.Text(), nullable=True),
        sa.Column("metadata_cached_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_movies_ml_movie_id", "movies", ["ml_movie_id"], unique=True)
    op.create_index("ix_movies_title", "movies", ["title"])
    op.create_index("ix_movies_year", "movies", ["year"])
    op.create_index("ix_movies_tmdb_id", "movies", ["tmdb_id"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("artifact_path", sa.String(length=512), nullable=False),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_model_versions_version", "model_versions", ["version"], unique=True)
    op.create_index(
        "uq_model_versions_one_active",
        "model_versions",
        ["is_active"],
        unique=True,
        postgresql_where=sa.text("is_active"),
    )

    op.create_table(
        "ratings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("rating", sa.Numeric(precision=2, scale=1), nullable=False),
        sa.Column("review", sa.Text(), nullable=True),
        sa.Column("rated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="ck_ratings_value"),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "movie_id", name="uq_ratings_user_movie"),
    )
    op.create_index("ix_ratings_user_id", "ratings", ["user_id"])
    op.create_index("ix_ratings_movie_id", "ratings", ["movie_id"])
    op.create_index("ix_ratings_user_rated_at", "ratings", ["user_id", "rated_at"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("kind", feedback_kind, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_user_id", "feedback", ["user_id"])
    op.create_index("ix_feedback_movie_id", "feedback", ["movie_id"])

    op.create_table(
        "usage_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("endpoint", sa.String(length=255), nullable=False),
        sa.Column("method", sa.String(length=10), nullable=False),
        sa.Column("status_code", sa.SmallInteger(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("feature", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_usage_logs_user_id", "usage_logs", ["user_id"])
    op.create_index("ix_usage_logs_endpoint", "usage_logs", ["endpoint"])
    op.create_index("ix_usage_logs_created_at", "usage_logs", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_usage_logs_created_at", table_name="usage_logs")
    op.drop_index("ix_usage_logs_endpoint", table_name="usage_logs")
    op.drop_index("ix_usage_logs_user_id", table_name="usage_logs")
    op.drop_table("usage_logs")

    op.drop_index("ix_feedback_movie_id", table_name="feedback")
    op.drop_index("ix_feedback_user_id", table_name="feedback")
    op.drop_table("feedback")

    op.drop_index("ix_ratings_user_rated_at", table_name="ratings")
    op.drop_index("ix_ratings_movie_id", table_name="ratings")
    op.drop_index("ix_ratings_user_id", table_name="ratings")
    op.drop_table("ratings")

    op.drop_index("uq_model_versions_one_active", table_name="model_versions")
    op.drop_index("ix_model_versions_version", table_name="model_versions")
    op.drop_table("model_versions")

    op.drop_index("ix_movies_tmdb_id", table_name="movies")
    op.drop_index("ix_movies_year", table_name="movies")
    op.drop_index("ix_movies_title", table_name="movies")
    op.drop_index("ix_movies_ml_movie_id", table_name="movies")
    op.drop_table("movies")

    op.drop_index("ix_users_email", table_name="users")
    op.drop_table("users")

    op.execute("DROP TYPE IF EXISTS feedback_kind")
    op.execute("DROP TYPE IF EXISTS user_role")
