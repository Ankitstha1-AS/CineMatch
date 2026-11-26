"""Schema-level tests for the Phase 1 data model.

These run on SQLite so they need no live PostgreSQL server; the Alembic
migration is exercised separately for the real PostgreSQL dialect.
"""

from datetime import datetime

import pytest
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models import Feedback, FeedbackKind, ModelVersion, Movie, Rating, UsageLog, User, UserRole

EXPECTED_TABLES = {
    "feedback",
    "model_versions",
    "movies",
    "ratings",
    "usage_logs",
    "users",
}


def test_all_tables_are_created(engine: Engine) -> None:
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_user_defaults_are_applied(db_session: Session) -> None:
    user = User(email="viewer@example.com", password_hash="hash", display_name="Viewer")
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    assert user.id is not None
    assert user.role is UserRole.USER
    assert user.favourite_genres == []
    assert user.consent_to_study is False
    assert user.created_at is not None


def test_email_is_unique(db_session: Session) -> None:
    db_session.add(User(email="same@example.com", password_hash="a", display_name="A"))
    db_session.commit()

    db_session.add(User(email="same@example.com", password_hash="b", display_name="B"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def _user_and_movie(db_session: Session) -> tuple[User, Movie]:
    user = User(email="rater@example.com", password_hash="hash", display_name="Rater")
    movie = Movie(ml_movie_id=1, title="Toy Story (1995)", year=1995, genres=["Animation"])
    db_session.add_all([user, movie])
    db_session.commit()
    return user, movie


def test_rating_rejects_values_outside_the_scale(db_session: Session) -> None:
    user, movie = _user_and_movie(db_session)
    db_session.add(Rating(user_id=user.id, movie_id=movie.id, rating=6.0, rated_at=datetime.now()))

    with pytest.raises(IntegrityError):
        db_session.commit()


def test_a_user_can_rate_a_movie_only_once(db_session: Session) -> None:
    user, movie = _user_and_movie(db_session)
    first = Rating(user_id=user.id, movie_id=movie.id, rating=4.5, rated_at=datetime.now())
    db_session.add(first)
    db_session.commit()

    db_session.add(Rating(user_id=user.id, movie_id=movie.id, rating=1.0, rated_at=datetime.now()))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_half_star_ratings_are_stored_with_one_decimal(db_session: Session) -> None:
    user, movie = _user_and_movie(db_session)
    rating = Rating(user_id=user.id, movie_id=movie.id, rating=3.5, rated_at=datetime.now())
    db_session.add(rating)
    db_session.commit()
    db_session.refresh(rating)

    assert float(rating.rating) == 3.5


def test_only_one_model_version_can_be_active(db_session: Session) -> None:
    db_session.add(ModelVersion(version="v1", artifact_path="a.joblib", is_active=True))
    db_session.commit()

    db_session.add(ModelVersion(version="v2", artifact_path="b.joblib", is_active=True))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_inactive_model_versions_do_not_conflict(db_session: Session) -> None:
    db_session.add_all(
        [
            ModelVersion(version="v1", artifact_path="a.joblib"),
            ModelVersion(version="v2", artifact_path="b.joblib"),
        ]
    )
    db_session.commit()

    assert db_session.query(ModelVersion).count() == 2


def test_feedback_and_usage_log_rows_round_trip(db_session: Session) -> None:
    user, movie = _user_and_movie(db_session)
    db_session.add(Feedback(user_id=user.id, movie_id=movie.id, kind=FeedbackKind.UP))
    db_session.add(
        UsageLog(
            user_id=user.id,
            endpoint="/api/v1/health",
            method="GET",
            status_code=200,
            latency_ms=12,
            feature="health",
        )
    )
    db_session.commit()

    assert db_session.query(Feedback).count() == 1
    assert db_session.query(UsageLog).count() == 1
