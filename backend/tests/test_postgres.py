"""Integration tests against a real PostgreSQL server.

These run only when ``CINEMATCH_TEST_POSTGRES_URL`` is set (CI does this with a
PostgreSQL 16 service container). Locally they are skipped, so the fast SQLite
suite still works without a database server.

The CI job applies the Alembic migration before running these tests; they verify
the importer, JSONB behaviour and PostgreSQL-only constraints.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.services.catalogue_import import import_catalogue

POSTGRES_URL = os.environ.get("CINEMATCH_TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(
    POSTGRES_URL is None,
    reason="set CINEMATCH_TEST_POSTGRES_URL to run PostgreSQL integration tests",
)

MOVIES_CSV = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
"""

TAGS_CSV = """userId,movieId,tag,timestamp
1,1,pixar,1131497929
"""

LINKS_CSV = """movieId,imdbId,tmdbId
1,0114709,862
"""


@pytest.fixture
def pg_engine() -> Iterator[Engine]:
    """Return an engine with an empty catalogue and registry."""

    assert POSTGRES_URL is not None
    engine = create_engine(POSTGRES_URL)
    with engine.begin() as connection:
        connection.execute(text("delete from feedback"))
        connection.execute(text("delete from ratings"))
        connection.execute(text("delete from usage_logs"))
        connection.execute(text("delete from model_versions"))
        connection.execute(text("delete from movies"))
        connection.execute(text("delete from users"))
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    (tmp_path / "movies.csv").write_text(MOVIES_CSV, encoding="utf-8")
    (tmp_path / "tags.csv").write_text(TAGS_CSV, encoding="utf-8")
    (tmp_path / "links.csv").write_text(LINKS_CSV, encoding="utf-8")
    return tmp_path


def test_import_is_idempotent_and_stores_jsonb(pg_engine: Engine, dataset: Path) -> None:
    with Session(pg_engine) as session:
        first = import_catalogue(session, dataset)
        second = import_catalogue(session, dataset)

    assert first.created == 2
    assert second.created == 0
    assert second.updated == 2

    with pg_engine.connect() as connection:
        assert connection.execute(text("select count(*) from movies")).scalar_one() == 2
        genre_type = connection.execute(
            text("select pg_typeof(genres)::text from movies limit 1")
        ).scalar_one()
        assert genre_type == "jsonb"
        animated = connection.execute(
            text("select count(*) from movies where genres @> '[\"Animation\"]'::jsonb")
        ).scalar_one()
        assert animated == 1
        tags = connection.execute(
            text("select tags from movies where ml_movie_id = 1")
        ).scalar_one()
        assert tags == ["pixar"]


def test_only_one_active_model_version(pg_engine: Engine) -> None:
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "insert into model_versions (version, artifact_path, is_active) "
                "values ('v1', 'a.joblib', true)"
            )
        )

    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        connection.execute(
            text(
                "insert into model_versions (version, artifact_path, is_active) "
                "values ('v2', 'b.joblib', true)"
            )
        )


def test_rating_check_constraint(pg_engine: Engine) -> None:
    with pg_engine.begin() as connection:
        connection.execute(
            text(
                "insert into users (email, password_hash, display_name) "
                "values ('pg@example.com', 'x', 'PG')"
            )
        )
        connection.execute(
            text(
                "insert into movies (ml_movie_id, title, genres, tags) "
                "values (1, 'Toy Story (1995)', '[]'::jsonb, '[]'::jsonb)"
            )
        )

    with pytest.raises(IntegrityError), pg_engine.begin() as connection:
        connection.execute(
            text(
                "insert into ratings (user_id, movie_id, rating, rated_at) "
                "select u.id, m.id, 5.5, now() from users u, movies m"
            )
        )
