"""Tests for the database session plumbing."""

from app.core.config import get_settings
from app.db.session import SessionLocal, engine


def test_session_factory_binds_to_the_configured_database() -> None:
    assert str(engine.url.render_as_string(hide_password=False)) == get_settings().database_url
    assert SessionLocal.kw["bind"] is engine
