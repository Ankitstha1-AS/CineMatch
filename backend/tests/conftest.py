"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import app.models  # noqa: F401  (register every table on Base.metadata)
from app.db.base import Base
from app.main import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    """Return a test client bound to the application."""

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def engine() -> Iterator[Engine]:
    """Create the full schema in an in-memory SQLite database."""

    test_engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    try:
        yield test_engine
    finally:
        test_engine.dispose()


@pytest.fixture
def db_session(engine: Engine) -> Iterator[Session]:
    """Return a session bound to the test schema."""

    with Session(engine, expire_on_commit=False) as session:
        yield session
