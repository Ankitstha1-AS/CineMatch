"""Portable column types.

Production runs PostgreSQL; tests run SQLite. SQLite only autoincrements a
column declared exactly ``INTEGER PRIMARY KEY``, so big integer primary keys
need a dialect variant to stay usable in tests.
"""

from sqlalchemy import JSON, BigInteger, Integer
from sqlalchemy.dialects.postgresql import JSONB


def json_type() -> JSONB:
    """Return a JSONB type with a SQLite-compatible variant."""

    return JSONB().with_variant(JSON(), "sqlite")


def big_int_pk_type() -> BigInteger:
    """Return a BIGINT type that autoincrements on SQLite in tests."""

    return BigInteger().with_variant(Integer(), "sqlite")
