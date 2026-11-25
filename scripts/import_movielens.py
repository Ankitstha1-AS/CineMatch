"""Load the MovieLens catalogue into CineMatch's database.

Only the catalogue is imported (movies, genres, tags, TMDB ids). MovieLens
ratings stay in the offline dataset; the ``ratings`` table is reserved for real
CineMatch accounts. See ``docs/database.md`` for the reasoning.

The import is idempotent: re-running it updates existing rows matched on
``ml_movie_id`` instead of duplicating them.

Apply the schema first:

    cd backend && alembic upgrade head

Then run from the repository root:

    python scripts/import_movielens.py data/processed/ml-25m-subset

For a quick check without PostgreSQL, point it at SQLite and let it create the
tables:

    python scripts/import_movielens.py data/processed/ml-25m-subset \\
        --database-url sqlite:///data/processed/cinematch-dev.sqlite --create-tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "backend"
for entry in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

import app.models  # noqa: E402,F401  (register every table on Base.metadata)
from app.core.config import get_settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.services.catalogue_import import import_catalogue  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Processed dataset directory containing movies.csv (e.g. data/processed/ml-25m-subset)",
    )
    parser.add_argument(
        "--database-url",
        default=None,
        help="SQLAlchemy URL; defaults to DATABASE_URL / .env / built-in dev value.",
    )
    parser.add_argument("--batch-size", type=int, default=500, help="Rows per insert batch.")
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create missing tables with SQLAlchemy instead of running Alembic (dev only).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_dir}")

    database_url = args.database_url or get_settings().database_url
    print(f"Database: {database_url}")
    print(f"Dataset : {args.dataset_dir}")

    engine = create_engine(database_url, pool_pre_ping=True)
    try:
        if args.create_tables:
            Base.metadata.create_all(engine)
            print("Created missing tables (development shortcut).")

        with Session(engine) as session:
            stats = import_catalogue(session, args.dataset_dir, batch_size=args.batch_size)
    finally:
        engine.dispose()

    print(
        "\nImported catalogue: "
        f"{stats.movies:,} movies "
        f"({stats.created:,} created, {stats.updated:,} updated), "
        f"{stats.tagged_movies:,} with tags, {stats.tmdb_linked:,} with TMDB ids."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
