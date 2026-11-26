"""Import the MovieLens catalogue into the ``movies`` table.

The importer loads only the **catalogue** (metadata, genres, tags, TMDB id).
MovieLens ratings deliberately stay in the offline dataset: the ``ratings``
table holds ratings given by real CineMatch accounts, and the model artifacts
are trained from the files in ``data/processed/``. This keeps the online and
offline data paths separate, as required by the architecture.

The function is idempotent: it matches rows on ``ml_movie_id`` and updates
existing movies instead of inserting duplicates.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from app.models import Movie

YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")
NO_GENRES = "(no genres listed)"
GENRE_SEPARATOR = "|"
BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class MovieRow:
    """One row of ``movies.csv`` after parsing."""

    ml_movie_id: int
    title: str
    year: int | None
    genres: list[str]


@dataclass(frozen=True, slots=True)
class CatalogueStats:
    """Summary of one import run."""

    movies: int
    created: int
    updated: int
    tagged_movies: int
    tmdb_linked: int


def parse_year(title: str) -> int | None:
    """Return the release year embedded in a MovieLens title."""

    match = YEAR_PATTERN.search(title)
    return int(match.group(1)) if match else None


def parse_genres(raw: str) -> list[str]:
    """Split the pipe-separated genre field, dropping the sentinel value."""

    if not raw or raw == NO_GENRES:
        return []
    return [genre for genre in raw.split(GENRE_SEPARATOR) if genre]


def _chunks(items: list[dict[str, object]], size: int) -> Iterator[list[dict[str, object]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def read_movies(path: Path) -> list[MovieRow]:
    """Parse ``movies.csv`` into rows, preserving file order."""

    rows: list[MovieRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                MovieRow(
                    ml_movie_id=int(record["movieId"]),
                    title=record["title"].strip(),
                    year=parse_year(record["title"]),
                    genres=parse_genres(record["genres"]),
                )
            )
    return rows


def read_tags(path: Path) -> dict[int, list[str]]:
    """Collect the distinct tags per movie, sorted for stable storage."""

    tags: dict[int, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            tag = record["tag"].strip()
            if tag:
                tags[int(record["movieId"])].add(tag)
    return {movie_id: sorted(values) for movie_id, values in tags.items()}


def read_links(path: Path) -> dict[int, int]:
    """Map MovieLens movie ids to TMDB ids, skipping unmapped rows.

    ``tmdbId`` is parsed through ``float`` because pandas round-trips the column
    as ``862.0`` once the empty cells make it a float column.
    """

    links: dict[int, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            raw = (record.get("tmdbId") or "").strip()
            if raw:
                links[int(record["movieId"])] = int(float(raw))
    return links


def _optional(path: Path) -> Path | None:
    return path if path.is_file() else None


def import_catalogue(
    session: Session,
    dataset_dir: Path,
    *,
    batch_size: int = BATCH_SIZE,
) -> CatalogueStats:
    """Load the catalogue from ``dataset_dir`` into the database.

    Args:
        session: Open SQLAlchemy session; the caller owns the transaction.
        dataset_dir: Directory containing at least ``movies.csv``.
        batch_size: Rows per bulk insert.

    Returns:
        Counts describing what changed.
    """

    movies_path = dataset_dir / "movies.csv"
    if not movies_path.is_file():
        raise FileNotFoundError(f"No movies.csv in {dataset_dir}")

    rows = read_movies(movies_path)
    tags_path = _optional(dataset_dir / "tags.csv")
    links_path = _optional(dataset_dir / "links.csv")
    tags = read_tags(tags_path) if tags_path else {}
    links = read_links(links_path) if links_path else {}

    existing: dict[int, Movie] = {
        movie.ml_movie_id: movie
        for movie in session.scalars(select(Movie).where(Movie.ml_movie_id.is_not(None)))
        if movie.ml_movie_id is not None
    }

    pending: list[dict[str, object]] = []
    created = 0
    updated = 0

    for row in rows:
        payload: dict[str, object] = {
            "title": row.title,
            "year": row.year,
            "genres": row.genres,
            "tags": tags.get(row.ml_movie_id, []),
            "tmdb_id": links.get(row.ml_movie_id),
        }

        movie = existing.get(row.ml_movie_id)
        if movie is None:
            pending.append({"ml_movie_id": row.ml_movie_id, **payload})
            created += 1
        else:
            for field, value in payload.items():
                setattr(movie, field, value)
            updated += 1

    for batch in _chunks(pending, batch_size):
        session.execute(insert(Movie), batch)
    session.commit()

    return CatalogueStats(
        movies=len(rows),
        created=created,
        updated=updated,
        tagged_movies=sum(1 for row in rows if tags.get(row.ml_movie_id)),
        tmdb_linked=sum(1 for row in rows if row.ml_movie_id in links),
    )


__all__ = [
    "CatalogueStats",
    "MovieRow",
    "import_catalogue",
    "parse_genres",
    "parse_year",
    "read_links",
    "read_movies",
    "read_tags",
]
