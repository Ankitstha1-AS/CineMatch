"""Tests for the MovieLens catalogue importer."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Movie
from app.services.catalogue_import import (
    import_catalogue,
    parse_genres,
    parse_year,
    read_links,
    read_tags,
)

MOVIES_CSV = """movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
3,Grumpier Old Men (1995),(no genres listed)
"""

TAGS_CSV = """userId,movieId,tag,timestamp
1,1,pixar,1131497929
2,1,fun,1131498000
1,1,pixar,1131498100
1,2,board game,1131498200
"""

LINKS_CSV = """movieId,imdbId,tmdbId
1,0114709,862
2,0113497,
"""


def write_dataset(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "movies.csv").write_text(MOVIES_CSV, encoding="utf-8")
    (directory / "tags.csv").write_text(TAGS_CSV, encoding="utf-8")
    (directory / "links.csv").write_text(LINKS_CSV, encoding="utf-8")
    return directory


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Toy Story (1995)", 1995),
        ("A Movie (2019) ", 2019),
        ("No Year Here", None),
        ("Se7en (1995)", 1995),
    ],
)
def test_parse_year(title: str, expected: int | None) -> None:
    assert parse_year(title) == expected


def test_parse_genres_splits_and_drops_sentinel() -> None:
    assert parse_genres("Comedy|Drama") == ["Comedy", "Drama"]
    assert parse_genres("(no genres listed)") == []
    assert parse_genres("") == []


def test_read_tags_deduplicates_and_sorts(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "subset")

    tags = read_tags(dataset / "tags.csv")

    assert tags[1] == ["fun", "pixar"]
    assert tags[2] == ["board game"]


def test_read_links_skips_missing_tmdb_ids(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "subset")

    assert read_links(dataset / "links.csv") == {1: 862}


def test_read_links_accepts_float_formatted_ids(tmp_path: Path) -> None:
    path = tmp_path / "links.csv"
    path.write_text("movieId,imdbId,tmdbId\n1,0114709,862.0\n", encoding="utf-8")

    assert read_links(path) == {1: 862}


def test_import_creates_catalogue_rows(db_session: Session, tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "subset")

    stats = import_catalogue(db_session, dataset)

    assert stats.movies == 3
    assert stats.created == 3
    assert stats.updated == 0
    assert stats.tagged_movies == 2
    assert stats.tmdb_linked == 1

    movies = {movie.ml_movie_id: movie for movie in db_session.scalars(select(Movie)).all()}
    toy_story = movies[1]
    assert toy_story.title == "Toy Story (1995)"
    assert toy_story.year == 1995
    assert toy_story.genres == ["Adventure", "Animation", "Children", "Comedy", "Fantasy"]
    assert toy_story.tags == ["fun", "pixar"]
    assert toy_story.tmdb_id == 862

    assert movies[3].genres == []
    assert movies[3].tags == []
    assert movies[3].tmdb_id is None


def test_import_is_idempotent(db_session: Session, tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "subset")
    import_catalogue(db_session, dataset)

    stats = import_catalogue(db_session, dataset)

    assert stats.created == 0
    assert stats.updated == 3
    assert db_session.query(Movie).count() == 3


def test_import_updates_changed_metadata(db_session: Session, tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path / "subset")
    import_catalogue(db_session, dataset)

    (dataset / "movies.csv").write_text(
        "movieId,title,genres\n1,Toy Story (1995),Animation\n",
        encoding="utf-8",
    )
    (dataset / "tags.csv").unlink()
    (dataset / "links.csv").unlink()

    stats = import_catalogue(db_session, dataset)

    assert stats.updated == 1
    # Rows missing from the new file are kept: an import never deletes catalogue entries.
    assert db_session.query(Movie).count() == 3
    movie = db_session.scalars(select(Movie).where(Movie.ml_movie_id == 1)).one()
    assert movie.genres == ["Animation"]
    assert movie.tags == []
    assert movie.tmdb_id is None


def test_import_without_movies_csv_fails(db_session: Session, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_catalogue(db_session, tmp_path)
