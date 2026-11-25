"""Dataset discovery and loading for the MovieLens releases used by CineMatch.

The loaders understand both layouts:

- CSV releases (latest-small, 25M): ``ratings.csv``, ``movies.csv``, ``tags.csv``
- the legacy 1M release: ``ratings.dat``, ``movies.dat``, ``users.dat`` (``::``)

Dtypes are kept compact because the 25M release is loaded on a laptop.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]

_RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}


def _find(dataset_dir: Path, *names: str) -> Path | None:
    for name in names:
        path = dataset_dir / name
        if path.is_file():
            return path
    return None


def load_ratings(dataset_dir: Path) -> pd.DataFrame:
    """Load ratings from either release layout."""

    csv_path = _find(dataset_dir, "ratings.csv")
    if csv_path is not None:
        return pd.read_csv(csv_path, dtype=_RATING_DTYPES)

    dat_path = _find(dataset_dir, "ratings.dat")
    if dat_path is not None:
        ratings = pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=RATING_COLUMNS,
            encoding="latin-1",
        )
        return ratings.astype(_RATING_DTYPES)

    raise FileNotFoundError(f"No ratings.csv or ratings.dat in {dataset_dir}")


def load_movies(dataset_dir: Path) -> pd.DataFrame:
    """Load the movie catalogue."""

    csv_path = _find(dataset_dir, "movies.csv")
    if csv_path is not None:
        return pd.read_csv(
            csv_path,
            dtype={"movieId": "int32", "title": "string", "genres": "string"},
        )

    dat_path = _find(dataset_dir, "movies.dat")
    if dat_path is not None:
        return pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
            dtype={"movieId": "int32", "title": "string", "genres": "string"},
        )

    raise FileNotFoundError(f"No movies.csv or movies.dat in {dataset_dir}")


def load_tags(dataset_dir: Path) -> pd.DataFrame | None:
    """Load user tags, or return ``None`` when the release has none."""

    csv_path = _find(dataset_dir, "tags.csv")
    if csv_path is not None:
        return pd.read_csv(
            csv_path,
            dtype={
                "userId": "int32",
                "movieId": "int32",
                "tag": "string",
                "timestamp": "int64",
            },
        )

    dat_path = _find(dataset_dir, "tags.dat")
    if dat_path is not None:
        return pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "tag", "timestamp"],
            encoding="latin-1",
        )

    return None


def count_genome_tags(dataset_dir: Path) -> int:
    """Return the number of tag-genome tags, or 0 when the release has none."""

    path = dataset_dir / "genome-tags.csv"
    if not path.is_file():
        return 0

    return len(pd.read_csv(path, usecols=["tagId"], dtype={"tagId": "int32"}))
