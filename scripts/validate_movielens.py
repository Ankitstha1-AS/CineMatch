"""Validate a downloaded MovieLens dataset and print summary statistics.

This is the first pass of Phase 1 EDA: it checks schema, duplicates, rating
ranges, referential integrity and sparsity before any modelling starts.

Usage:
    python scripts/validate_movielens.py data/raw/ml-latest-small
    python scripts/validate_movielens.py data/raw/ml-1m
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


def load_ratings(dataset_dir: Path) -> pd.DataFrame:
    """Load ratings from the CSV (latest/25M) or DAT (1M) layout."""

    csv_path = dataset_dir / "ratings.csv"
    dat_path = dataset_dir / "ratings.dat"

    if csv_path.is_file():
        return pd.read_csv(csv_path)

    if dat_path.is_file():
        return pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=RATING_COLUMNS,
            encoding="latin-1",
        )

    raise FileNotFoundError(f"No ratings.csv or ratings.dat in {dataset_dir}")


def load_movies(dataset_dir: Path) -> pd.DataFrame:
    """Load the movie catalogue."""

    csv_path = dataset_dir / "movies.csv"
    dat_path = dataset_dir / "movies.dat"

    if csv_path.is_file():
        return pd.read_csv(csv_path)

    if dat_path.is_file():
        return pd.read_csv(
            dat_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )

    raise FileNotFoundError(f"No movies.csv or movies.dat in {dataset_dir}")


def load_tags(dataset_dir: Path) -> pd.DataFrame | None:
    """Load tags when the dataset provides them."""

    for name in ("tags.csv", "tags.dat"):
        path = dataset_dir / name
        if path.is_file():
            if name.endswith(".csv"):
                return pd.read_csv(path)
            return pd.read_csv(
                path,
                sep="::",
                engine="python",
                names=["userId", "movieId", "tag", "timestamp"],
                encoding="latin-1",
            )
    return None


def validate(dataset_dir: Path) -> dict[str, object]:
    """Run validation checks and return the collected statistics."""

    ratings = load_ratings(dataset_dir)
    movies = load_movies(dataset_dir)
    tags = load_tags(dataset_dir)

    missing_columns = [
        column for column in RATING_COLUMNS if column not in ratings.columns
    ]
    if missing_columns:
        raise ValueError(f"ratings is missing columns: {missing_columns}")

    duplicate_pairs = int(ratings.duplicated(subset=["userId", "movieId"]).sum())
    duplicate_movie_ids = int(movies.duplicated(subset=["movieId"]).sum())
    unknown_movies = int((~ratings["movieId"].isin(movies["movieId"])).sum())

    n_users = int(ratings["userId"].nunique())
    n_movies_rated = int(ratings["movieId"].nunique())
    n_ratings = int(len(ratings))
    possible = n_users * n_movies_rated
    sparsity = 1 - (n_ratings / possible) if possible else 0.0

    rating_counts = ratings["rating"].value_counts().sort_index()
    rated_movie_counts = ratings.groupby("movieId").size()
    movie_popularity = movies["movieId"].map(rated_movie_counts).fillna(0).astype(int)

    stats: dict[str, object] = {
        "dataset": dataset_dir.name,
        "ratings": n_ratings,
        "users": n_users,
        "movies_total": int(len(movies)),
        "movies_rated": n_movies_rated,
        "movies_with_zero_ratings": int((movie_popularity == 0).sum()),
        "sparsity": sparsity,
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_std": float(ratings["rating"].std()),
        "duplicate_user_movie_pairs": duplicate_pairs,
        "duplicate_movie_ids": duplicate_movie_ids,
        "ratings_for_unknown_movies": unknown_movies,
        "ratings_per_user_median": float(ratings.groupby("userId").size().median()),
        "ratings_per_movie_median": float(rated_movie_counts.median()),
        "timestamp_min": pd.to_datetime(ratings["timestamp"].min(), unit="s", utc=True),
        "timestamp_max": pd.to_datetime(ratings["timestamp"].max(), unit="s", utc=True),
        "rating_distribution": rating_counts.to_dict(),
    }

    if tags is not None:
        stats["tags"] = int(len(tags))
        stats["distinct_tags"] = int(tags["tag"].nunique())
        stats["movies_with_tags"] = int(tags["movieId"].nunique())
    else:
        stats["tags"] = 0
        stats["distinct_tags"] = 0
        stats["movies_with_tags"] = 0

    return stats


def print_report(stats: dict[str, object]) -> None:
    """Print a Markdown-friendly report."""

    print(f"# MovieLens validation: {stats['dataset']}\n")
    print("| Metric | Value |")
    print("|---|---|")
    for key, value in stats.items():
        if key == "rating_distribution":
            continue
        if isinstance(value, float):
            print(f"| {key} | {value:.4f} |")
        else:
            print(f"| {key} | {value} |")

    distribution = stats["rating_distribution"]
    assert isinstance(distribution, dict)
    print("\n## Rating distribution\n")
    print("| Rating | Count |")
    print("|---|---|")
    for rating, count in distribution.items():
        print(f"| {rating} | {count} |")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_dir", type=Path, help="Directory such as data/raw/ml-latest-small"
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_dir}")

    print_report(validate(args.dataset_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
