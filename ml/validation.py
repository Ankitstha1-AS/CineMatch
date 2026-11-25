"""Validation checks and reporting for a MovieLens dataset directory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.data import RATING_COLUMNS, load_movies, load_ratings, load_tags


def validate(dataset_dir: Path) -> dict[str, object]:
    """Run schema, duplicate, integrity and sparsity checks."""

    ratings = load_ratings(dataset_dir)
    movies = load_movies(dataset_dir)
    tags = load_tags(dataset_dir)

    missing_columns = [column for column in RATING_COLUMNS if column not in ratings.columns]
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


def render_report(stats: dict[str, object]) -> str:
    """Render the statistics as Markdown."""

    lines = [f"# MovieLens validation: {stats['dataset']}", "", "| Metric | Value |", "|---|---|"]
    for key, value in stats.items():
        if key == "rating_distribution":
            continue
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")

    distribution = stats["rating_distribution"]
    assert isinstance(distribution, dict)
    lines.extend(["", "## Rating distribution", "", "| Rating | Count |", "|---|---|"])
    for rating, count in distribution.items():
        lines.append(f"| {rating} | {count} |")

    return "\n".join(lines)
