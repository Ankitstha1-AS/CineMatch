"""Build a reproducible MovieLens training subset.

The filters follow the Phase 1 EDA conclusions:

- keep movies with at least ``--min-movie-ratings`` ratings (content features need signal);
- keep users with at least ``--min-user-ratings`` ratings (enough history for a temporal split);
- sample at most ``--max-users`` users with a fixed seed so the subset fits a laptop;
- re-apply the movie filter after sampling so no retained movie is too sparse.

Outputs ``ratings.csv``, ``movies.csv``, ``tags.csv`` and ``manifest.json``.
Raw and processed data are gitignored and must not be redistributed.

Usage:
    python scripts/build_subset.py data/raw/ml-25m \
        --out data/processed/ml-25m-subset \
        --min-movie-ratings 10 --min-user-ratings 20 --max-users 10000 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.data import load_movies, load_ratings, load_tags  # noqa: E402


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filter_counts(ratings, min_users: int, min_movies: int):
    """Iteratively drop users and movies below the activity thresholds."""

    while True:
        before = len(ratings)

        user_counts = ratings.groupby("userId").size()
        ratings = ratings[ratings["userId"].isin(user_counts[user_counts >= min_users].index)]

        movie_counts = ratings.groupby("movieId").size()
        ratings = ratings[ratings["movieId"].isin(movie_counts[movie_counts >= min_movies].index)]

        if len(ratings) == before:
            return ratings


def build(dataset_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """Apply filters, sample users and write the subset files."""

    print(f"Loading {dataset_dir} ...")
    ratings = load_ratings(dataset_dir)
    movies = load_movies(dataset_dir)
    tags = load_tags(dataset_dir)

    source_counts = {
        "ratings": int(len(ratings)),
        "users": int(ratings["userId"].nunique()),
        "movies": int(ratings["movieId"].nunique()),
    }

    print("Applying activity filters ...")
    ratings = filter_counts(ratings, args.min_user_ratings, args.min_movie_ratings)

    selected_users = int(ratings["userId"].nunique())
    if args.max_users and selected_users > args.max_users:
        print(f"Sampling {args.max_users:,} of {selected_users:,} users (seed {args.seed}) ...")
        rng = np.random.default_rng(args.seed)
        users = np.sort(ratings["userId"].unique())
        chosen = rng.choice(users, size=args.max_users, replace=False)
        ratings = ratings[ratings["userId"].isin(chosen)]

    print("Re-applying filters after sampling ...")
    ratings = filter_counts(ratings, args.min_user_ratings, args.min_movie_ratings)

    ratings = ratings.sort_values(["userId", "timestamp"], kind="stable").reset_index(drop=True)
    keep_movies = set(ratings["movieId"].unique())
    subset_movies = movies[movies["movieId"].isin(keep_movies)].sort_values("movieId")

    keep_users = set(ratings["userId"].unique())
    if tags is not None:
        subset_tags = tags[
            tags["movieId"].isin(keep_movies) & tags["userId"].isin(keep_users)
        ].sort_values(["userId", "timestamp"])
    else:
        subset_tags = None

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ratings.to_csv(out_dir / "ratings.csv", index=False)
    subset_movies.to_csv(out_dir / "movies.csv", index=False)
    if subset_tags is not None:
        subset_tags.to_csv(out_dir / "tags.csv", index=False)

    links = dataset_dir / "links.csv"
    if links.is_file():
        link_frame = pd.read_csv(links, dtype={"imdbId": "string", "tmdbId": "Int64"})
        link_frame[link_frame["movieId"].isin(keep_movies)].to_csv(
            out_dir / "links.csv", index=False
        )

    genome_tags = dataset_dir / "genome-tags.csv"
    if genome_tags.is_file():
        shutil.copy2(genome_tags, out_dir / "genome-tags.csv")

    ratings_file = dataset_dir / "ratings.csv"
    if not ratings_file.is_file():
        ratings_file = dataset_dir / "ratings.dat"

    manifest: dict[str, object] = {
        "name": out_dir.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(dataset_dir),
        "source_ratings_file": str(ratings_file),
        "source_ratings_sha256": sha256_file(ratings_file),
        "filters": {
            "min_movie_ratings": args.min_movie_ratings,
            "min_user_ratings": args.min_user_ratings,
            "max_users": args.max_users,
            "seed": args.seed,
        },
        "source_counts": source_counts,
        "counts": {
            "ratings": int(len(ratings)),
            "users": int(ratings["userId"].nunique()),
            "movies": int(subset_movies["movieId"].nunique()),
            "tags": 0 if subset_tags is None else int(len(subset_tags)),
        },
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote subset to {out_dir}")
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="e.g. data/raw/ml-25m")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--min-movie-ratings", type=int, default=10)
    parser.add_argument("--min-user-ratings", type=int, default=20)
    parser.add_argument("--max-users", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional second manifest location (e.g. docs/subsets/ml-25m.json).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_dir}")

    build(args.dataset_dir, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
