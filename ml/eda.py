"""Exploratory data analysis for MovieLens datasets.

Produces a Markdown summary and PNG figures under ``docs/eda/<dataset>/``:

- rating distribution
- ratings per user (activity)
- movie popularity long tail
- Lorenz curve of rating concentration
- ratings per year
- most common genres

Usage:
    python -m ml.eda data/raw/ml-latest-small
    python -m ml.eda data/raw/ml-25m --out docs/eda
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ml.data import count_genome_tags, load_movies, load_ratings, load_tags  # noqa: E402

FIGSIZE = (8, 4.5)
DPI = 150
GENRE_LIMIT = 15


def gini(values: pd.Series) -> float:
    """Return the Gini coefficient of a non-negative distribution."""

    array = np.sort(values.to_numpy(dtype=float))
    n = array.size
    total = array.sum()
    if n == 0 or total == 0:
        return 0.0
    rank = np.arange(1, n + 1)
    return float((2 * np.sum(rank * array)) / (n * total) - (n + 1) / n)


def build_bundle(dataset_dir: Path) -> dict[str, object]:
    """Load a dataset and compute everything the report and figures need."""

    ratings = load_ratings(dataset_dir)
    movies = load_movies(dataset_dir)
    tags = load_tags(dataset_dir)

    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size().sort_values(ascending=False)
    total_ratings = float(per_movie.sum())

    top_shares = {}
    for fraction in (0.01, 0.05, 0.10, 0.50):
        count = max(1, int(round(len(per_movie) * fraction)))
        top_shares[fraction] = float(per_movie.iloc[:count].sum() / total_ratings)

    rating_years = pd.to_datetime(ratings["timestamp"], unit="s", utc=True).dt.year
    movie_years = movies["title"].str.extract(r"\((\d{4})\)\s*$", expand=False).astype("float64")

    genres = movies["genres"].dropna().str.split("|").explode()
    genre_counts = genres[genres != "(no genres listed)"].value_counts()

    n_users = int(ratings["userId"].nunique())
    n_movies_rated = int(ratings["movieId"].nunique())

    summary: dict[str, object] = {
        "dataset": dataset_dir.name,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "ratings": int(len(ratings)),
        "users": n_users,
        "movies_total": int(len(movies)),
        "movies_rated": n_movies_rated,
        "sparsity": 1 - (len(ratings) / (n_users * n_movies_rated)),
        "rating_min": float(ratings["rating"].min()),
        "rating_max": float(ratings["rating"].max()),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_std": float(ratings["rating"].std()),
        "ratings_per_user_median": float(per_user.median()),
        "ratings_per_user_p90": float(per_user.quantile(0.90)),
        "ratings_per_user_p99": float(per_user.quantile(0.99)),
        "ratings_per_user_max": int(per_user.max()),
        "ratings_per_movie_median": float(per_movie.median()),
        "ratings_per_movie_p90": float(per_movie.quantile(0.90)),
        "ratings_per_movie_p99": float(per_movie.quantile(0.99)),
        "ratings_per_movie_max": int(per_movie.max()),
        "users_with_at_most_20_ratings": int((per_user <= 20).sum()),
        "users_with_at_most_20_ratings_pct": float((per_user <= 20).mean() * 100),
        "top_movie_share_pct": top_shares[0.01] * 100,
        "top_5pct_movie_share_pct": top_shares[0.05] * 100,
        "top_10pct_movie_share_pct": top_shares[0.10] * 100,
        "top_50pct_movie_share_pct": top_shares[0.50] * 100,
        "movie_popularity_gini": gini(per_movie),
        "movie_year_min": int(movie_years.min()),
        "movie_year_max": int(movie_years.max()),
        "ratings_year_min": int(rating_years.min()),
        "ratings_year_max": int(rating_years.max()),
        "tags": 0 if tags is None else int(len(tags)),
        "distinct_tags": 0 if tags is None else int(tags["tag"].nunique()),
        "movies_with_tags": 0 if tags is None else int(tags["movieId"].nunique()),
        "genome_tags": count_genome_tags(dataset_dir),
        "genres": int(genre_counts.size),
    }

    if tags is not None and len(tags) > 0:
        summary["tag_coverage_pct"] = float(tags["movieId"].nunique() / len(movies) * 100)
        summary["median_tags_per_tagged_movie"] = float(tags.groupby("movieId").size().median())
    else:
        summary["tag_coverage_pct"] = 0.0
        summary["median_tags_per_tagged_movie"] = 0.0

    return {
        "summary": summary,
        "rating_counts": ratings["rating"].value_counts().sort_index(),
        "per_user": per_user,
        "per_movie": per_movie,
        "rating_years": rating_years.value_counts().sort_index(),
        "genre_counts": genre_counts,
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save and close a figure."""

    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def plot_figures(bundle: dict[str, object], out_dir: Path) -> list[str]:
    """Render the EDA figures and return their file names."""

    plt.rcParams.update({"figure.figsize": FIGSIZE, "axes.grid": True, "grid.alpha": 0.3})
    names: list[str] = []

    rating_counts = bundle["rating_counts"]
    assert isinstance(rating_counts, pd.Series)
    fig, ax = plt.subplots()
    rating_counts.plot(kind="bar", ax=ax, color="#ffcf5c")
    ax.set_title("Rating distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of ratings")
    save_figure(fig, out_dir / "rating_distribution.png")
    names.append("rating_distribution.png")

    per_user = bundle["per_user"]
    assert isinstance(per_user, pd.Series)
    fig, ax = plt.subplots()
    ax.hist(per_user, bins=60, color="#5aa9ff")
    ax.set_yscale("log")
    ax.set_title("Ratings per user")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Users (log scale)")
    save_figure(fig, out_dir / "ratings_per_user.png")
    names.append("ratings_per_user.png")

    per_movie = bundle["per_movie"]
    assert isinstance(per_movie, pd.Series)
    ranks = np.arange(1, len(per_movie) + 1)
    fig, ax = plt.subplots()
    ax.plot(ranks, per_movie.to_numpy(), color="#55d187")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Movie popularity long tail")
    ax.set_xlabel("Movie rank (log scale)")
    ax.set_ylabel("Ratings (log scale)")
    save_figure(fig, out_dir / "long_tail.png")
    names.append("long_tail.png")

    cumulative = np.cumsum(per_movie.to_numpy()) / per_movie.sum()
    movie_share = np.arange(1, len(per_movie) + 1) / len(per_movie)
    fig, ax = plt.subplots()
    ax.plot(movie_share * 100, cumulative * 100, color="#ff8a65")
    ax.plot([0, 100], [0, 100], linestyle="--", color="grey", linewidth=1)
    ax.set_title("Concentration of ratings (Lorenz curve)")
    ax.set_xlabel("Share of movies, most popular first (%)")
    ax.set_ylabel("Share of ratings (%)")
    save_figure(fig, out_dir / "lorenz_curve.png")
    names.append("lorenz_curve.png")

    rating_years = bundle["rating_years"]
    assert isinstance(rating_years, pd.Series)
    fig, ax = plt.subplots()
    rating_years.plot(ax=ax, color="#9b8cff")
    ax.set_title("Ratings per year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of ratings")
    save_figure(fig, out_dir / "ratings_per_year.png")
    names.append("ratings_per_year.png")

    genre_counts = bundle["genre_counts"]
    assert isinstance(genre_counts, pd.Series)
    top = genre_counts.head(GENRE_LIMIT).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    top.plot(kind="barh", ax=ax, color="#4dd0c4")
    ax.set_title(f"Top {GENRE_LIMIT} genres by catalogue size")
    ax.set_xlabel("Movies")
    save_figure(fig, out_dir / "top_genres.png")
    names.append("top_genres.png")

    return names


def render_summary(summary: dict[str, object], figures: list[str]) -> str:
    """Render the Markdown report."""

    p90_user = float(summary["ratings_per_user_p90"])
    p99_user = float(summary["ratings_per_user_p99"])
    few_users = int(summary["users_with_at_most_20_ratings"])
    few_pct = float(summary["users_with_at_most_20_ratings_pct"])
    p90_movie = float(summary["ratings_per_movie_p90"])
    p99_movie = float(summary["ratings_per_movie_p99"])
    tagged = int(summary["movies_with_tags"])
    tag_pct = float(summary["tag_coverage_pct"])

    lines = [
        f"# EDA: {summary['dataset']}",
        "",
        f"_Generated {summary['generated_at']} by `python -m ml.eda`._",
        "",
        "## Headline counts",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Ratings | {summary['ratings']:,} |",
        f"| Users | {summary['users']:,} |",
        f"| Movies in catalogue | {summary['movies_total']:,} |",
        f"| Movies with at least one rating | {summary['movies_rated']:,} |",
        f"| Sparsity | {summary['sparsity']:.4f} |",
        "",
        "## Rating distribution",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Min / max | {summary['rating_min']} / {summary['rating_max']} |",
        f"| Mean | {summary['rating_mean']:.4f} |",
        f"| Standard deviation | {summary['rating_std']:.4f} |",
        "",
        "## Activity",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Median ratings per user | {summary['ratings_per_user_median']:.1f} |",
        f"| p90 / p99 ratings per user | {p90_user:.1f} / {p99_user:.1f} |",
        f"| Max ratings by one user | {summary['ratings_per_user_max']:,} |",
        f"| Users with <= 20 ratings | {few_users:,} ({few_pct:.1f}%) |",
        f"| Median ratings per movie | {summary['ratings_per_movie_median']:.1f} |",
        f"| p90 / p99 ratings per movie | {p90_movie:.1f} / {p99_movie:.1f} |",
        f"| Max ratings for one movie | {summary['ratings_per_movie_max']:,} |",
        f"| Movie years | {summary['movie_year_min']} – {summary['movie_year_max']} |",
        f"| Rating years | {summary['ratings_year_min']} – {summary['ratings_year_max']} |",
        "",
        "## Long tail",
        "",
        "| Popular-movie group | Share of all ratings |",
        "|---|---|",
        f"| Top 1% of rated movies | {summary['top_movie_share_pct']:.1f}% |",
        f"| Top 5% of rated movies | {summary['top_5pct_movie_share_pct']:.1f}% |",
        f"| Top 10% of rated movies | {summary['top_10pct_movie_share_pct']:.1f}% |",
        f"| Top 50% of rated movies | {summary['top_50pct_movie_share_pct']:.1f}% |",
        "",
        f"Gini coefficient of movie popularity: **{summary['movie_popularity_gini']:.3f}** "
        "(0 = perfectly even, 1 = all ratings on one movie).",
        "",
        "## Tags and genres",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tags | {summary['tags']:,} |",
        f"| Distinct tags | {summary['distinct_tags']:,} |",
        f"| Movies with tags | {tagged:,} ({tag_pct:.1f}%) |",
        f"| Median tags per tagged movie | {summary['median_tags_per_tagged_movie']:.1f} |",
        f"| Tag-genome tags | {summary['genome_tags']:,} |",
        f"| Distinct genres | {summary['genres']:,} |",
        "",
        "## Figures",
        "",
    ]

    titles = {
        "rating_distribution.png": "Rating distribution",
        "ratings_per_user.png": "Ratings per user",
        "long_tail.png": "Movie popularity long tail",
        "lorenz_curve.png": "Rating concentration (Lorenz curve)",
        "ratings_per_year.png": "Ratings per year",
        "top_genres.png": "Most common genres",
    }
    for name in figures:
        lines.extend([f"### {titles.get(name, name)}", "", f"![{name}]({name})", ""])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="e.g. data/raw/ml-25m")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/eda"),
        help="Output root (default: docs/eda).",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_dir}")

    out_dir = args.out / args.dataset_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset_dir} ...")
    bundle = build_bundle(args.dataset_dir)

    print("Rendering figures ...")
    figures = plot_figures(bundle, out_dir)

    summary = bundle["summary"]
    assert isinstance(summary, dict)
    render = render_summary(summary, figures)
    (out_dir / "summary.md").write_text(render, encoding="utf-8")

    print(render)
    print(f"\nWrote {len(figures)} figures and summary.md to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
