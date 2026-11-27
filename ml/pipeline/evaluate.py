"""Evaluate the implemented models on the held-out split.

Development runs use the validation split; the test split is reserved for the
final, once-only evaluation after all tuning is complete.

Usage:
    python -m ml.pipeline.evaluate --config ml/config.yaml
    python -m ml.pipeline.evaluate --split test      # final run only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml.data import load_movies, load_tags
from ml.evaluation.ranking import evaluate_ranking
from ml.evaluation.rating import evaluate_rating
from ml.features import MovieFeatures
from ml.models import (
    BiasBaseline,
    ContentModel,
    ItemKNN,
    PopularityBaseline,
    RatingModel,
    SVDModel,
)
from ml.pipeline.run import load_config

RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}


def load_split(dataset_path: Path, name: str) -> pd.DataFrame:
    """Load one partition written by ``ml.pipeline.split``."""

    path = dataset_path / "splits" / f"{name}.csv"
    if not path.is_file():
        raise SystemExit(f"Missing {path}; run `python -m ml.pipeline.split` first.")
    return pd.read_csv(path, dtype=RATING_DTYPES)


def dataset_digests(dataset_path: Path) -> dict[str, str]:
    """Return SHA-256 digests of the source files for the manifest."""

    digests: dict[str, str] = {}
    for name in ("ratings.csv", "movies.csv", "tags.csv"):
        path = dataset_path / name
        if not path.is_file():
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        digests[name] = hasher.hexdigest()
    return digests


def build_models(config: dict[str, object], dataset_path: Path) -> list[RatingModel]:
    """Instantiate every model implemented so far."""

    baselines = config.get("baselines", {})
    models = config.get("models", {})
    assert isinstance(baselines, dict) and isinstance(models, dict)

    bias = baselines.get("bias", {})
    popularity = baselines.get("popularity", {})
    content = models.get("content", {})
    knn = models.get("item_knn", {})
    svd = models.get("svd", {})
    assert isinstance(bias, dict) and isinstance(popularity, dict)
    assert isinstance(content, dict) and isinstance(knn, dict) and isinstance(svd, dict)

    features = MovieFeatures.build(
        load_movies(dataset_path),
        load_tags(dataset_path),
        min_df=int(content.get("min_df", 2)),
    )
    print(
        f"Catalogue features: {features.matrix.shape[0]:,} movies, "
        f"{features.vocabulary_size:,} terms."
    )

    return [
        BiasBaseline(
            regularisation=float(bias.get("regularisation", 10.0)),
            iterations=int(bias.get("iterations", 10)),
        ),
        PopularityBaseline(min_votes=float(popularity.get("min_votes", 10.0))),
        ContentModel(features, neighbour_k=int(content.get("neighbour_k", 20))),
        ItemKNN(
            neighbours=int(knn.get("neighbours", 50)),
            significance=float(knn.get("significance", 10.0)),
            shrinkage=float(knn.get("shrinkage", 5.0)),
        ),
        SVDModel(
            factors=int(svd.get("factors", 32)),
            bias_regularisation=float(bias.get("regularisation", 10.0)),
            bias_iterations=int(bias.get("iterations", 10)),
            seed=int(config.get("seed", 42)),
        ),
    ]


def print_table(results: dict[str, dict[str, object]], k: int) -> None:
    """Print a compact Markdown table."""

    print(f"| Model | RMSE | MAE | P@{k} | R@{k} | NDCG@{k} | Coverage@{k} | Novelty@{k} |")
    print("|---|---|---|---|---|---|---|---|")
    for name, payload in results.items():
        rating = payload["rating"]
        ranking = payload["ranking"]
        assert isinstance(rating, dict) and isinstance(ranking, dict)
        metrics = ranking["metrics"]
        coverage = ranking["coverage"]
        novelty = ranking["novelty"]
        assert isinstance(metrics, dict)
        assert isinstance(coverage, dict)
        assert isinstance(novelty, dict)
        print(
            f"| {name} | {rating['rmse']:.4f} | {rating['mae']:.4f} | "
            f"{metrics[f'precision@{k}']:.4f} | {metrics[f'recall@{k}']:.4f} | "
            f"{metrics[f'ndcg@{k}']:.4f} | {coverage[f'coverage@{k}']:.4f} | "
            f"{novelty[f'novelty@{k}']:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ml/config.yaml"))
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
        help="Evaluation partition. Use `test` only for the final run.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Deterministically sample users for a quick ranking run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Results JSON (default: docs/results/<dataset>-<split>.json).",
    )
    return parser.parse_args()


def main() -> int:
    """Fit each model, evaluate and write the results JSON."""

    args = parse_args()
    config = load_config(args.config)

    dataset = config.get("dataset", {})
    assert isinstance(dataset, dict)
    name = str(dataset.get("name", "dataset"))
    dataset_path = Path(dataset.get("path", "data/raw/ml-latest-small"))

    evaluation_config = config.get("evaluation", {})
    assert isinstance(evaluation_config, dict)
    k_values = tuple(int(value) for value in evaluation_config.get("k_values", [5, 10, 20]))
    threshold = float(evaluation_config.get("relevance_threshold", 4.0))
    configured_users = evaluation_config.get("max_users")
    max_users = args.max_users if args.max_users is not None else configured_users
    max_users = int(max_users) if max_users else None

    train = load_split(dataset_path, "train")
    evaluation = load_split(dataset_path, args.split)

    if args.split == "test":
        validation = load_split(dataset_path, "validation")
        fit_frame = pd.concat([train, validation], ignore_index=True)
    else:
        fit_frame = train

    catalogue = load_movies(dataset_path)["movieId"].unique()
    print(
        f"Dataset {name}: fit on {len(fit_frame):,} ratings, "
        f"evaluate {args.split} ({len(evaluation):,}), catalogue {len(catalogue):,}."
    )

    results: dict[str, dict[str, object]] = {}
    for model in build_models(config, dataset_path):
        print(f"Fitting {model.name} ...")
        started = time.perf_counter()
        model.fit(fit_frame)
        fit_seconds = time.perf_counter() - started

        rating = evaluate_rating(model, evaluation)
        ranking = evaluate_ranking(
            model,
            test=evaluation,
            history=fit_frame,
            catalogue=catalogue,
            k_values=k_values,
            relevance_threshold=threshold,
            max_users=max_users,
            seed=int(config.get("seed", 42)),
        )
        results[model.name] = {
            "rating": rating,
            "ranking": ranking,
            "parameters": model.describe(),
            "fit_seconds": round(fit_seconds, 1),
        }
        print(
            f"  {model.name}: RMSE={rating['rmse']:.4f} "
            f"NDCG@{max(k_values)}={ranking['metrics'][f'ndcg@{max(k_values)}']:.4f} "
            f"({fit_seconds:.0f}s fit)"
        )

    report_k = 10 if 10 in k_values else max(k_values)
    print()
    print_table(results, report_k)

    out = args.out or Path("docs/results") / f"{name}-{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": name,
        "split": args.split,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "fitted_on": "train+validation" if args.split == "test" else "train",
        "dataset_digests": dataset_digests(dataset_path),
        "config": {
            "k_values": list(k_values),
            "relevance_threshold": threshold,
            "max_users": max_users,
            "seed": config.get("seed"),
        },
        "models": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
