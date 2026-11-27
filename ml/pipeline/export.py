"""Fit the baseline models and export a versioned artifact bundle.

Usage:
    python -m ml.pipeline.export --config ml/config.yaml --version baselines

The bundle lands in ``ml/artifacts/<version>/`` (gitignored) and contains
``models.joblib`` plus ``manifest.json`` with hyper-parameters, validation
metrics, seeds and dataset hashes.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from ml.data import load_movies
from ml.evaluation.ranking import evaluate_ranking
from ml.evaluation.rating import evaluate_rating
from ml.export import save_artifact_bundle
from ml.pipeline.evaluate import (
    build_models,
    dataset_digests,
    load_split,
)
from ml.pipeline.run import load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ml/config.yaml"))
    parser.add_argument("--version", default="baselines", help="Bundle directory name.")
    parser.add_argument("--out", type=Path, default=Path("ml/artifacts"))
    return parser.parse_args()


def main() -> int:
    """Fit models on train, evaluate on validation and write the bundle."""

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

    train = load_split(dataset_path, "train")
    validation = load_split(dataset_path, "validation")

    catalogue = load_movies(dataset_path)["movieId"].unique()

    model_entries: dict[str, dict[str, object]] = {}
    fitted = {}
    for model in build_models(config, dataset_path):
        print(f"Fitting {model.name} ...")
        model.fit(train)
        fitted[model.name] = model
        model_entries[model.name] = {
            "parameters": model.describe(),
            "rating": evaluate_rating(model, validation),
            "ranking": evaluate_ranking(
                model,
                test=validation,
                history=train,
                catalogue=catalogue,
                k_values=k_values,
                relevance_threshold=threshold,
                seed=int(config.get("seed", 42)),
            ),
        }
        print(f"  {model.name}: validation RMSE={model_entries[model.name]['rating']['rmse']:.4f}")

    manifest: dict[str, object] = {
        "version": args.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": name,
        "dataset_digests": dataset_digests(dataset_path),
        "fitted_on": "train",
        "split": config.get("split", {}),
        "seed": config.get("seed"),
        "k_values": list(k_values),
        "relevance_threshold": threshold,
        "models": model_entries,
    }

    bundle = save_artifact_bundle(fitted, args.out / args.version, manifest)
    size_mb = (bundle / "models.joblib").stat().st_size / (1024 * 1024)
    print(f"\nWrote {bundle} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
