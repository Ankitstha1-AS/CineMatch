"""Build the per-user temporal split for the configured dataset.

Usage:
    python -m ml.pipeline.split --config ml/config.yaml

Writes ``train.csv``, ``validation.csv`` and ``test.csv`` under
``<dataset>/splits/`` (gitignored) plus a small manifest in ``docs/splits/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.data import load_ratings
from ml.pipeline.run import load_config
from ml.splits import save_split, temporal_split


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("ml/config.yaml"))
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Committed manifest path (default: docs/splits/<dataset>.json).",
    )
    return parser.parse_args()


def main() -> int:
    """Build and save the split."""

    args = parse_args()
    config = load_config(args.config)

    dataset = config.get("dataset", {})
    name = str(dataset.get("name", "dataset"))
    dataset_path = Path(dataset.get("path", "data/raw/ml-latest-small"))
    split_config = config.get("split", {})

    print(f"Loading ratings from {dataset_path} ...")
    ratings = load_ratings(dataset_path)

    splits = temporal_split(
        ratings,
        test_fraction=float(split_config.get("test_fraction", 0.2)),
        validation_fraction=float(split_config.get("validation_fraction", 0.1)),
        min_ratings=int(split_config.get("min_ratings", 5)),
        seed=int(config.get("seed", 42)),
    )

    manifest_path = args.manifest_out or Path("docs/splits") / f"{name}.json"
    manifest = save_split(
        splits,
        dataset_path / "splits",
        dataset=name,
        config=split_config,
        manifest_path=manifest_path,
    )

    counts = manifest["counts"]
    assert isinstance(counts, dict)
    print(
        "Split: "
        f"train={counts['train']:,} validation={counts['validation']:,} "
        f"test={counts['test']:,}; dropped users={manifest['dropped_users']}"
    )
    print(f"Wrote partitions to {dataset_path / 'splits'} and manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
