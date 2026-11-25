"""Validate a downloaded MovieLens dataset and print summary statistics.

This is the first pass of Phase 1 EDA: it checks schema, duplicates, rating
ranges, referential integrity and sparsity before any modelling starts.

Usage:
    python scripts/validate_movielens.py data/raw/ml-latest-small
    python scripts/validate_movielens.py data/raw/ml-1m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.validation import render_report, validate  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Directory such as data/raw/ml-latest-small")
    return parser.parse_args()


def main() -> int:
    """Entry point."""

    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.dataset_dir}")

    print(render_report(validate(args.dataset_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
