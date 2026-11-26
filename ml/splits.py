"""Per-user temporal hold-out split.

Protocol (AGENTS.md section 4):

- sort each user's ratings by timestamp (movie id breaks ties) so the newest
  ratings always land in the test set;
- the latest ``test_fraction`` of each user's ratings becomes the test set;
- the latest ``validation_fraction`` of the remaining history becomes the
  validation set, used for tuning only;
- users with fewer than ``min_ratings`` ratings are dropped from the experiment.

The split is deterministic: timestamps order the rows and the random seed is
recorded only for provenance. The test set must be evaluated once, after all
tuning is finished.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]


@dataclass(frozen=True, slots=True)
class SplitFrames:
    """The three partitions plus bookkeeping about dropped users."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    dropped_users: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        """Return a JSON-serialisable description of the split."""

        def describe(frame: pd.DataFrame) -> dict[str, object]:
            if frame.empty:
                return {"ratings": 0, "users": 0, "movies": 0}
            return {
                "ratings": int(len(frame)),
                "users": int(frame["userId"].nunique()),
                "movies": int(frame["movieId"].nunique()),
                "rating_mean": float(frame["rating"].mean()),
                "timestamp_min": _iso(frame["timestamp"].min()),
                "timestamp_max": _iso(frame["timestamp"].max()),
            }

        return {
            "counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "test": len(self.test),
            },
            "splits": {
                "train": describe(self.train),
                "validation": describe(self.validation),
                "test": describe(self.test),
            },
            "dropped_users": len(self.dropped_users),
        }


def _iso(timestamp: int | float) -> str:
    return pd.to_datetime(int(timestamp), unit="s", utc=True).isoformat()


def temporal_split(
    ratings: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    validation_fraction: float = 0.1,
    min_ratings: int = 5,
    seed: int = 42,
) -> SplitFrames:
    """Split ratings per user, newest first.

    Args:
        ratings: Frame with ``userId``, ``movieId``, ``rating``, ``timestamp``.
        test_fraction: Share of each user's ratings held out for the final test.
        validation_fraction: Share of each user's remaining history used for tuning.
        min_ratings: Users below this count are excluded from every partition.
        seed: Recorded for provenance; the split itself is deterministic.

    Returns:
        A :class:`SplitFrames` instance.
    """

    missing = [column for column in RATING_COLUMNS if column not in ratings.columns]
    if missing:
        raise ValueError(f"ratings is missing columns: {missing}")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if min_ratings < 3:
        raise ValueError("min_ratings must be at least 3")

    ordered = ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable").reset_index(
        drop=True
    )

    sizes = ordered.groupby("userId", sort=False).size()
    kept_users = sizes[sizes >= min_ratings].index
    dropped_users = tuple(int(user) for user in sizes[sizes < min_ratings].index)

    ordered = ordered[ordered["userId"].isin(kept_users)].reset_index(drop=True)
    if ordered.empty:
        raise ValueError("no user has enough ratings for a split")

    sizes = ordered.groupby("userId", sort=False).size().to_numpy()
    total = int(sizes.sum())

    test_counts = np.maximum(1, np.floor(sizes * test_fraction).astype(int))
    remaining = sizes - test_counts
    validation_counts = np.maximum(1, np.rint(remaining * validation_fraction).astype(int))
    # Always leave at least one rating in train.
    validation_counts = np.minimum(validation_counts, np.maximum(remaining - 1, 0))

    positions = np.arange(total)
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    offset = positions - np.repeat(starts, sizes)

    train_end = np.repeat(sizes - test_counts - validation_counts, sizes)
    validation_end = np.repeat(sizes - test_counts, sizes)

    train_mask = offset < train_end
    validation_mask = (offset >= train_end) & (offset < validation_end)
    test_mask = offset >= validation_end

    train = ordered[train_mask].reset_index(drop=True)
    validation = ordered[validation_mask].reset_index(drop=True)
    test = ordered[test_mask].reset_index(drop=True)

    return SplitFrames(
        train=train,
        validation=validation,
        test=test,
        dropped_users=dropped_users,
    )


def save_split(
    splits: SplitFrames,
    out_dir: Path,
    *,
    dataset: str,
    config: dict[str, object] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, object]:
    """Write the partitions as CSV and return the manifest.

    The CSV files stay in the gitignored data directory. Pass
    ``manifest_path`` to keep a small, committed record of the split.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    splits.train.to_csv(out_dir / "train.csv", index=False)
    splits.validation.to_csv(out_dir / "validation.csv", index=False)
    splits.test.to_csv(out_dir / "test.csv", index=False)

    manifest: dict[str, object] = {
        "dataset": dataset,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol": "per-user temporal hold-out",
        "config": config or {},
        **splits.summary(),
    }

    text = json.dumps(manifest, indent=2)
    (out_dir / "split.json").write_text(text, encoding="utf-8")
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(text, encoding="utf-8")

    return manifest
