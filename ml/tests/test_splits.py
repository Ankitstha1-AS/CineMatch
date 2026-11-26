"""Tests for the per-user temporal split."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from ml.splits import temporal_split


def make_ratings() -> pd.DataFrame:
    rows = []
    for user, count in ((1, 10), (2, 5), (3, 2)):
        for index in range(1, count + 1):
            rows.append(
                {
                    "userId": user,
                    "movieId": user * 100 + index,
                    "rating": 4.0,
                    "timestamp": index,
                }
            )
    return pd.DataFrame(rows)


def test_split_respects_min_ratings_and_counts() -> None:
    splits = temporal_split(
        make_ratings(), test_fraction=0.2, validation_fraction=0.1, min_ratings=3
    )

    assert splits.dropped_users == (3,)
    assert len(splits.train) == 10  # 7 for user 1 + 3 for user 2
    assert len(splits.validation) == 2
    assert len(splits.test) == 3


def test_split_keeps_the_newest_ratings_for_test() -> None:
    splits = temporal_split(
        make_ratings(), test_fraction=0.2, validation_fraction=0.1, min_ratings=3
    )

    user_one = splits.test[splits.test["userId"] == 1]
    assert list(user_one["timestamp"]) == [9, 10]

    user_two = splits.test[splits.test["userId"] == 2]
    assert list(user_two["timestamp"]) == [5]

    assert list(splits.validation[splits.validation["userId"] == 1]["timestamp"]) == [8]
    assert list(splits.validation[splits.validation["userId"] == 2]["timestamp"]) == [4]


def test_partitions_are_chronological_and_disjoint() -> None:
    splits = temporal_split(make_ratings(), min_ratings=3)

    for user in (1, 2):
        train = splits.train[splits.train["userId"] == user]
        validation = splits.validation[splits.validation["userId"] == user]
        test = splits.test[splits.test["userId"] == user]

        assert train["timestamp"].max() < validation["timestamp"].min()
        assert validation["timestamp"].max() < test["timestamp"].min()

        seen = set(train["movieId"]) | set(validation["movieId"]) | set(test["movieId"])
        assert len(seen) == len(train) + len(validation) + len(test)


def test_split_is_deterministic() -> None:
    first = temporal_split(make_ratings(), min_ratings=3)
    second = temporal_split(make_ratings().sample(frac=1.0, random_state=7), min_ratings=3)

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.validation, second.validation)
    pd.testing.assert_frame_equal(first.test, second.test)


def test_split_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        temporal_split(make_ratings(), test_fraction=0)
    with pytest.raises(ValueError):
        temporal_split(make_ratings(), validation_fraction=1)
    with pytest.raises(ValueError):
        temporal_split(make_ratings(), min_ratings=1)


def test_summary_reports_counts_and_ranges() -> None:
    splits = temporal_split(make_ratings(), min_ratings=3)
    summary = splits.summary()

    assert summary["counts"] == {"train": 10, "validation": 2, "test": 3}
    assert summary["dropped_users"] == 1
    assert summary["splits"]["train"]["users"] == 2  # type: ignore[index]


def test_save_split_writes_partitions_and_manifest(tmp_path: Path) -> None:
    from ml.splits import save_split

    splits = temporal_split(make_ratings(), min_ratings=3)
    manifest_path = tmp_path / "docs" / "split.json"

    manifest = save_split(
        splits,
        tmp_path / "splits",
        dataset="toy",
        config={"test_fraction": 0.2},
        manifest_path=manifest_path,
    )

    assert (tmp_path / "splits" / "train.csv").is_file()
    assert (tmp_path / "splits" / "validation.csv").is_file()
    assert (tmp_path / "splits" / "test.csv").is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["dataset"] == "toy"
    assert manifest["protocol"] == "per-user temporal hold-out"
