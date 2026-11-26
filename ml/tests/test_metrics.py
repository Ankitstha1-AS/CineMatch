"""Tests for the rating and ranking metrics."""

from __future__ import annotations

import math

import pytest
from ml.evaluation.metrics import (
    catalogue_coverage,
    mae,
    ndcg_at_k,
    novelty_at_k,
    precision_at_k,
    recall_at_k,
    rmse,
)


def test_rmse_and_mae_hand_computed() -> None:
    actual = [3.0, 4.0, 5.0]
    predicted = [2.0, 4.0, 4.0]

    assert rmse(actual, predicted) == pytest.approx(math.sqrt(2 / 3))
    assert mae(actual, predicted) == pytest.approx(2 / 3)


def test_rating_metrics_reject_empty_or_mismatched_input() -> None:
    with pytest.raises(ValueError):
        rmse([], [])
    with pytest.raises(ValueError):
        mae([1.0], [1.0, 2.0])


def test_precision_and_recall_at_k() -> None:
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}

    assert precision_at_k(recommended, relevant, 5) == pytest.approx(2 / 5)
    assert recall_at_k(recommended, relevant, 5) == pytest.approx(2 / 3)
    assert precision_at_k(recommended, relevant, 2) == pytest.approx(1 / 2)
    assert recall_at_k(recommended, relevant, 2) == pytest.approx(1 / 3)


def test_metrics_return_zero_without_relevant_items() -> None:
    assert precision_at_k([1, 2], set(), 2) == 0.0
    assert recall_at_k([1, 2], set(), 2) == 0.0
    assert ndcg_at_k([1, 2], set(), 2) == 0.0


def test_ndcg_hand_computed() -> None:
    recommended = [1, 2, 3, 4, 5]
    relevant = {2, 4, 6}

    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)

    assert ndcg_at_k(recommended, relevant, 5) == pytest.approx(dcg / ideal)
    assert ndcg_at_k(recommended, relevant, 2) == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_ndcg_rewards_higher_positions() -> None:
    relevant = {1, 2}

    assert ndcg_at_k([1, 3], relevant, 2) > ndcg_at_k([3, 1], relevant, 2)


def test_ndcg_ideal_when_all_relevant_items_are_on_top() -> None:
    assert ndcg_at_k([1, 2, 3], {1, 2}, 3) == pytest.approx(1.0)


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        precision_at_k([1], {1}, 0)


def test_catalogue_coverage_counts_distinct_items() -> None:
    recommendations = {1: [10, 11], 2: [11, 12]}

    assert catalogue_coverage(recommendations, catalogue_size=10, k=1) == pytest.approx(0.2)
    assert catalogue_coverage(recommendations, catalogue_size=10) == pytest.approx(0.3)


def test_novelty_prefers_long_tail_items() -> None:
    popularity = {1: 100, 2: 1}

    head_score = novelty_at_k({1: [1]}, popularity, total_ratings=100, k=1)
    tail_score = novelty_at_k({1: [2]}, popularity, total_ratings=100, k=1)

    assert head_score == pytest.approx(0.0)
    assert tail_score > head_score


def test_novelty_ignores_items_without_training_ratings() -> None:
    assert novelty_at_k({1: [99]}, {1: 100}, total_ratings=100, k=1) == 0.0
