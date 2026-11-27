"""Rating-error and top-N ranking metrics.

Ranking follows the protocol in AGENTS.md section 4: a rating of at least the
relevance threshold marks a test item as relevant, and models are ranked against
**all unrated items**, never a sampled negative set. Metrics are pure functions
so they can be unit-tested without a model.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence

import numpy as np


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the root mean squared error."""

    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if len(actual) == 0:
        raise ValueError("cannot compute RMSE on an empty set")
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(np.square(errors))))


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Return the mean absolute error."""

    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if len(actual) == 0:
        raise ValueError("cannot compute MAE on an empty set")
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(errors)))


def precision_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Return the share of the top ``k`` recommendations that are relevant."""

    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Return the share of relevant items that appear in the top ``k``."""

    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Collection[int], k: int) -> float:
    """Return binary-relevance NDCG at ``k``."""

    _validate_k(k)
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

    return dcg / idcg if idcg else 0.0


def catalogue_coverage(
    recommendations: Mapping[int, Sequence[int]],
    catalogue_size: int,
    k: int | None = None,
) -> float:
    """Return the share of the catalogue that appears in any recommendation list."""

    if catalogue_size <= 0:
        raise ValueError("catalogue_size must be positive")

    shown: set[int] = set()
    for items in recommendations.values():
        shown.update(items if k is None else items[:k])

    return len(shown) / catalogue_size


def novelty_at_k(
    recommendations: Mapping[int, Sequence[int]],
    popularity: Mapping[int, int],
    total_ratings: int,
    k: int,
) -> float:
    """Return mean self-information (in bits) of the recommended items.

    ``-log2(popularity / total_ratings)`` rewards long-tail items; items with no
    training ratings contribute nothing and do not dilute the mean.
    """

    _validate_k(k)
    if total_ratings <= 0:
        raise ValueError("total_ratings must be positive")

    scores: list[float] = []
    for items in recommendations.values():
        for item in items[:k]:
            count = popularity.get(item, 0)
            if count > 0:
                scores.append(-math.log2(count / total_ratings))

    return float(np.mean(scores)) if scores else 0.0


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
