"""Top-N ranking evaluation against the full unrated catalogue.

For every test user the model scores **all** catalogue items the user did not
rate in the training history, the top ``k`` are selected with a heap-style
partial sort, and Precision/Recall/NDCG are computed against test items at or
above the relevance threshold. Users without a relevant test item are skipped.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from ml.models.base import RatingModel


def evaluate_ranking(
    model: RatingModel,
    *,
    test: pd.DataFrame,
    history: pd.DataFrame,
    catalogue: np.ndarray | list[int],
    k_values: tuple[int, ...] = (5, 10, 20),
    relevance_threshold: float = 4.0,
    max_users: int | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Rank the full catalogue per user and aggregate top-N metrics.

    Args:
        model: Fitted model exposing ``score_items``.
        test: Held-out ratings; items at or above the threshold are relevant.
        history: Ratings the model was fitted on, excluded from candidates.
        catalogue: Every movie id in the catalogue.
        k_values: Cut-offs to report.
        relevance_threshold: Minimum test rating considered relevant.
        max_users: Optional deterministic sample of users (development only).
        seed: Sampling seed when ``max_users`` is set.

    Returns:
        A JSON-serialisable mapping with per-k metrics, coverage and novelty.
    """

    if test.empty or history.empty:
        raise ValueError("test and history must both be non-empty")

    cuts = tuple(sorted(set(k_values)))
    movies = np.array(sorted({int(movie) for movie in catalogue}), dtype=int)
    if movies.size == 0:
        raise ValueError("catalogue is empty")

    position = {int(movie): index for index, movie in enumerate(movies)}

    history_by_user: dict[int, set[int]] = {
        int(user): {int(movie) for movie in frame["movieId"]}
        for user, frame in history.groupby("userId")
    }
    relevant_by_user: dict[int, set[int]] = {
        int(user): {
            int(movie) for movie in frame.loc[frame["rating"] >= relevance_threshold, "movieId"]
        }
        for user, frame in test.groupby("userId")
    }

    users = sorted(user for user, relevant in relevant_by_user.items() if relevant)
    if not users:
        raise ValueError("no test user has a relevant item")

    if max_users is not None and len(users) > max_users:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(np.array(users), size=max_users, replace=False)
        users = sorted(int(user) for user in sampled)

    popularity = history.groupby("movieId").size().to_dict()
    total_history = len(history)

    metric_sums = {(cut, metric): 0.0 for cut in cuts for metric in ("precision", "recall", "ndcg")}
    shown: dict[int, set[int]] = {cut: set() for cut in cuts}
    novelty_sum = dict.fromkeys(cuts, 0.0)
    novelty_count = dict.fromkeys(cuts, 0)
    evaluated = 0

    for user in users:
        mask = np.ones(movies.size, dtype=bool)
        for movie in history_by_user.get(user, set()):
            index = position.get(movie)
            if index is not None:
                mask[index] = False

        candidates = movies[mask]
        if candidates.size == 0:
            continue

        scores = np.asarray(model.score_items(user, candidates), dtype=float)
        scores = np.where(np.isfinite(scores), scores, -np.inf)

        largest = min(max(cuts), candidates.size)
        if largest < candidates.size:
            top_index = np.argpartition(-scores, largest - 1)[:largest]
        else:
            top_index = np.arange(candidates.size)
        top_index = top_index[np.argsort(-scores[top_index], kind="stable")]
        ranked = candidates[top_index]

        relevant = relevant_by_user[user]
        for cut in cuts:
            recommendations = ranked[:cut]
            metric_sums[(cut, "precision")] += precision_at_k(recommendations, relevant, cut)
            metric_sums[(cut, "recall")] += recall_at_k(recommendations, relevant, cut)
            metric_sums[(cut, "ndcg")] += ndcg_at_k(recommendations, relevant, cut)
            shown[cut].update(int(movie) for movie in recommendations)

            for movie in recommendations:
                count = popularity.get(int(movie), 0)
                if count > 0:
                    novelty_sum[cut] += -math.log2(count / total_history)
                    novelty_count[cut] += 1

        evaluated += 1

    if evaluated == 0:
        raise ValueError("no user had a candidate left after removing history")

    metrics: dict[str, float] = {}
    for cut in cuts:
        for metric in ("precision", "recall", "ndcg"):
            metrics[f"{metric}@{cut}"] = metric_sums[(cut, metric)] / evaluated

    coverage = {f"coverage@{cut}": len(shown[cut]) / movies.size for cut in cuts}
    novelty = {
        f"novelty@{cut}": (novelty_sum[cut] / novelty_count[cut] if novelty_count[cut] else 0.0)
        for cut in cuts
    }

    return {
        "users": evaluated,
        "relevant_users": len(users),
        "metrics": metrics,
        "coverage": coverage,
        "novelty": novelty,
    }
