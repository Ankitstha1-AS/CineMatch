"""Tests for the ranking and rating evaluation harness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.evaluation.ranking import evaluate_ranking
from ml.evaluation.rating import evaluate_rating, per_user_errors


class FixedScores:
    """Toy model returning pre-set scores per movie."""

    name = "toy"

    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def fit(self, ratings: pd.DataFrame) -> None:
        del ratings

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        del user_ids
        return np.array([self.scores.get(int(movie), 0.0) for movie in movie_ids])

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        del user_id
        return np.array([self.scores.get(int(movie), 0.0) for movie in movie_ids])


HISTORY = pd.DataFrame(
    {
        "userId": [1, 1, 2, 2, 2],
        "movieId": [1, 2, 1, 2, 5],
        "rating": [5.0, 4.0, 4.0, 3.0, 3.5],
    }
)

TEST = pd.DataFrame(
    {
        "userId": [1, 1, 1, 2],
        "movieId": [3, 4, 5, 4],
        "rating": [5.0, 2.0, 4.0, 3.0],
    }
)

CATALOGUE = [1, 2, 3, 4, 5]

SCORES = {1: 1.0, 2: 1.0, 3: 0.8, 4: 0.7, 5: 0.9}


def test_ranking_excludes_history_and_skips_users_without_relevant_items() -> None:
    model = FixedScores(SCORES)

    result = evaluate_ranking(
        model,
        test=TEST,
        history=HISTORY,
        catalogue=CATALOGUE,
        k_values=(1, 2, 3),
        relevance_threshold=4.0,
    )

    assert result["users"] == 1  # user 2 has no rating >= 4 in the test set
    metrics = result["metrics"]
    assert metrics["precision@1"] == pytest.approx(1.0)
    assert metrics["precision@2"] == pytest.approx(1.0)
    assert metrics["precision@3"] == pytest.approx(2 / 3)
    assert metrics["recall@1"] == pytest.approx(1 / 2)
    assert metrics["recall@2"] == pytest.approx(1.0)
    assert metrics["ndcg@2"] == pytest.approx(1.0)

    # Only the three unseen movies can appear; the rated history must be gone.
    assert result["coverage"]["coverage@3"] == pytest.approx(3 / 5)
    assert result["novelty"]["novelty@1"] > 0.0


def test_ranking_requires_a_relevant_user() -> None:
    irrelevant = TEST.assign(rating=1.0)
    model = FixedScores(SCORES)

    with pytest.raises(ValueError):
        evaluate_ranking(
            model,
            test=irrelevant,
            history=HISTORY,
            catalogue=CATALOGUE,
        )


def test_ranking_user_sampling_is_deterministic() -> None:
    test = pd.concat(
        [
            TEST,
            pd.DataFrame({"userId": [3], "movieId": [5], "rating": [5.0]}),
        ],
        ignore_index=True,
    )
    model = FixedScores(SCORES)

    first = evaluate_ranking(
        model, test=test, history=HISTORY, catalogue=CATALOGUE, max_users=1, seed=7
    )
    second = evaluate_ranking(
        model, test=test, history=HISTORY, catalogue=CATALOGUE, max_users=1, seed=7
    )

    assert first["users"] == 1
    assert first["metrics"] == second["metrics"]


def test_rating_evaluation_and_per_user_errors() -> None:
    frame = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [10, 11, 10],
            "rating": [3.0, 5.0, 4.0],
        }
    )
    model = FixedScores({10: 4.0, 11: 4.0})

    metrics = evaluate_rating(model, frame)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2 / 3))
    assert metrics["mae"] == pytest.approx(2 / 3)

    per_user = per_user_errors(model, frame)
    assert list(per_user["userId"]) == [1, 2]
    assert per_user.loc[per_user["userId"] == 1, "rmse"].iloc[0] == pytest.approx(1.0)
