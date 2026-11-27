"""Tests for the M0 and M1 baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.models.baselines import BiasBaseline, PopularityBaseline


def test_bias_baseline_learns_user_and_item_effects() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 2, 3, 1, 2, 3],
            "rating": [5.0, 4.5, 5.0, 2.0, 2.5, 1.5],
        }
    )
    model = BiasBaseline(regularisation=1.0, iterations=20)
    model.fit(ratings)

    generous = model.predict(np.array([1]), np.array([3]))[0]
    harsh = model.predict(np.array([2]), np.array([3]))[0]

    assert generous > harsh
    assert 0.5 <= generous <= 5.0
    assert 0.5 <= harsh <= 5.0


def test_bias_baseline_falls_back_to_known_means() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 2, 1, 2],
            "rating": [4.0, 4.0, 2.0, 2.0],
        }
    )
    model = BiasBaseline(regularisation=1.0, iterations=10)
    model.fit(ratings)

    unknown_user = model.predict(np.array([999]), np.array([1]))[0]
    unknown_movie = model.predict(np.array([1]), np.array([999]))[0]

    assert 0.5 <= unknown_user <= 5.0
    assert 0.5 <= unknown_movie <= 5.0


def test_popularity_baseline_shrinks_towards_the_mean() -> None:
    ratings = pd.DataFrame(
        {
            "userId": np.arange(102),
            "movieId": [1] * 100 + [2, 3],
            "rating": [4.9] * 100 + [5.0, 1.0],
        }
    )

    no_shrinkage = PopularityBaseline(min_votes=0.0)
    no_shrinkage.fit(ratings)
    confident = PopularityBaseline(min_votes=1000.0)
    confident.fit(ratings)

    sparse_score = no_shrinkage.score_items(0, np.array([2]))[0]
    popular_score = no_shrinkage.score_items(0, np.array([1]))[0]
    assert sparse_score > popular_score

    shrunk_sparse = confident.score_items(0, np.array([2]))[0]
    shrunk_popular = confident.score_items(0, np.array([1]))[0]
    assert abs(shrunk_sparse - shrunk_popular) < 0.05


def test_popularity_baseline_returns_the_global_mean_for_unknown_items() -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 2, 3],
            "movieId": [1, 1, 1],
            "rating": [3.0, 4.0, 5.0],
        }
    )
    model = PopularityBaseline(min_votes=0.0)
    model.fit(ratings)

    assert model.score_items(1, np.array([999]))[0] == pytest.approx(4.0)
