"""Tests for content features and the M2/M3/M4 models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ml.features import MovieFeatures
from ml.models import ContentModel, ItemKNN, SVDModel

MOVIES = pd.DataFrame(
    {
        "movieId": [1, 2, 3, 4, 5],
        "title": ["A", "B", "C", "D", "E"],
        "genres": [
            "Action",
            "Action|Adventure",
            "Comedy",
            "Drama",
            "Action|Thriller",
        ],
    }
)

TAGS = pd.DataFrame(
    {
        "userId": [1, 2],
        "movieId": [1, 2],
        "tag": ["explosions", "explosions"],
        "timestamp": [1, 2],
    }
)


def test_features_are_normalised_and_indexed() -> None:
    features = MovieFeatures.build(MOVIES, TAGS, min_df=1)

    assert list(features.movie_ids) == [1, 2, 3, 4, 5]
    assert features.index[3] == 2
    assert features.matrix.shape[0] == 5
    assert features.vocabulary_size >= 6  # five genres plus the shared tag

    norms = np.sqrt(features.matrix.multiply(features.matrix).sum(axis=1)).A1
    assert np.allclose(norms, 1.0)

    positions, valid = features.rows_for([1, 99])
    assert positions.tolist() == [0, -1]
    assert valid.tolist() == [True, False]


def test_features_treat_missing_tags_as_genres_only() -> None:
    features = MovieFeatures.build(MOVIES, None, min_df=1)
    assert features.vocabulary_size > 0

    with pytest.raises(ValueError):
        MovieFeatures.build(pd.DataFrame({"movieId": [], "genres": []}))


def test_content_model_prefers_shared_genres() -> None:
    features = MovieFeatures.build(MOVIES, TAGS, min_df=1)
    model = ContentModel(features)

    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2],
            "movieId": [1, 3, 4],
            "rating": [5.0, 1.0, 5.0],
        }
    )
    model.fit(ratings)

    scores = model.score_items(1, np.array([2, 3, 5]))
    assert scores[0] > scores[1]  # Action/Adventure beats Comedy
    assert scores[2] > scores[1]  # Action/Thriller beats Comedy

    # A user with no history gets neutral scores.
    assert np.allclose(model.score_items(999, np.array([1, 2])), 0.0)

    predictions = model.predict(np.array([1, 1, 999]), np.array([2, 3, 1]))
    assert np.all((predictions >= 0.5) & (predictions <= 5.0))
    assert predictions[0] >= predictions[1]  # closer to the 5-star movie


def test_item_knn_scores_correlated_items_higher() -> None:
    rows = [
        (10, 1, 5.0),
        (10, 2, 5.0),
        (10, 3, 5.0),
        (10, 4, 1.0),
        (20, 1, 4.0),
        (20, 2, 5.0),
        (20, 3, 4.0),
        (20, 4, 2.0),
        (30, 1, 5.0),
        (30, 2, 4.0),
        (30, 3, 5.0),
        (30, 4, 1.0),
        (40, 1, 4.0),
        (40, 2, 4.0),
        (40, 3, 5.0),
        (40, 4, 2.0),
        (50, 4, 5.0),
        (50, 5, 5.0),
        (60, 4, 4.0),
        (60, 5, 5.0),
    ]
    ratings = pd.DataFrame(rows, columns=["userId", "movieId", "rating"])

    model = ItemKNN(neighbours=3)
    model.fit(ratings)

    assert 10 in model.user_history  # keyed by user id, not matrix row

    scores = model.score_items(10, np.array([2, 5]))
    assert scores[0] > scores[1]

    predictions = model.predict(np.array([10, 999]), np.array([2, 2]))
    assert 0.5 <= predictions[0] <= 5.0
    assert predictions[1] == pytest.approx(model.global_mean)


def test_svd_reconstructs_a_low_rank_rating_matrix() -> None:
    rng = np.random.default_rng(7)
    user_factors = rng.normal(size=(10, 2))
    item_factors = rng.normal(size=(6, 2))
    truth = np.clip(3.0 + user_factors @ item_factors.T, 0.5, 5.0)

    users, movies, ratings = [], [], []
    for user in range(10):
        for movie in range(6):
            users.append(user + 1)
            movies.append(movie + 1)
            ratings.append(float(truth[user, movie]))
    frame = pd.DataFrame({"userId": users, "movieId": movies, "rating": ratings})

    model = SVDModel(factors=2, bias_regularisation=0.1, bias_iterations=5, seed=7)
    model.fit(frame)

    predictions = model.predict(frame["userId"].to_numpy(), frame["movieId"].to_numpy())
    residual = np.sqrt(np.mean(np.square(frame["rating"].to_numpy() - predictions)))
    assert residual < 0.35

    scores = model.score_items(1, np.array([1, 2, 3, 4, 5, 6]))
    assert np.all((scores >= 0.5) & (scores <= 5.0))
    assert np.allclose(scores, predictions[:6], atol=1e-6)
