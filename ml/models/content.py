"""M2: TF-IDF content-based recommender.

The user taste vector is a weighted sum of the feature vectors of movies they
rated at least 4.0, so the cosine score measures how close a candidate is to the
user's demonstrated taste. Rating prediction averages the ratings of the most
similar movies the user has already seen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import MovieFeatures
from ml.models.baselines import MAX_RATING, MIN_RATING

HIGH_RATING = 4.0


class ContentModel:
    """Content-based model over genres and tags (M2)."""

    name = "M2"

    def __init__(self, features: MovieFeatures, *, neighbour_k: int = 20) -> None:
        self.features = features
        self.neighbour_k = neighbour_k
        self.global_mean = 0.0
        self.user_means: dict[int, float] = {}
        self.history: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, ratings: pd.DataFrame) -> None:
        users = ratings["userId"].to_numpy(dtype=int)
        movies = ratings["movieId"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)

        self.global_mean = float(values.mean())
        self.user_means = ratings.groupby("userId")["rating"].mean().astype(float).to_dict()

        positions, valid = self.features.rows_for(movies)
        self.history = {}
        for user in np.unique(users):
            mask = (users == user) & valid
            if mask.any():
                self.history[int(user)] = (positions[mask], values[mask])

    def _profile(self, user_id: int) -> np.ndarray | None:
        history = self.history.get(int(user_id))
        if history is None:
            return None

        positions, values = history
        weights = np.where(values >= HIGH_RATING, values - (HIGH_RATING - 1.0), 0.0)
        if weights.sum() <= 0:
            weights = np.ones_like(values)

        profile = self.features.matrix[positions].T @ weights
        profile = np.asarray(profile, dtype=float).ravel()
        norm = np.linalg.norm(profile)
        if norm == 0:
            return None
        return profile / norm

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(movie_ids, dtype=int)
        scores = np.zeros(len(ids), dtype=float)

        profile = self._profile(user_id)
        if profile is None:
            return scores

        positions, valid = self.features.rows_for(ids)
        if valid.any():
            scores[valid] = np.asarray(self.features.matrix[positions[valid]] @ profile).ravel()
        return scores

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        users = np.asarray(user_ids, dtype=int)
        items = np.asarray(movie_ids, dtype=int)
        predictions = np.full(len(users), self.global_mean, dtype=float)

        for user in np.unique(users):
            mask = users == user
            mean = self.user_means.get(int(user), self.global_mean)
            predictions[mask] = mean

            history = self.history.get(int(user))
            if history is None:
                continue
            history_positions, history_ratings = history

            positions, valid = self.features.rows_for(items[mask])
            if not valid.any():
                continue

            similarity = (
                self.features.matrix[history_positions] @ self.features.matrix[positions[valid]].T
            )
            similarity = np.asarray(similarity.todense(), dtype=float)
            similarity = np.clip(similarity, 0.0, None)

            top = min(self.neighbour_k, similarity.shape[0])
            if top > 0:
                best = np.argpartition(-similarity, top - 1, axis=0)[:top]
                weights = np.take_along_axis(similarity, best, axis=0)
                neighbours = history_ratings[best]
                denominator = weights.sum(axis=0)
                numerator = (weights * neighbours).sum(axis=0)
                blended = np.full(valid.sum(), mean)
                np.divide(numerator, denominator, out=blended, where=denominator > 0)
            else:  # pragma: no cover - history is never empty here
                blended = np.full(valid.sum(), mean)

            user_predictions = predictions[mask]
            user_predictions[valid] = blended
            predictions[mask] = user_predictions

        return np.clip(predictions, MIN_RATING, MAX_RATING)

    def describe(self) -> dict[str, object]:
        """Return the hyper-parameters for the model manifest."""

        return {
            "features": "tf-idf over genres and tags",
            "vocabulary_size": self.features.vocabulary_size,
            "neighbour_k": self.neighbour_k,
            "high_rating": HIGH_RATING,
        }
