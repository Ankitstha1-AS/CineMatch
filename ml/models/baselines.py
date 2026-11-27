"""M0 (global mean + biases) and M1 (popularity) baselines.

M0 is the rating-prediction lower bound from AGENTS.md section 4. M1 is the
ranking baseline and the cold-start prior; its weighted rating shrinks sparse
items towards the catalogue mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_RATING = 0.5
MAX_RATING = 5.0


class BiasBaseline:
    """Global mean plus regularised user and item biases (M0)."""

    name = "M0"

    def __init__(self, regularisation: float = 10.0, iterations: int = 10) -> None:
        self.regularisation = regularisation
        self.iterations = iterations
        self.global_mean = 0.0
        self.user_ids = np.empty(0, dtype=int)
        self.item_ids = np.empty(0, dtype=int)
        self.user_bias = np.empty(0, dtype=float)
        self.item_bias = np.empty(0, dtype=float)

    def fit(self, ratings: pd.DataFrame) -> None:
        users = ratings["userId"].to_numpy(dtype=int)
        items = ratings["movieId"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)

        self.global_mean = float(values.mean())
        self.user_ids, user_pos = np.unique(users, return_inverse=True)
        self.item_ids, item_pos = np.unique(items, return_inverse=True)

        user_counts = np.bincount(user_pos, minlength=len(self.user_ids))
        item_counts = np.bincount(item_pos, minlength=len(self.item_ids))
        self.user_bias = np.zeros(len(self.user_ids))
        self.item_bias = np.zeros(len(self.item_ids))

        for _ in range(self.iterations):
            residuals = values - self.global_mean - self.item_bias[item_pos]
            self.user_bias = np.bincount(
                user_pos, weights=residuals, minlength=len(self.user_ids)
            ) / (user_counts + self.regularisation)

            residuals = values - self.global_mean - self.user_bias[user_pos]
            self.item_bias = np.bincount(
                item_pos, weights=residuals, minlength=len(self.item_ids)
            ) / (item_counts + self.regularisation)

    def _positions(self, ids: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.searchsorted(known, ids)
        if known.size == 0:
            return positions, np.zeros(len(ids), dtype=bool)
        safe = np.minimum(positions, known.size - 1)
        valid = (positions < known.size) & (known[safe] == ids)
        return positions, valid

    def user_positions(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map user ids to bias-vector positions."""

        return self._positions(np.asarray(ids, dtype=int), self.user_ids)

    def item_positions(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map movie ids to bias-vector positions."""

        return self._positions(np.asarray(ids, dtype=int), self.item_ids)

    def predict_raw(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Predict without clipping, so residuals are continuous."""

        users = np.asarray(user_ids, dtype=int)
        items = np.asarray(movie_ids, dtype=int)

        user_pos, user_valid = self.user_positions(users)
        item_pos, item_valid = self.item_positions(items)

        prediction = np.full(len(users), self.global_mean)
        if self.user_bias.size:
            prediction += np.where(
                user_valid, self.user_bias[np.minimum(user_pos, self.user_bias.size - 1)], 0.0
            )
        if self.item_bias.size:
            prediction += np.where(
                item_valid, self.item_bias[np.minimum(item_pos, self.item_bias.size - 1)], 0.0
            )
        return prediction

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        return np.clip(self.predict_raw(user_ids, movie_ids), MIN_RATING, MAX_RATING)

    def describe(self) -> dict[str, object]:
        """Return the hyper-parameters for the model manifest."""

        return {"regularisation": self.regularisation, "iterations": self.iterations}

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        users = np.full(len(movie_ids), user_id, dtype=int)
        return self.predict(users, movie_ids)


class PopularityBaseline:
    """Weighted-rating popularity baseline (M1)."""

    name = "M1"

    def __init__(self, min_votes: float = 10.0) -> None:
        self.min_votes = min_votes
        self.global_mean = 0.0
        self.item_ids = np.empty(0, dtype=int)
        self.weighted_rating = np.empty(0, dtype=float)

    def fit(self, ratings: pd.DataFrame) -> None:
        stats = ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        self.item_ids = stats.index.to_numpy(dtype=int)
        counts = stats["count"].to_numpy(dtype=float)
        means = stats["mean"].to_numpy(dtype=float)
        self.global_mean = float(ratings["rating"].mean())

        self.weighted_rating = (counts * means + self.min_votes * self.global_mean) / (
            counts + self.min_votes
        )

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        del user_ids  # M1 is user-independent
        items = np.asarray(movie_ids, dtype=int)
        positions = np.searchsorted(self.item_ids, items)
        if self.item_ids.size == 0:
            valid = np.zeros(len(items), dtype=bool)
        else:
            safe = np.minimum(positions, self.item_ids.size - 1)
            valid = (positions < self.item_ids.size) & (self.item_ids[safe] == items)

        scores = np.full(len(items), self.global_mean)
        if self.item_ids.size:
            scores[valid] = self.weighted_rating[positions[valid]]
        return np.clip(scores, MIN_RATING, MAX_RATING)

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        users = np.full(len(movie_ids), user_id, dtype=int)
        return self.predict(users, movie_ids)

    def describe(self) -> dict[str, object]:
        """Return the hyper-parameters for the model manifest."""

        return {"min_votes": self.min_votes}
