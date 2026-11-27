"""Model interfaces shared by the offline pipeline."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class RatingModel(Protocol):
    """Minimal interface every baseline and hybrid implements."""

    name: str

    def fit(self, ratings: pd.DataFrame) -> None:
        """Learn from a ratings frame (``userId``, ``movieId``, ``rating``)."""

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        """Predict ratings for user/movie pairs, clipped to the rating scale."""

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        """Return a ranking score per candidate movie; higher is better."""

    def describe(self) -> dict[str, object]:
        """Return hyper-parameters for the results manifest."""
