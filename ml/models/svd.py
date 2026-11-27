"""M4: biased matrix factorisation with truncated SVD.

scikit-surprise is unavailable in this environment (ADR-002), so the model is a
NumPy/SciPy implementation: regularised user/item biases carry the rating level,
and truncated SVD of the residual matrix learns the latent factors. The solver
is seeded, so runs are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds

from ml.models.baselines import MAX_RATING, MIN_RATING, BiasBaseline


class SVDModel:
    """Biased matrix-factorisation recommender (M4)."""

    name = "M4"

    def __init__(
        self,
        *,
        factors: int = 32,
        bias_regularisation: float = 10.0,
        bias_iterations: int = 10,
        seed: int = 42,
    ) -> None:
        self.factors = factors
        self.bias_regularisation = bias_regularisation
        self.bias_iterations = bias_iterations
        self.seed = seed
        self.bias = BiasBaseline(regularisation=bias_regularisation, iterations=bias_iterations)
        self.user_factors = np.empty((0, 0), dtype=float)
        self.item_factors = np.empty((0, 0), dtype=float)

    def fit(self, ratings: pd.DataFrame) -> None:
        users = ratings["userId"].to_numpy(dtype=int)
        movies = ratings["movieId"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)

        self.bias.fit(ratings)
        self.user_factors = np.zeros((len(self.bias.user_ids), 0), dtype=float)
        self.item_factors = np.zeros((len(self.bias.item_ids), 0), dtype=float)

        shape = (len(self.bias.user_ids), len(self.bias.item_ids))
        components = min(self.factors, min(shape) - 1)
        if components < 1:
            return

        user_pos, _ = self.bias.user_positions(users)
        item_pos, _ = self.bias.item_positions(movies)
        residuals = values - self.bias.predict_raw(users, movies)

        matrix = sparse.csr_matrix((residuals, (user_pos, item_pos)), shape=shape, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        v0 = rng.standard_normal(min(shape))

        left, singular, right = svds(matrix, k=components, v0=v0, maxiter=2000, tol=1e-4)

        self.user_factors = left * singular
        self.item_factors = right.T

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        users = np.asarray(user_ids, dtype=int)
        items = np.asarray(movie_ids, dtype=int)

        base = self.bias.predict_raw(users, items)
        latent = self._latent(users, items)
        return np.clip(base + latent, MIN_RATING, MAX_RATING)

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(movie_ids, dtype=int)
        scores = np.full(len(ids), self.bias.global_mean, dtype=float)

        user_pos, user_valid = self.bias.user_positions(np.array([user_id]))
        if user_valid[0] and self.user_factors.shape[1]:
            scores += self.bias.user_bias[user_pos[0]]

        item_pos, item_valid = self.bias.item_positions(ids)
        if item_valid.any() and self.user_factors.shape[1]:
            scores[item_valid] += self.bias.item_bias[item_pos[item_valid]]
            if user_valid[0]:
                factors = self.item_factors[item_pos[item_valid]]
                scores[item_valid] += factors @ self.user_factors[user_pos[0]]

        return np.clip(scores, MIN_RATING, MAX_RATING)

    def _latent(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        latent = np.zeros(len(users), dtype=float)
        if not self.user_factors.shape[1]:
            return latent

        user_pos, user_valid = self.bias.user_positions(users)
        item_pos, item_valid = self.bias.item_positions(items)
        both = user_valid & item_valid
        if both.any():
            latent[both] = np.einsum(
                "ij,ij->i",
                self.user_factors[user_pos[both]],
                self.item_factors[item_pos[both]],
            )
        return latent

    def describe(self) -> dict[str, object]:
        """Return the hyper-parameters for the model manifest."""

        return {
            "factors": int(self.user_factors.shape[1]),
            "bias_regularisation": self.bias_regularisation,
            "bias_iterations": self.bias_iterations,
            "seed": self.seed,
        }
