"""M3: item-based collaborative filtering with cosine similarity.

Similarities are computed between item rating vectors and pruned to the ``k``
nearest neighbours per item, which keeps the model small enough to score the
whole catalogue in memory. A user's score for an item is the similarity-weighted
average of their ratings on that item's neighbours.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from ml.models.baselines import MAX_RATING, MIN_RATING


class ItemKNN:
    """Item-based KNN recommender (M3)."""

    name = "M3"

    def __init__(
        self,
        *,
        neighbours: int = 50,
        significance: float = 10.0,
        shrinkage: float = 5.0,
    ) -> None:
        self.neighbours = neighbours
        self.significance = significance
        self.shrinkage = shrinkage
        self.global_mean = 0.0
        self.user_means: dict[int, float] = {}
        self.item_ids = np.empty(0, dtype=int)
        self.item_index: dict[int, int] = {}
        self.user_history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.similarity = sparse.csr_matrix((0, 0), dtype=np.float32)
        self.abs_similarity = sparse.csr_matrix((0, 0), dtype=np.float32)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def fit(self, ratings: pd.DataFrame) -> None:
        users = ratings["userId"].to_numpy(dtype=int)
        movies = ratings["movieId"].to_numpy(dtype=int)
        values = ratings["rating"].to_numpy(dtype=float)

        self.global_mean = float(values.mean())
        self.user_means = ratings.groupby("userId")["rating"].mean().astype(float).to_dict()

        user_ids, user_pos = np.unique(users, return_inverse=True)
        item_ids, item_pos = np.unique(movies, return_inverse=True)
        self.item_ids = item_ids
        self.item_index = {int(movie): position for position, movie in enumerate(item_ids)}

        matrix = sparse.csr_matrix(
            (values, (user_pos, item_pos)),
            shape=(len(user_ids), len(item_ids)),
            dtype=np.float32,
        )
        item_matrix = matrix.T.tocsr()

        norms = np.sqrt(item_matrix.multiply(item_matrix).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        normalised = sparse.csr_matrix(sparse.diags(1.0 / norms) @ item_matrix, dtype=np.float32)

        binary = item_matrix.copy()
        binary.data = np.ones_like(binary.data, dtype=np.float32)

        self.similarity = self._neighbour_matrix(normalised, binary)
        self.abs_similarity = abs(self.similarity)

        frame = pd.DataFrame({"user": users, "item": item_pos, "rating": values})
        self.user_history = {
            int(user): (
                group["item"].to_numpy(dtype=int),
                group["rating"].to_numpy(dtype=float),
            )
            for user, group in frame.groupby("user", sort=False)
        }

    def _neighbour_matrix(
        self, normalised: sparse.csr_matrix, binary: sparse.csr_matrix
    ) -> sparse.csr_matrix:
        n_items = normalised.shape[0]
        neighbour_count = min(self.neighbours, n_items - 1)
        if neighbour_count <= 0:
            return sparse.csr_matrix((n_items, n_items), dtype=np.float32)

        model = NearestNeighbors(
            n_neighbors=neighbour_count + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        model.fit(normalised)
        distances, indices = model.kneighbors(normalised)
        similarities = 1.0 - distances

        rows = np.repeat(np.arange(n_items), indices.shape[1])
        columns = indices.ravel()
        keep = rows != columns
        rows, columns = rows[keep], columns[keep]
        similarities = similarities.ravel()[keep]

        # Down-weight pairs whose similarity rests on very few co-rating users.
        overlap = np.asarray(
            binary[rows].multiply(binary[columns]).sum(axis=1), dtype=float
        ).ravel()
        confidence = overlap / (overlap + self.significance)
        similarities = similarities * confidence

        return sparse.csr_matrix(
            (similarities.astype(np.float32), (rows, columns)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )

    def _user_scores(self, user_id: int) -> np.ndarray | None:
        history = self.user_history.get(int(user_id))
        if history is None:
            return None

        positions, values = history
        fallback = self.user_means.get(int(user_id), self.global_mean)

        numerator = np.asarray(self.similarity[positions].T.dot(values), dtype=float).ravel()
        denominator = np.asarray(
            self.abs_similarity[positions].T.dot(np.ones(len(values), dtype=float)),
            dtype=float,
        ).ravel()

        scores = (numerator + self.shrinkage * fallback) / (denominator + self.shrinkage)
        return scores

    def score_items(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(movie_ids, dtype=int)
        fallback = self.user_means.get(int(user_id), self.global_mean)
        scores = np.full(len(ids), fallback, dtype=float)

        vector = self._user_scores(user_id)
        if vector is None:
            return scores

        positions = np.fromiter(
            (self.item_index.get(int(movie), -1) for movie in ids),
            dtype=int,
            count=len(ids),
        )
        valid = positions >= 0
        scores[valid] = vector[positions[valid]]
        return scores

    def predict(self, user_ids: np.ndarray, movie_ids: np.ndarray) -> np.ndarray:
        users = np.asarray(user_ids, dtype=int)
        items = np.asarray(movie_ids, dtype=int)
        predictions = np.full(len(users), self.global_mean, dtype=float)

        for user in np.unique(users):
            mask = users == user
            predictions[mask] = self.score_items(user, items[mask])

        return np.clip(predictions, MIN_RATING, MAX_RATING)

    def describe(self) -> dict[str, object]:
        """Return the hyper-parameters for the model manifest."""

        return {
            "neighbours": self.neighbours,
            "significance": self.significance,
            "shrinkage": self.shrinkage,
            "items": self.n_items,
            "similarity_pairs": int(self.similarity.nnz),
        }
