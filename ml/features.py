"""Catalogue feature engineering shared by the content model.

A movie is described by its genres and its aggregated MovieLens tags, because
AGENTS.md section 4 defines the content score over "genres + tags". The text is
turned into a TF-IDF matrix whose rows are L2-normalised, so the dot product of
two rows is their cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

NO_GENRES = "(no genres listed)"


@dataclass(frozen=True, slots=True)
class MovieFeatures:
    """TF-IDF vectors for the catalogue, indexed by MovieLens movie id."""

    movie_ids: np.ndarray
    matrix: sparse.csr_matrix
    index: dict[int, int]
    vocabulary_size: int

    @classmethod
    def build(
        cls,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
        *,
        min_df: int = 2,
    ) -> MovieFeatures:
        """Build features from ``movies.csv`` and (optionally) ``tags.csv``."""

        if movies.empty:
            raise ValueError("movies frame is empty")
        if "movieId" not in movies.columns or "genres" not in movies.columns:
            raise ValueError("movies must have movieId and genres columns")

        catalogue = movies[["movieId", "genres"]].drop_duplicates("movieId").copy()
        catalogue = catalogue.sort_values("movieId").reset_index(drop=True)

        tag_text: dict[int, str] = {}
        if tags is not None and not tags.empty:
            aggregated = (
                tags.dropna(subset=["tag"])
                .assign(tag=lambda frame: frame["tag"].str.strip().str.lower())
                .query("tag != ''")
                .groupby("movieId")["tag"]
                .agg(lambda values: " ".join(sorted(set(values))))
            )
            tag_text = {int(movie): str(text) for movie, text in aggregated.items()}

        documents: list[str] = []
        for movie_id, genres in zip(
            catalogue["movieId"].to_numpy(dtype=int),
            catalogue["genres"].fillna("").to_numpy(dtype=str),
            strict=True,
        ):
            tokens = [
                genre.strip().lower().replace("-", "")
                for genre in genres.split("|")
                if genre and genre != NO_GENRES
            ]
            tags_for_movie = tag_text.get(int(movie_id), "")
            documents.append(" ".join(tokens + ([tags_for_movie] if tags_for_movie else [])))

        vectorizer = TfidfVectorizer(
            token_pattern=r"[^\s]+",
            min_df=min_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = vectorizer.fit_transform(documents)
        matrix = sparse.csr_matrix(matrix, dtype=np.float32)
        matrix = _normalise_rows(matrix)

        movie_ids = catalogue["movieId"].to_numpy(dtype=int)
        return cls(
            movie_ids=movie_ids,
            matrix=matrix,
            index={int(movie): position for position, movie in enumerate(movie_ids)},
            vocabulary_size=len(vectorizer.vocabulary_),
        )

    def rows_for(self, movie_ids: np.ndarray | list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return positions and a validity mask for the requested movie ids."""

        ids = np.asarray(movie_ids, dtype=int)
        positions = np.fromiter(
            (self.index.get(int(movie), -1) for movie in ids),
            dtype=int,
            count=len(ids),
        )
        valid = positions >= 0
        return positions, valid


def _normalise_rows(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return a copy with every row scaled to unit L2 norm."""

    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    scale = sparse.diags((1.0 / norms).astype(np.float32))
    return sparse.csr_matrix(scale @ matrix, dtype=np.float32)
