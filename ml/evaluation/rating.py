"""Rating-prediction evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.evaluation.metrics import mae, rmse
from ml.models.base import RatingModel


def evaluate_rating(model: RatingModel, frame: pd.DataFrame) -> dict[str, float]:
    """Predict every pair in ``frame`` and return RMSE, MAE and the count."""

    if frame.empty:
        raise ValueError("cannot evaluate an empty ratings frame")

    predictions = model.predict(
        frame["userId"].to_numpy(dtype=int),
        frame["movieId"].to_numpy(dtype=int),
    )
    actual = frame["rating"].to_numpy(dtype=float)

    return {
        "rmse": rmse(actual, predictions),
        "mae": mae(actual, predictions),
        "predictions": float(len(frame)),
    }


def per_user_errors(model: RatingModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Return one RMSE/MAE row per user (used by the significance tests)."""

    if frame.empty:
        raise ValueError("cannot evaluate an empty ratings frame")

    scored = frame[["userId", "rating"]].copy()
    scored["prediction"] = model.predict(
        frame["userId"].to_numpy(dtype=int),
        frame["movieId"].to_numpy(dtype=int),
    )
    scored["squared_error"] = np.square(scored["rating"] - scored["prediction"])
    scored["absolute_error"] = np.abs(scored["rating"] - scored["prediction"])

    grouped = scored.groupby("userId", sort=True)
    return pd.DataFrame(
        {
            "rmse": np.sqrt(grouped["squared_error"].mean()),
            "mae": grouped["absolute_error"].mean(),
            "predictions": grouped.size(),
        }
    ).reset_index()
