"""Tests for the versioned artifact bundle."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ml.export import load_artifact_bundle, save_artifact_bundle
from ml.models import BiasBaseline


def test_bundle_round_trips_models_and_manifest(tmp_path) -> None:
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [1, 2, 1, 2],
            "rating": [5.0, 4.0, 2.0, 1.0],
        }
    )
    model = BiasBaseline(regularisation=1.0, iterations=5)
    model.fit(ratings)
    expected = model.predict(np.array([1, 2]), np.array([1, 2]))

    manifest = {
        "version": "test",
        "dataset": "toy",
        "models": {"M0": {"parameters": model.describe()}},
    }
    bundle = save_artifact_bundle({"M0": model}, tmp_path / "test", manifest)

    assert (bundle / "models.joblib").is_file()
    assert (bundle / "manifest.json").is_file()

    loaded, read_manifest = load_artifact_bundle(bundle)
    assert read_manifest == manifest
    assert set(loaded) == {"M0"}

    actual = loaded["M0"].predict(np.array([1, 2]), np.array([1, 2]))
    np.testing.assert_allclose(actual, expected)
