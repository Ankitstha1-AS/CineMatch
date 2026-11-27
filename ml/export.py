"""Persist fitted models and their manifest as a versioned artifact bundle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml.models.base import RatingModel


def save_artifact_bundle(
    models: dict[str, RatingModel],
    out_dir: Path,
    manifest: dict[str, object],
) -> Path:
    """Write ``models.joblib`` and ``manifest.json`` under ``out_dir``.

    Returns the bundle directory. The manifest must already contain the
    version, dataset, parameters and metrics; this function only serialises.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, out_dir / "models.joblib", compress=3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def load_artifact_bundle(bundle_dir: Path) -> tuple[dict[str, RatingModel], dict[str, object]]:
    """Load a bundle written by :func:`save_artifact_bundle`."""

    models = joblib.load(bundle_dir / "models.joblib")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    return models, manifest
