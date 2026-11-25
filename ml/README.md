# ML workspace

Offline training and evaluation live here, separate from the FastAPI runtime.

## Setup

```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use **Python 3.12**. `scikit-surprise==1.1.5` has no Python 3.13 wheel, so pip
would try to compile it.

### Surprise and Smart App Control

On this laptop the Surprise wheel installs but cannot be imported: Windows Smart
App Control blocks its unsigned DLL. NumPy, SciPy, scikit-learn and pandas are
unaffected. As recorded in [`../docs/decisions.md`](../docs/decisions.md), M4
uses Surprise when it imports and otherwise falls back to a pure NumPy/SciPy
SVD. Do not disable Smart App Control just for this project; CI runs on Linux,
where Surprise imports normally.

## Run

From the repository root:

```bash
python -m ml.pipeline.run --config ml/config.yaml
```

Phase 0 only validates the config. Ingest, splitting, training, and export are
implemented in Phases 2 and 3.

## Layout

- `pipeline/` — ingest, clean, features, split, train, export
- `evaluation/` — metrics, cold-start, alpha sweep, significance
- `notebooks/` — exploratory analysis only (never imported by the app)
- `artifacts/` — versioned `model.joblib` + `manifest.json` (gitignored)

Raw MovieLens data must never be committed or redistributed.
