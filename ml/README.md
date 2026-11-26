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

The runner currently validates the config. Data preparation and the evaluation
split are available; models arrive in the next Phase 2 slices.

## Data exploration

```bash
python -m ml.eda data/raw/ml-25m
python -m ml.eda data/processed/ml-25m-subset
```

Writes `summary.md` plus six figures to `docs/eda/<dataset>/`. The presentation
notebook is [`notebooks/01_data_exploration.ipynb`](notebooks/01_data_exploration.ipynb).

## Evaluation split

```bash
python -m ml.pipeline.split --config ml/config.yaml
```

Per-user temporal hold-out: the newest 20% of each user's ratings become the test
set, the newest 10% of the remaining history becomes validation, and the rest is
train. The split is deterministic (timestamp order, movie id breaks ties). Users
with fewer than 5 ratings are dropped. Partitions are written under
`data/processed/<dataset>/splits/` (gitignored) and summarised in
[`../docs/splits/ml-25m-subset.json`](../docs/splits/ml-25m-subset.json).

Current subset: train 1,068,184 / validation 118,785 / test 291,879 ratings,
9,960 users, 0 dropped.

## Tests

```bash
cd ml
python -m pip install pytest==8.3.4 ruff==0.8.4
python -m pytest
ruff check .
ruff format --check .
```

Metrics live in `ml/evaluation/metrics.py` (RMSE, MAE, Precision/Recall/NDCG@K,
coverage, novelty) and are unit-tested against hand-computed values.

## Layout

- `pipeline/` — ingest, clean, features, split, train, export
- `evaluation/` — metrics, cold-start, alpha sweep, significance
- `notebooks/` — exploratory analysis only (never imported by the app)
- `artifacts/` — versioned `model.joblib` + `manifest.json` (gitignored)

Raw MovieLens data must never be committed or redistributed.
