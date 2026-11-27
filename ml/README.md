# ML workspace

Offline training and evaluation live here, separate from the FastAPI runtime.

## Setup

```bash
cd ml
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use **Python 3.12**. scikit-surprise is not required: M4 is a NumPy/SciPy
implementation (see [ADR-002](../docs/decisions.md)).

### Surprise and Smart App Control

On this laptop the Surprise wheel installs but cannot be imported: Windows Smart
App Control blocks its unsigned DLL. NumPy, SciPy, scikit-learn and pandas are
unaffected. As recorded in [`../docs/decisions.md`](../docs/decisions.md), M4 is
implemented as a pure NumPy/SciPy biased matrix factorisation, so the pipeline
runs identically here and in CI. Do not disable Smart App Control just for this
project.

## Run

From the repository root:

```bash
python -m ml.pipeline.run --config ml/config.yaml      # validate config
python -m ml.pipeline.split --config ml/config.yaml    # build the split
python -m ml.pipeline.evaluate --config ml/config.yaml # M0-M4 on validation
```

`evaluate` fits the bias, popularity, TF-IDF content, item-KNN and biased SVD
models, then writes metrics, hyper-parameters, fit times and dataset hashes to
`docs/results/<dataset>-<split>.json`. Use `--split test` only for the final
once-only run; `--max-users` samples ranking users for quick development runs.

To persist fitted models and the manifest as a versioned bundle:

```bash
python -m ml.pipeline.export --config ml/config.yaml --version baselines
```

The bundle (`ml/artifacts/<version>/{models.joblib,manifest.json}`) is gitignored
and records the dataset hashes, seeds, hyper-parameters and validation metrics,
so a run can be reproduced and audited later.

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

## Phase 2 results

Validation results for M0–M4, the protocol and dataset fingerprints are written
up in [`../docs/results/phase-2-baselines.md`](../docs/results/phase-2-baselines.md).
Headline: M4 has the best rating error (RMSE 0.8460), M3 the best ranking
(NDCG@10 0.0609), and M2 the widest catalogue coverage (51.2% at K = 10).

## Layout

- `data.py`, `splits.py`, `features.py` — loading, temporal split, TF-IDF features
- `models/` — M0 biases, M1 popularity, M2 content, M3 item-KNN, M4 biased SVD
- `pipeline/` — ingest, clean, features, split, train, evaluate, export
- `evaluation/` — metrics, rating/ranking harness, cold-start, alpha sweep
- `notebooks/` — exploratory analysis only (never imported by the app)
- `artifacts/` — versioned `model.joblib` + `manifest.json` (gitignored)

Raw MovieLens data must never be committed or redistributed.
