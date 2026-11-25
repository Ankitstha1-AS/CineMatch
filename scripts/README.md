# Helper scripts

Data import, validation, seeding and maintenance scripts.

| Script | Purpose |
|---|---|
| `download_movielens.py` | Download/extract MovieLens into `data/raw/` (gitignored) |
| `validate_movielens.py` | Schema, duplicate, integrity and sparsity checks |
| `build_subset.py` | Fixed filters + seeded user sample → `data/processed/` |

## Download

```bash
python scripts/download_movielens.py latest-small 1m
python scripts/download_movielens.py all        # adds the ~250 MB ml-25m archive
```

Requires `certifi` (listed in `ml/requirements.txt`) so TLS verification works on
Windows Python installs without a complete default CA chain.

## Validate

```bash
python scripts/validate_movielens.py data/raw/ml-latest-small
python scripts/validate_movielens.py data/raw/ml-1m
```

Results from the first run are recorded in [`../docs/data-inventory.md`](../docs/data-inventory.md).

## Build the training subset

```bash
python scripts/build_subset.py data/raw/ml-25m \
  --out data/processed/ml-25m-subset \
  --min-movie-ratings 10 --min-user-ratings 20 \
  --max-users 10000 --seed 42 \
  --manifest-out docs/subsets/ml-25m-subset.json
```

Produces `ratings.csv`, `movies.csv`, `tags.csv`, `links.csv`, `genome-tags.csv`
and `manifest.json`. The reasoning behind the filters is in
[`../docs/eda/README.md`](../docs/eda/README.md).

## EDA

```bash
python -m ml.eda data/raw/ml-25m
python -m ml.eda data/processed/ml-25m-subset
```

Writes `summary.md` and six figures under `docs/eda/<dataset>/`.

Planned additions:

- `import_movielens.py` — load the cleaned subset into PostgreSQL (Phase 1)
- `seed_demo_data.py` — demo users and ratings (Phase 1/2)

Scripts must read configuration from environment variables or CLI flags and must
never contain secrets. Raw MovieLens files are never committed.
