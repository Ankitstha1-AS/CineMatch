# Helper scripts

Data import, validation, seeding and maintenance scripts.

| Script | Purpose |
|---|---|
| `download_movielens.py` | Download/extract MovieLens into `data/raw/` (gitignored) |
| `validate_movielens.py` | Schema, duplicate, integrity and sparsity checks |

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

Planned additions:

- `build_subset.py` — fixed subset filters after EDA (Phase 1)
- `import_movielens.py` — load the cleaned subset into PostgreSQL (Phase 1)
- `seed_demo_data.py` — demo users and ratings (Phase 1/2)

Scripts must read configuration from environment variables or CLI flags and must
never contain secrets. Raw MovieLens files are never committed.
