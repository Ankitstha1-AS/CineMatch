# Data inventory

Raw MovieLens files live in `data/raw/`; the training subset lives in
`data/processed/`. Both are gitignored. Raw data must never be committed or
redistributed; cite Harper & Konstan (2015).

## Downloaded

Fetched on 2026-11-25 with `scripts/download_movielens.py`.

| Dataset | Archive SHA-256 | Ratings | Users | Movies (total / rated) | Sparsity |
|---|---|---|---|---|---|
| ml-latest-small | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` | 100,836 | 610 | 9,742 / 9,724 | 0.9830 |
| ml-1m | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` | 1,000,209 | 6,040 | 3,883 / 3,706 | 0.9553 |
| ml-25m | `8b21cfb7eb1706b4ec0aac894368d90acf26ebdfb6aced3ebd4ad5bd1eb9c6aa` | 25,000,095 | 162,541 | 62,423 / 59,047 | 0.9974 |

## Training subset

Built by `scripts/build_subset.py` from ml-25m.

| Item | Value |
|---|---|
| Source ratings SHA-256 | `9c85525758597dca76e8d98cebb9c13e018c2317b09a0878b6d383179526e385` |
| Filters | min 10 ratings/movie, min 20 ratings/user, 10,000 users (seed 42) |
| Ratings | **1,478,848** |
| Users | 9,960 |
| Movies | 8,737 |
| Tags | 60,237 |
| Tag-genome tags | 1,128 |
| Sparsity | 0.9830 |
| Rating range / mean | 0.5 – 5.0 / 3.5453 |
| Time span | 1996-01-29 → 2019-11-21 |
| Manifest | [`subsets/ml-25m-subset.json`](subsets/ml-25m-subset.json) |

## Validation results

All three releases and the subset passed `scripts/validate_movielens.py`:

- no duplicate `(userId, movieId)` pairs;
- no duplicate movie ids;
- no ratings pointing at unknown movies;
- ratings and timestamps parse cleanly.

| Metric | latest-small | ml-1m | ml-25m | subset |
|---|---|---|---|---|
| Rating range | 0.5 – 5.0 | 1 – 5 | 0.5 – 5.0 | 0.5 – 5.0 |
| Mean rating | 3.5016 | 3.5816 | 3.5339 | 3.5453 |
| Median ratings/user | 70.5 | 96 | 71 | 70 |
| Median ratings/movie | 3 | 123.5 | 6 | 46 |
| Movies with zero ratings | 18 | 177 | 3,376 | 0 |
| Tags | 3,683 | none | 1,093,360 | 60,237 |
| Distinct tags | 1,589 | — | 73,050 | 12,016 |
| Movie tag coverage | 16.1% | — | 72.5% | 60.9% |

## Notes

- **Rating scales differ.** latest-small and 25M use half-star values; ml-1m uses
  integers. Normalise before comparing metrics across releases.
- ml-1m is a separate benchmark, not a superset of latest-small.
- Full EDA reports and figures are in [`eda/`](eda/README.md).
- Popularity concentration is severe in 25M (Gini 0.942); the seeded subset brings
  it to 0.712, comparable to latest-small, while keeping 1.48M ratings.

## Reproduce

```bash
python scripts/download_movielens.py latest-small 1m 25m
python scripts/validate_movielens.py data/raw/ml-25m
python -m ml.eda data/raw/ml-25m
python scripts/build_subset.py data/raw/ml-25m \
  --out data/processed/ml-25m-subset \
  --min-movie-ratings 10 --min-user-ratings 20 \
  --max-users 10000 --seed 42 \
  --manifest-out docs/subsets/ml-25m-subset.json
```
