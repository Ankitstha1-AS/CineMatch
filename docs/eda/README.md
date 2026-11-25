# Exploratory data analysis

Generated with `python -m ml.eda <dataset-dir>`. Every folder contains a
`summary.md` and six PNG figures (rating distribution, user activity, long tail,
Lorenz curve, ratings per year, top genres).

| Dataset | Ratings | Users | Movies | Sparsity | Popularity Gini |
|---|---|---|---|---|---|
| [ml-latest-small](ml-latest-small/summary.md) | 100,836 | 610 | 9,742 | 0.9830 | 0.716 |
| [ml-1m](ml-1m/summary.md) | 1,000,209 | 6,040 | 3,883 | 0.9553 | — |
| [ml-25m](ml-25m/summary.md) | 25,000,095 | 162,541 | 62,423 | 0.9974 | 0.942 |
| [ml-25m-subset](ml-25m-subset/summary.md) | 1,478,848 | 9,960 | 8,737 | 0.9830 | 0.712 |

## Findings

### 1. Popularity is extremely concentrated in the full 25M release

- The top **1%** of rated movies hold **47.6%** of all ratings.
- The top **10%** hold **94.0%**.
- Gini coefficient: **0.942**.

Ranking metrics on the raw release would be dominated by a few blockbusters, and
a laptop cannot hold 25M ratings and the full item-item neighbourhood comfortably.
This motivated a fixed, seeded subset.

### 2. The sampled subset keeps the long tail but tames the bias

After filtering and sampling (see below), the subset has Gini **0.712** and the top
1% of movies hold **15.6%** of ratings — very close to ml-latest-small (0.716 / 15.7%)
while retaining **1.48M** ratings. That is a realistic, laptop-sized training set.

### 3. Sparsity stays high, as expected

The subset is **98.3%** sparse. This is why the evaluation protocol ranks against
**all unrated items**, not a sampled negative set (AGENTS.md section 4).

### 4. Activity distribution supports the cold-start experiment

Median ratings per user in the subset is **70**; p99 is ~1,151; the maximum is 3,157.
Only **2.6%** of retained users have 20 or fewer ratings, so the cold-start experiment
must simulate sparse users by revealing 1, 3, 5, 10, 20, 50 ratings per test user, as
specified — not by relying on the subset composition.

### 5. Tags are usable, but not for every movie

| Release | Tags | Distinct tags | Movie coverage | Genome tags |
|---|---|---|---|---|
| ml-25m | 1,093,360 | 73,050 | 45,251 (72.5%) | 1,128 |
| ml-25m-subset | 60,237 | 12,016 | 5,323 (60.9%) | 1,128 |
| ml-1m | none | — | — | — |

Content features can combine genres (all releases) with tags (latest/25M). The 1M
benchmark is genres-only by design.

### 6. Rating scales differ

ml-latest-small and ml-25M use half-star values (0.5–5.0, mean ≈ 3.54). ml-1m uses
integers (1–5, mean ≈ 3.58). Normalise before comparing metrics across releases.

## Subset decision

Source: **ml-25m** (`ratings.csv` SHA-256 `9c855257…e385`).

| Filter | Value |
|---|---|
| Minimum ratings per movie | 10 |
| Minimum ratings per user | 20 |
| Maximum users | 10,000 (seeded sample, seed 42) |
| Post-sample filters | re-applied until stable |
| Result | **1,478,848 ratings**, 9,960 users, 8,737 movies |

The manifest is committed at [`../subsets/ml-25m-subset.json`](../subsets/ml-25m-subset.json)
and also written next to the processed files (gitignored).

## Reproduce

```bash
python -m ml.eda data/raw/ml-25m
python scripts/build_subset.py data/raw/ml-25m \
  --out data/processed/ml-25m-subset \
  --min-movie-ratings 10 --min-user-ratings 20 \
  --max-users 10000 --seed 42 \
  --manifest-out docs/subsets/ml-25m-subset.json
python -m ml.eda data/processed/ml-25m-subset
```
