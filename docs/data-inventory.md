# Data inventory

Raw MovieLens files live in `data/raw/` (gitignored). They must never be committed
or redistributed; cite Harper & Konstan (2015).

## Downloaded

Fetched on 2026-11-25 with `scripts/download_movielens.py`.

| Dataset | Archive SHA-256 | Ratings | Users | Movies (total / rated) | Sparsity |
|---|---|---|---|---|---|
| ml-latest-small | `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436` | 100,836 | 610 | 9,742 / 9,724 | 0.9830 |
| ml-1m | `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20` | 1,000,209 | 6,040 | 3,883 / 3,706 | 0.9553 |
| ml-25m | pending | — | — | — | — |

## Validation results

Both datasets passed the first-pass checks in `scripts/validate_movielens.py`:

- no duplicate `(userId, movieId)` pairs;
- no duplicate movie ids;
- no ratings pointing at unknown movies;
- ratings and timestamps parse cleanly.

| Metric | ml-latest-small | ml-1m |
|---|---|---|
| Rating range | 0.5 – 5.0 | 1 – 5 |
| Mean rating | 3.5016 | 3.5816 |
| Median ratings per user | 70.5 | 96 |
| Median ratings per movie | 3 | 123.5 |
| Movies with zero ratings | 18 | 177 |
| Timestamp range (UTC) | 1996-03-29 → 2018-09-24 | 2000-04-25 → 2003-02-28 |
| Tags | 3,683 tags, 1,589 distinct, 1,572 movies | none |

## Notes

- **Rating scales differ.** ml-latest-small uses half-star values (0.5–5.0); ml-1m
  uses integers (1–5). Normalise before comparing metrics across datasets.
- ml-1m is a separate benchmark, not a superset of latest-small.
- Tag data exists only in the latest/25M releases. Content features for ml-1m will
  rely on genres alone, as planned in AGENTS.md.
- The final ~1–2M rating training subset is chosen after the full EDA (long tail,
  tag coverage, tag-genome dimensions).

## Reproduce

```bash
python scripts/download_movielens.py latest-small 1m
python scripts/validate_movielens.py data/raw/ml-latest-small
python scripts/validate_movielens.py data/raw/ml-1m
```
