# CineMatch

**Explainable Hybrid Movie Recommendation System with Cold-Start Handling**

[![CI](https://github.com/Ankitstha1-AS/CineMatch/actions/workflows/ci.yml/badge.svg)](https://github.com/Ankitstha1-AS/CineMatch/actions/workflows/ci.yml)

CineMatch is a final-year project (BIT Honours, Lincoln University College, Malaysia).
It learns a viewer's taste from star ratings and returns movie recommendations that
combine content-based and collaborative filtering — and it explains every suggestion
in plain language.

> **Status: Phase 1 — Data and schema.**
> Phase 0 delivered the repository skeleton, health-check API, Docker Compose stack
> and CI. Phase 1 adds the validated MovieLens data pipeline (download → EDA →
> seeded subset) and the PostgreSQL schema with ORM models, an Alembic migration
> and a catalogue importer. The recommendation engine and UI follow in later phases.

## Why it exists

1. Popularity lists ignore personal taste.
2. Single-technique recommenders fail for brand-new users (cold start).
3. Most systems never say *why* they recommended something.
4. Rating-error metrics alone do not capture top-N list quality.

The proposed model blends a content score with a collaborative score using an adaptive
weight that shifts towards content for new users and towards collaborative signals as a
user rates more films.

## Tech stack

| Layer | Choice |
|---|---|
| API | Python 3.12, FastAPI, Pydantic v2 |
| Frontend | React 18, Vite, TypeScript |
| Database | PostgreSQL 16 (SQLAlchemy 2.0 + Alembic) |
| ML | scikit-learn, scikit-surprise, pandas, NumPy, SciPy, joblib |
| Security (Phase 4) | JWT access/refresh, argon2id, rate limiting, RBAC |
| Testing | pytest + httpx + coverage; Vitest + React Testing Library |
| DevOps | Docker Compose, GitHub Actions, Nginx |
| External | TMDB API (posters/overviews, display only) |

## Repository layout

```text
.
├── AGENTS.md                 # authoritative engineering guide
├── docker-compose.yml        # Postgres + API + frontend
├── .env.example              # copy to .env
├── .github/workflows/ci.yml  # lint + tests + build
├── backend/
│   ├── app/
│   │   ├── api/v1/           # routers (health now; auth/catalogue later)
│   │   ├── core/             # settings, security, dependencies
│   │   ├── db/               # declarative base, session factory, portable types
│   │   ├── services/         # business logic (catalogue importer)
│   │   ├── repositories/     # SQLAlchemy access
│   │   ├── models/           # ORM models (users, movies, ratings, ...)
│   │   ├── schemas/          # Pydantic schemas
│   │   └── inference/        # TF-IDF, SVD, item-KNN, blender, explainer
│   ├── alembic/              # migration environment and versions
│   ├── alembic.ini
│   ├── tests/
│   ├── pyproject.toml
│   └── Dockerfile
├── frontend/
│   ├── src/                  # pages, components, typed API client
│   ├── package.json
│   └── Dockerfile
├── ml/
│   ├── pipeline/             # ingest → clean → features → split → train → export
│   ├── evaluation/           # metrics, cold-start, alpha sweep, significance
│   ├── notebooks/            # EDA only
│   └── requirements.txt
├── docs/                     # SRS, design, AI report, test report
└── scripts/                  # data import and helpers
```

## Prerequisites

- **Python 3.12** — required. The pinned `scikit-surprise==1.1.5` has no Python 3.13
  wheel, so 3.13 will try (and usually fail) to compile from source.
- **Node.js 20+** (Node 22 is what CI uses).
- **Docker Desktop** — optional for local dev, required for the full Compose stack.
- **Git**.

## Getting started

### 1. Configuration

```bash
cp .env.example .env
```

Edit `.env` if your ports or database credentials differ. Never commit `.env`.

### 2. Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -e ".[dev]"
alembic upgrade head          # needs a reachable PostgreSQL (see Database)
uvicorn app.main:app --reload
```

The API is at <http://localhost:8000>, interactive docs at
<http://localhost:8000/docs>, and the health check at
<http://localhost:8000/api/v1/health>.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to the backend on port 8000.

### 4. Full stack with Docker

```bash
docker compose up --build
```

- Frontend: <http://localhost:8080>
- API: <http://localhost:8000>
- PostgreSQL: `localhost:5432`

Create the schema inside the Compose network:

```bash
docker compose run --rm api alembic upgrade head
```

Then import the catalogue from the host (the database port is published):

```bash
python scripts/import_movielens.py data/processed/ml-25m-subset
```

## Database

The schema has six tables — `users`, `movies`, `ratings`, `feedback`,
`model_versions` and `usage_logs` — created by Alembic from the SQLAlchemy models.
See [`docs/database.md`](docs/database.md) for the ERD and design rules.

```bash
cd backend
alembic upgrade head            # create/upgrade the schema
alembic upgrade head --sql      # print the PostgreSQL DDL without connecting
```

Load the MovieLens catalogue (idempotent; imports movies, genres, tags and TMDB ids):

```bash
python scripts/import_movielens.py data/processed/ml-25m-subset
```

MovieLens **ratings** are not imported: they are training data for the offline
pipeline. The `ratings` table holds ratings from real CineMatch accounts.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service index |
| GET | `/api/v1/health` | Liveness/version check |
| GET | `/docs` | Swagger UI |
| GET | `/openapi.json` | OpenAPI schema |

## Tests and linting

```bash
# Backend
cd backend
pytest                     # includes coverage; fails below 70%
ruff check .
ruff format --check .

# Frontend
cd frontend
npm run lint
npm run test
npm run build
```

CI runs the same commands on every push and pull request.

## ML pipeline

```bash
cd ml
python -m venv .venv
.venv\Scripts\Activate.ps1     # or: source .venv/bin/activate
pip install -r requirements.txt

cd ..
python -m ml.pipeline.run --config ml/config.yaml
```

### Data preparation

```bash
# Download and validate MovieLens releases (raw files are gitignored)
python scripts/download_movielens.py latest-small 1m 25m
python scripts/validate_movielens.py data/raw/ml-25m

# Exploratory analysis -> docs/eda/<dataset>/
python -m ml.eda data/raw/ml-25m

# Fixed, seeded, laptop-sized training subset -> data/processed/
python scripts/build_subset.py data/raw/ml-25m \
  --out data/processed/ml-25m-subset \
  --min-movie-ratings 10 --min-user-ratings 20 \
  --max-users 10000 --seed 42 \
  --manifest-out docs/subsets/ml-25m-subset.json
```

Current subset: **1,478,848 ratings**, 9,960 users, 8,737 movies. Findings and
figures are in [`docs/eda/`](docs/eda/README.md); the model pipeline itself is
implemented in Phases 2–3. Raw MovieLens data is never committed or redistributed.

## Design summary

The hybrid score is:

```text
S(u, i) = alpha(n_u) * C(u, i) + (1 - alpha(n_u)) * R(u, i)
```

- `C(u, i)` — cosine similarity between a movie's TF-IDF vector and the user's taste vector.
- `R(u, i)` — normalised collaborative score from SVD and item-KNN.
- `alpha(n_u)` — decays as the user's rating count grows, and is tuned on validation data only.

Every recommendation carries a content reason and/or a collaborative reason derived from
the actual signals, never a canned template. See [`AGENTS.md`](AGENTS.md) section 4 for the
full ML design and evaluation protocol.

## Documentation

Project documents are collected under `docs/`. The authoritative engineering guide,
including the 18-week plan, Definition of Done and working rules, is
[`AGENTS.md`](AGENTS.md).

## Attribution

- MovieLens data: GroupLens Research. Cite Harper, F. M., & Konstan, J. A. (2015).
  The MovieLens Datasets: History and Context. *ACM TiiS, 5*(4), 1–19.
- This product uses the TMDB API but is not endorsed or certified by TMDB.

## Licence

Academic, non-commercial use only.
