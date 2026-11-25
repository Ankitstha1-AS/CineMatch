# AGENTS.md: CineMatch

> Guide for AI coding agents and human contributors working on this repo.
> Read this first. It tells you what the project is, how it is built, what the rules are, and what to build next.

---

## 1. Project Snapshot

| Item | Value |
|---|---|
| **Name** | CineMatch |
| **Title** | Explainable Hybrid Movie Recommendation System with Cold-Start Handling |
| **Type** | Final Year Project (BIT Honours, Lincoln University College, Malaysia) |
| **Team** | Single developer (individual project) |
| **Goal** | Web app that learns taste from ratings, recommends movies with a hybrid of content-based and collaborative filtering, and explains every recommendation in plain language |
| **Use** | Non-commercial, academic |

**One-line analogy:** CineMatch is a film-savvy friend. A new friend asks "what genres do you like?" (cold start). A close friend knows your taste (collaborative). It always tells you *why* it suggests something (explainability).

### Core problems it solves
1. Generic popularity lists ignore personal taste.
2. Single-technique recommenders fail for new users (cold start).
3. Most systems give no reason for a suggestion.
4. Rating-error metrics alone do not reflect top-N list quality.

### Research questions
- **RQ1**: Does the hybrid beat each single model on rating-prediction and ranking metrics?
- **RQ2**: How does quality change with the number of ratings a user has given, and does the adaptive weight help new users?
- **RQ3**: Do users find explanations helpful, and does the app reach SUS >= 68?

---

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Backend API | Python 3.11/3.12, **FastAPI**, Pydantic |
| Frontend | **React + Vite + TypeScript** |
| Database | **PostgreSQL** (full-text search, JSONB) |
| Data access | SQLAlchemy 2.0 + Alembic migrations |
| ML | scikit-learn, **Surprise (pin `scikit-surprise==1.1.5`)**, pandas, NumPy, SciPy (sparse), joblib |
| Security | JWT (access + refresh), argon2-cffi (argon2id), rate limiting |
| Cache (optional) | Redis |
| Testing | pytest, httpx, coverage; Vitest + React Testing Library; Locust or k6 |
| DevOps | Docker, Docker Compose, GitHub Actions, Nginx |
| External | TMDB API (posters/overviews, **display only**) |
| Lint/format | Ruff, ESLint |

**Do not introduce** no-code/low-code tools, external LLM/AI APIs as the "AI", or other stacks without asking. The FYP guidelines forbid them.

---

## 3. Architecture

**Style:** layered modular monolith with an offline training pipeline and a versioned model registry (online/offline separation).

```
CLIENT TIER      React SPA (onboarding, browse, rate, recs, admin dashboard)
                      | HTTPS / JSON
                 Nginx reverse proxy (TLS, static, gzip)
                      |
APPLICATION      FastAPI service (one Docker container)
TIER             - REST routers  /api/v1   (Pydantic validation, OpenAPI docs)
                 - Security layer (JWT, argon2id, rate limit, RBAC)
                 - Service layer (Catalogue, Ratings, Recommendation, Admin)
                 - Inference engine (TF-IDF, SVD, item-KNN, blender, explainer)
                 - Repository layer (SQLAlchemy + Alembic)
                      |  load at startup / hot-reload
DATA TIER        Model artifact store (joblib + manifest.json)
                 PostgreSQL (users, movies, ratings, feedback, model_versions, metrics, usage_logs)
                      ^  best model + manifest     ^ ratings snapshot, metrics
OFFLINE ML TIER Training pipeline (admin- or schedule-triggered)
                 ingest -> clean -> features -> split -> train & tune
                 Evaluation module (RMSE, MAE, P/R/NDCG@K, cold-start, alpha sweep)
                      ^
                 MovieLens dataset (one-time batch import)
```

### Golden rules of the architecture
- **Training never runs inside a user request.** It is an offline job that writes versioned artifacts.
- The API **loads the active model into memory once** at startup and can hot-reload after admin activation.
- Every unseen movie is scored directly in memory (catalogue is small). No separate candidate-retrieval stage.
- Layers talk downward only: router -> service -> repository / inference engine. Routers never touch the DB directly.
- The API is stateless so it can run as several replicas.

### Data stores
| ID | Store | Contents |
|---|---|---|
| D1 | Users & Profiles | id, email, password hash, role, favourite genres, consent flag |
| D2 | Movies & Metadata | id, title, year, genres, tags, TMDB id, cached display metadata |
| D3 | Ratings & Feedback | ratings, reviews, thumbs up/down events (timestamped) |
| D4 | Model Registry | versioned artifacts, manifests, active-version flag |
| D5 | Usage Logs | requests, latency, feature usage |

### Main processes (DFD Level 1)
1.0 Manage Accounts and Profiles | 2.0 Browse and Search Catalogue | 3.0 Capture Ratings and Feedback | 4.0 Generate Recommendations | 5.0 Train and Evaluate Models (offline) | 6.0 Administer and Monitor

Process 4.0 splits into: 4.1 Resolve User State, 4.2 Score by Content, 4.3 Score by CF, 4.4 Blend and Rank, 4.5 Generate Explanations, 4.6 Apply Live Feedback.

---

## 4. ML Design

### Hybrid score
```
S(u, i) = alpha(n_u) * C(u, i) + (1 - alpha(n_u)) * R(u, i)
```
- `C(u,i)`: cosine similarity between movie TF-IDF vector (genres + tags) and the user taste vector (built from highly rated movies).
- `R(u,i)`: collaborative score from SVD and item-KNN, **normalised to the same range** as C.
- `n_u`: number of ratings by user u. `alpha` decays from `alpha_max` to `alpha_min` as `n_u` grows. `alpha_max`, `alpha_min` and the decay rate are **tuned on the validation set only**.
- Zero-rating users get **genre-matched, popularity-weighted** results (popularity prior).
- Top-N selection uses a **heap**. Already-rated movies are removed.

### Explanations (must come from the real signals, not templates)
- Content reason: top-rated movies the user saw + shared tags/genres with the recommendation.
- Collaborative reason: neighbour movies that drove the item-KNN score.
- Examples: "Because you rated X highly", "Viewers with similar taste liked this".

### Live feedback
Thumbs up/down shifts the **session taste vector** and removes disliked items at once. The event is stored and used in the next retraining. No real-time online learning.

### Models to compare
| ID | Model | Role |
|---|---|---|
| M0 | Global mean + user/item biases | RMSE/MAE lower bound |
| M1 | Popularity (weighted rating) | Ranking baseline, cold-start prior |
| M2 | TF-IDF + cosine | Content-based; similar movies; content reasons |
| M3 | Item-based KNN | Collaborative; interpretable neighbours |
| M4 | SVD matrix factorisation | Collaborative; strong rating prediction |
| M5 | Adaptive hybrid (M2 + M3 + M4) | **Proposed model** |

Fallback if Surprise fails to install: custom SVD in NumPy/PyTorch (also planned as an extra comparison model).

### Evaluation protocol (do not change without noting it in the report)
- **Split:** per-user temporal hold-out (latest 20% of each user's ratings = test; slice of train = validation). Fixed random seeds.
- **Tuning:** validation data only. **Test set is used once.**
- **Rating metrics:** RMSE, MAE.
- **Ranking metrics:** Precision@K, Recall@K, NDCG@K for K = 5, 10, 20. Rating >= 4.0 = relevant. Rank against **all unrated items**, not a sampled subset.
- **Secondary:** coverage, intra-list diversity, novelty.
- **Cold-start experiment:** reveal 1, 3, 5, 10, 20, 50 ratings per test user; plot metrics for M1 to M5.
- **Alpha sweep:** 0 to 1 in 0.1 steps; compare with adaptive schedule.
- **Significance:** paired Wilcoxon signed-rank + bootstrap confidence intervals over users.
- **User study:** >= 20 participants, anonymous, consented; SUS + relevance + explanation helpfulness (5-point). Report **separately** from offline metrics.

### Data
| Dataset | Role |
|---|---|
| MovieLens 25M (reduced subset, ~1 to 2M ratings) | **Primary** (has tags + tag-genome) |
| MovieLens 1M | Secondary benchmark (genres only) |
| MovieLens latest-small | Development and unit tests only |
| TMDB API | Posters/overviews, display only |

- Subset filters (min ratings per movie, sampled users) are fixed **after EDA**, sized to fit a standard laptop.
- Every model manifest stores: parameters, seeds, metrics, **dataset version + hash**.
- **Never redistribute raw MovieLens data.** Cite Harper & Konstan (2015).
- **Train on MovieLens only.** TMDB content is display-only, cached <= 6 months, and must show: *"This product uses the TMDB API but is not endorsed or certified by TMDB."*

---

## 5. Functional Scope (priority: M = must, S = should, C = could)

| ID | Feature | Pri |
|---|---|---|
| FR-01 | Register/login/logout, JWT access + refresh | M |
| FR-02 | Profile: display name, favourite genres, rating history | M |
| FR-03 | Cold-start onboarding: pick >= 3 genres, rate seed movies | M |
| FR-04 | Catalogue browse: pagination, genre/year filters, sort | M |
| FR-05 | Full-text search by title/keyword | M |
| FR-06 | Movie detail: metadata, poster, avg rating, reviews | M |
| FR-07 | Rate 0.5 to 5.0 (half-star); edit/remove | M |
| FR-08 | Write/edit/delete own review | S |
| FR-09 | Personalised top-N from hybrid engine | M |
| FR-10 | "Why recommended" on every item | M |
| FR-11 | Thumbs up/down re-ranks immediately | S |
| FR-12 | "Similar movies" on detail page | S |
| FR-13 | Admin dashboard: metric tables/charts, usage stats | M |
| FR-14 | Admin: retrain, activate version, rollback | S |
| FR-15 | Admin: manage catalogue and users | C |
| FR-16 | User exports/deletes own data | C |
| FR-17 | Auto-generated OpenAPI docs | S |

**Out of scope:** video streaming, real-time online learning, native mobile apps, social/friend graph, commercial operation, non-English UI.

### Non-functional targets (design goals, validate in testing)
| Area | Target |
|---|---|
| Performance | Top-10 endpoint p95 <= 500 ms at 50 concurrent users |
| Security | argon2id/bcrypt, short-lived JWT, HTTPS, Pydantic validation, parameterised queries, auth rate limiting, RBAC, OWASP Top 10 review |
| Usability | SUS >= 68, responsive, keyboard nav, WCAG 2.1 AA basics |
| Maintainability | Type hints, Ruff/ESLint, >= 70% backend unit-test coverage, CI on every push |
| Reproducibility | Fixed seeds, pinned dependencies, dataset hash in manifest |
| Portability | Docker Compose; config via environment variables |
| Privacy | Minimum personal data, account deletion, anonymous consented study |

---

## 6. Suggested Repo Layout

> Proposed structure. Adjust if the real repo differs, then update this section.

```
cinematch/
├── AGENTS.md
├── README.md
├── docker-compose.yml
├── .github/workflows/ci.yml
├── backend/
│   ├── app/
│   │   ├── api/v1/          # routers
│   │   ├── core/            # config, security, deps
│   │   ├── services/        # catalogue, ratings, recommendation, admin
│   │   ├── repositories/    # SQLAlchemy access
│   │   ├── models/          # ORM models
│   │   ├── schemas/         # Pydantic schemas
│   │   └── inference/       # tfidf, svd, knn, blender, explainer
│   ├── alembic/
│   └── tests/
├── ml/
│   ├── pipeline/            # ingest, clean, features, split, train, export
│   ├── evaluation/          # metrics, cold-start, alpha sweep, significance
│   ├── notebooks/           # EDA only
│   └── artifacts/           # versioned model.joblib + manifest.json (gitignored if large)
├── frontend/
│   └── src/                 # pages, components, api client, hooks
├── docs/                    # SRS, design, ERD, UML, AI report, test report
└── scripts/                 # data import, seed, helpers
```

---

## 7. Working Rules for Agents

### Do
- Keep **training code out of request handlers**.
- Use **type hints** everywhere (Python) and strict TypeScript.
- Validate all input with Pydantic; use ORM parameterised queries only.
- Put config in **environment variables**. Never hard-code secrets or API keys.
- Fix random seeds; log parameters + dataset hash in every manifest.
- Write or update tests with every feature (pytest / Vitest).
- Add Alembic migrations for any schema change.
- Update docs (README, this file) when architecture or commands change.
- Keep changes small and focused; one concern per commit.

### Don't
- Don't tune on the test set. Don't peek at it.
- Don't train any model on TMDB data.
- Don't commit raw MovieLens files, `.env`, secrets, or large artifacts.
- Don't add heavy frameworks (Kubernetes, microservices, PyTorch) without discussion.
- Don't replace the hybrid with an external AI API.
- Don't "fix" metrics by sampling negatives. Rank against all unrated items.
- Don't mark requirements as done without a test or evidence.

### Git rules (FYP guideline, mandatory)
- GitHub from **day 1**; **supervisor invited as collaborator**.
- **3 to 5 meaningful commits per week.** No last-minute dumps.
- Branches: `main` (stable), `dev` (integration), `feature/<name>` (e.g. `feature/model-training`, `feature/login-api`).
- Conventional commits:
  - `feat: add user authentication API`
  - `fix: resolve model prediction accuracy issue`
  - `refactor: optimize database queries`
  - `docs: update AI training process`
  - `test:`, `chore:`, `ci:` as needed
- Pull requests encouraged.
- README must include: overview, tech stack, setup, database schema, AI model details, screenshots, deployment guide.

### Code style
- Python: Ruff (lint + format), `snake_case`, docstrings on public functions.
- TypeScript: ESLint, `camelCase`, `PascalCase` components, typed API client.
- Keep functions short; prefer pure functions in the ML modules for testability.

---

## 8. Commands

> Fill in or correct once the repo exists.

```bash
# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # includes scikit-surprise==1.1.5
alembic upgrade head
uvicorn app.main:app --reload            # API + docs at /docs
pytest --cov=app

# ML pipeline
python -m ml.pipeline.run --config ml/config.yaml     # train + evaluate + export
python -m ml.evaluation.report                        # metrics tables/charts

# Frontend
cd frontend && npm install
npm run dev
npm run test
npm run lint

# Whole stack
docker compose up --build

# Load test
locust -f backend/tests/load/locustfile.py
```

---

## 9. Phase-Wise Distribution

Total plan: **~18 weeks** (adjust to the real academic calendar). Phases overlap where noted.

### Phase 0: Foundation (Week 1)
- Create GitHub repo, branches (`main`, `dev`), invite supervisor.
- Project skeleton, Docker Compose (API, Postgres, frontend), `.env.example`.
- **Test clean install of `scikit-surprise==1.1.5`** in a fresh venv (decide on custom-SVD fallback now).
- CI: GitHub Actions running lint + pytest on every push.
- **Deliverable:** green CI, running skeleton, first commits.

### Phase 1: Data and EDA (Weeks 2 to 3)
- Download MovieLens 25M, 1M, latest-small. Validate schema, duplicates.
- EDA: sparsity, rating distribution, long tail, tag coverage; measure tag-genome dimensions.
- Fix subset filters (min ratings, sampled users) to ~1 to 2M ratings.
- Design ERD and DB schema; first Alembic migrations; import script.
- **Deliverable:** cleaned subset, EDA notebook, ERD, DB populated.

### Phase 2: ML Core and Evaluation Harness (Weeks 3 to 6)
- Per-user temporal split + validation slice, fixed seeds.
- Build evaluation module first (RMSE, MAE, P/R/NDCG@K, coverage, diversity, novelty).
- Implement M0 and M1 baselines, M2 content, M3 item-KNN, M4 SVD.
- Hyper-parameter tuning on validation only. Model manifest + joblib export.
- **Deliverable:** results table for M0 to M4, reproducible run script.

### Phase 3: Hybrid, Cold-Start and Explanations (Weeks 6 to 8)
- Score normalisation + adaptive alpha schedule; tune `alpha_max`, `alpha_min`, decay.
- Alpha sweep (0 to 1) and cold-start experiment (1 to 50 ratings).
- Popularity-prior path for zero-rating users.
- Explanation generator (shared tags, neighbour movies).
- Significance tests (Wilcoxon, bootstrap CI).
- **Deliverable:** M5 results, RQ1 and RQ2 evidence, explainer module with unit tests.

### Phase 4: Backend API (Weeks 4 to 9, parallel with Phases 2 to 3)
- Auth (FR-01) with argon2id, JWT access/refresh, rate limiting, RBAC.
- Catalogue, search, detail, ratings, reviews (FR-02 to FR-08).
- Recommendation endpoint serving the in-memory model (FR-09, FR-10, FR-12).
- TMDB enrichment with caching (<= 6 months) and attribution.
- OpenAPI docs (FR-17), health-check endpoint, usage logging.
- **Deliverable:** documented API with integration tests, >= 70% coverage trend.

### Phase 5: Frontend (Weeks 7 to 11)
- Auth pages, onboarding (genre pick + seed ratings), browse/search, detail page.
- Rating widget (half-star), review UI, recommendation list with "why" text.
- Thumbs up/down with instant re-rank (FR-11).
- Responsive layout, keyboard navigation, contrast (WCAG 2.1 AA basics).
- **Deliverable:** usable end-to-end user flow against the real API.

### Phase 6: Admin, Model Registry and Live Feedback (Weeks 10 to 12)
- Model registry (D4): versions, manifests, activate, rollback (FR-14).
- Retrain trigger runs the offline pipeline as a separate job.
- Admin dashboard: metric tables/charts, usage stats (FR-13).
- Session taste-vector update + stored feedback events.
- Optional: catalogue/user management (FR-15), data export/delete (FR-16).
- **Deliverable:** admin can retrain, compare, activate and roll back a model.

### Phase 7: Testing, Security and Performance (Weeks 12 to 14)
- Backend unit + integration tests, frontend component tests.
- Security review against OWASP Top 10; dependency audit.
- Locust/k6 load test: p95 <= 500 ms at 50 concurrent users; tune (compact float32 artifacts, stored top-k neighbours, optional Redis).
- Write test cases table and test report.
- **Deliverable:** test report with coverage + load-test results.

### Phase 8: Deployment and User Study (Weeks 14 to 16)
- Deploy: cloud VM (DigitalOcean/AWS EC2) for API + DB, Netlify/Vercel for frontend, HTTPS.
- Scheduled DB backups, environment config, deployment documentation.
- Run user study (>= 20 participants): consent notice, anonymous SUS + relevance + explanation helpfulness.
- **Deliverable:** live URL, study results (RQ3).

### Phase 9: Documentation and Submission (runs throughout; final push Weeks 16 to 18)
- **Continuous:** keep README current; commit docs alongside code.
- Documents required: Proposal (done), SRS, System Design (UML, ERD, architecture), AI Training and Evaluation Report, Test Report, Final Report (APA), repo link, deployment doc, slides.
- Final report follows the 7-chapter template (Intro, Literature Review, Methodology, System Design, Implementation, Testing and Results, Conclusion).
- Formatting: Times New Roman 12 pt, 1.5 spacing, left margin 1.5", others 1", Roman numerals for preliminary pages, APA references.
- Add items not yet covered: feasibility study, use-case, class, sequence/activity diagrams, UI/UX design, limitations, future work.
- **Deliverable:** complete submission pack and presentation.

### Timeline at a glance
```
Week:        1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18
P0 Foundation ██
P1 Data/EDA      ██ ██
P2 ML core          ██ ██ ██ ██
P3 Hybrid                  ██ ██ ██
P4 Backend            ██ ██ ██ ██ ██ ██
P5 Frontend                     ██ ██ ██ ██ ██
P6 Admin/Registry                        ██ ██ ██
P7 Testing                                  ██ ██ ██
P8 Deploy/Study                                   ██ ██ ██
P9 Docs        ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ░░ ██ ██ ██ ██ ██
```

### Cut-line if time runs short (drop in this order)
1. FR-15 catalogue/user management and FR-16 data export/delete (Could).
2. Optional Redis cache and k-means movie clustering.
3. Secondary metrics (diversity, novelty) and extra comparison models.
4. FR-14 one-click retrain from the UI (keep CLI retrain + registry).

**Never cut:** hybrid engine, cold-start onboarding, explanations, evaluation protocol, deployment, tests, GitHub activity, documentation.

---

## 10. Definition of Done

A task is done when:
- [ ] It meets the linked requirement (FR/NFR ID).
- [ ] Tests exist and pass in CI; coverage does not drop.
- [ ] Lint and type checks pass (Ruff, ESLint, TypeScript).
- [ ] No secrets, raw datasets, or large artifacts committed.
- [ ] Docs/README/this file updated if behaviour or commands changed.
- [ ] For ML changes: seeds fixed, manifest written, metrics logged, test set untouched.
- [ ] Committed with a conventional message on a feature branch.

---

## 11. Future Goals (post-FYP / stretch)

**Recommendation quality**
- Implicit feedback signals (clicks, watch time) with ALS/BPR.
- Neural collaborative filtering or two-tower models; sequence models (GRU4Rec / SASRec) for session-aware recommendations.
- Sentence-embedding content features (movie overviews) alongside TF-IDF.
- Diversity/novelty-aware re-ranking (MMR) and fairness checks.
- Contextual bandits for exploration vs exploitation.

**Explainability**
- Counterfactual explanations ("recommended because of X; without X it drops").
- Per-user control: tweak genre weights and see the list change.
- Controlled study comparing explanation styles.

**Platform**
- Real-time or incremental online learning.
- Redis-backed caching and a dedicated model-serving service.
- A/B testing framework and richer admin analytics.
- Candidate-retrieval stage (ANN index) if the catalogue grows large.
- Monitoring and drift detection (data and model).

**Product**
- Native mobile apps or PWA, multilingual UI.
- Watchlists, social/friend features, group recommendations.
- Streaming-availability integration (subject to licensing).

**Research**
- Publish the cold-start and adaptive-weight findings (workshop/short paper).
- Release reproducible evaluation scripts (without redistributing MovieLens).

---

## 12. Open Items (resolve early)

- [ ] Fill cover-page details: surname, LCID, supervisor.
- [ ] Decide whether to actually implement k-means clustering or drop the claim.
- [ ] Confirm TMDB terms on ML use (full page not verified; models stay MovieLens-only).
- [ ] Verify citation page numbers/DOIs before final submission.
- [ ] Confirm final subset size after EDA.
- [ ] Confirm deployment target and free-tier memory limits.

---

## 13. Quick Reference

- **Allowed by guidelines:** FastAPI, React, PostgreSQL, Docker, Redis (optional), cloud VM, Netlify/Vercel.
- **Rejected by guidelines:** simple CRUD, basic login apps, no-code/low-code, template apps, projects without real AI/ML (an external API alone does not count), last-minute code pushes, copy-paste GitHub projects.
- **Algorithms to showcase:** TF-IDF + cosine, KNN, SVD, hybrid recommender, heap-based top-N, argon2id hashing (k-means optional).
- **Attribution:** MovieLens (GroupLens, Harper & Konstan 2015) and the TMDB notice must be visible/cited.
