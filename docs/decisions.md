# Architecture decisions

Short records of decisions made during the project. Each entry states the context,
the decision, and the consequences.

## ADR-001: Python 3.12 for backend and ML

- **Context:** the development laptop shipped with Python 3.13 only. The stack in
  AGENTS.md targets 3.11/3.12 and pins `scikit-surprise==1.1.5`.
- **Decision:** install Python 3.12.10 and pin `requires-python = ">=3.11,<3.13"`
  for the backend. Use the same interpreter for the ML virtual environment.
- **Consequences:** `pip install` fails fast on 3.13 instead of attempting a
  source build. CI uses 3.12.

## ADR-002: Surprise with a NumPy SVD fallback (M4)

- **Context:** the `scikit-surprise==1.1.5` wheel for CPython 3.12 installs, but
  importing it on the development laptop fails:

  ```text
  ImportError: DLL load failed while importing optimize_baselines:
  An Application Control policy has blocked this file.
  ```

  Windows Smart App Control is in enforcement mode
  (`VerifiedAndReputablePolicyState = 1`) and blocks the unsigned Surprise DLL.
  NumPy, SciPy, scikit-learn and pandas import normally.
- **Decision:** keep `scikit-surprise==1.1.5` as the preferred M4 implementation,
  but ship a pure NumPy/SciPy SVD fallback that is selected automatically when
  Surprise cannot be imported. Do **not** disable Smart App Control as part of
  the project. CI (Linux) exercises the Surprise path.
- **Consequences:** local Windows runs use the fallback; results must record
  which implementation produced each manifest. The fallback also serves as the
  extra comparison model planned in AGENTS.md section 4.

## ADR-003: Native dev servers first, Docker Compose once WSL2 is enabled

- **Context:** Docker Desktop is installed, but WSL2 is not installed and the
  Docker daemon is not running yet.
- **Decision:** develop and verify the backend/frontend natively for Phase 0.
  `docker-compose.yml` is committed but marked unverified until WSL2 is enabled
  and the machine is rebooted.
- **Consequences:** Phase 0's "running skeleton" is demonstrated with local
  `uvicorn` + Vite. Compose validation is a follow-up task.
