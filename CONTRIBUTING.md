# Contributing & Parallel Development Guidelines

This document defines the architectural boundaries, contract-first practices, and Git workflows that allow multiple contributors (or autonomous AI agents) to develop features **in parallel** across **VeriText** without merge conflicts, broken API contracts, or environment collisions.

---

## 1. Module Boundaries & Ownership Scopes

To prevent parallel tasks from colliding in the same files, the codebase is structured into isolated layers. When scoping a task, keep your changes confined to a single scope whenever possible:

| Scope | Files & Directories | Responsibility | Shared Contract / Boundary |
| :--- | :--- | :--- | :--- |
| **`engine` / `nlp`** | `backend/engine.py`<br>`backend/preprocessing.py`<br>`backend/markers.py`<br>`backend/cache.py`<br>`backend/config.py` | Transformer inference (RuBERT / DeBERTa), text normalization, sliding-window chunking, cliché detection, weight caching | Return payload of `ModernAIDetector.score_text()` and `get_confidence_status()` |
| **`parsers` / `reports`** | `backend/parsers.py`<br>`backend/reports.py` | Document text extraction (`.pdf`, `.docx`, `.md`, `.txt`) and Markdown verification report generation | Input payload dictionary (`data: dict`) produced by `/api/score*` and CLI |
| **`api`** | `backend/api/app.py`<br>`backend/api/routes.py`<br>`backend/api/schemas.py`<br>`backend/cancellation.py` | FastAPI HTTP routes, request validation, live progress tracking (`/api/status/{id}`), and task cancellation (`/api/abort`) | Pydantic models in `backend/api/schemas.py` and JSON response payloads consumed by the Web UI |
| **`cli`** | `backend/cli.py`<br>`backend/__main__.py` | `veritext` and `veritext-web` CLI entrypoints, CSV/TSV batch scoring, `rich` progress bars | Command-line flags (`argparse`) and process exit codes |
| **`ui` / `web`** | `web/public/index.html`<br>`web/public/js/app.js`<br>`web/public/css/style.css`<br>`web/nginx.conf` | Single-page Web UI, gauge visualizations, paragraph sorting, progress polling, and request cancellation | REST API contract (`/api/*`) and reverse proxy rules in `web/nginx.conf` |
| **`docker` / `infra`** | `docker/*.Dockerfile`<br>`docker-compose*.yml`<br>`pyproject.toml` | Multi-stage container builds, weight preloading, project dependencies, and packaging | Entrypoints in `pyproject.toml` and environment variables |

---

## 2. Contract-First Development

When a feature spans multiple layers (e.g., **Backend API** + **Web UI**, or **ML Engine** + **Markdown Reports**), decouple the work into two phases:

1. **Lock the Contract First (Fast Synchronous Step):**
   * Define or update the Pydantic request/response schema in `backend/api/schemas.py` and document the expected JSON payload structure.
   * If needed, merge a lightweight stub or schema commit to `main` first.
2. **Execute in Parallel:**
   * **Backend Contributor:** Implements core logic in `backend/engine.py` and wires the endpoint in `backend/api/routes.py`.
   * **Frontend Contributor:** Builds the UI components in `web/public/index.html` and state handlers in `web/public/js/app.js` against the agreed JSON schema.
   * **CLI / Reports Contributor:** Updates `backend/cli.py` and `backend/reports.py` using the agreed dictionary keys.

> **Backward Compatibility Rule:** Never rename or remove existing keys in `/api/score`, `/api/score-document`, `/api/score-dataset`, or `/api/status/{request_id}` responses without updating `web/public/js/app.js` and `backend/reports.py` in the same pull request. Introduce new schema fields as optional (`Optional[...] = None`).

---

## 3. Branching Strategy & Git Workflow

We follow **Trunk-Based Development with short-lived feature branches**:

* `main` must always remain healthy: all unit tests (`pytest`) pass and Docker images build cleanly.
* **One task = One branch = One Pull Request.** Do not bundle unrelated refactoring, UI tweaks, and infrastructure changes into a single branch.
* **Branch Naming Convention:**
  * `feat/<scope>-<short-description>` — new features (e.g., `feat/ui-export-pdf`, `feat/engine-chunk-batching`)
  * `fix/<scope>-<short-description>` — bug fixes (e.g., `fix/parsers-empty-docx`)
  * `perf/<scope>-<short-description>` — performance, latency, or Docker image optimizations
  * `refactor/<scope>-<short-description>` — internal code restructuring with no behavior changes
  * `test/<scope>-<short-description>` — adding or updating automated tests
  * `chore/<scope>-<short-description>` — dependencies, documentation, and build tooling

### Commit Messages (Conventional Commits)

Write all commit messages, code comments, docstrings, and documentation in **English** following the Conventional Commits specification:

```text
<type>(<scope>): <concise summary in imperative mood>
```

Examples:
* `feat(api): add markdown report endpoint for batch datasets`
* `perf(docker): optimize multi-stage builds and layer caching`
* `fix(preprocessing): preserve paragraph breaks after markdown stripping`
* `test(cancellation): cover concurrent request abort edge cases`

### Staying in Sync with `main`

To avoid painful merge conflicts during parallel work:
1. Run `git fetch origin && git rebase origin/main` before opening a PR and at least once a day on active branches.
2. Prefer **Rebase** or **Squash & Merge** to keep the `main` commit history linear.

---

## 4. Local Parallel Workspaces (`git worktree` & Multi-Agent Setup)

When working on multiple tasks simultaneously on a single machine (or running multiple AI coding agents in parallel), **never switch branches inside the same working directory** while servers or tests are running. Use `git worktree` and shared model caches instead.

### 4.1. Isolated Working Trees (`git worktree`)

```bash
# From the root of the main VeriText repository:
git worktree add ../veritext-feat-ui -b feat/ui-heatmap-filter
git worktree add ../veritext-feat-api -b feat/api-batch-streaming
```

Each worktree (`../veritext-feat-ui`, `../veritext-feat-api`) has an independent working directory where editors, test runners, and AI agents can operate concurrently without file locks or index collisions. Once a branch is merged, clean up the worktree:

```bash
git worktree remove ../veritext-feat-ui
```

### 4.2. Shared Hugging Face Weights Cache & Port Isolation

By default, transformer weights (`toiletsandpaper/russian-ai-text-detector-bert` and `mdrakibali/deberta-ai-detector-v3`) are cached in `~/.cache/huggingface/hub`.
* **Native Python environments:** All worktrees and virtual environments automatically share `~/.cache/huggingface/hub` — no duplicate multi-gigabyte downloads occur.
* **Parallel Server Instances:** Assign distinct ports when testing multiple branches locally at the same time:

```bash
# Instance 1 (Docker Compose): Port 8080 (Web) / 8000 (API)
docker compose up -d

# Instance 2 (Native Uvicorn in a parallel worktree): Port 8001
source .venv/bin/activate
uvicorn backend.api.app:app --host 127.0.0.1 --port 8001 --reload
```

---

## 5. Preventing Merge Conflicts in Hotspot Files

Certain files are touched frequently across features. Follow these rules to minimize overlap:

1. **`tests/` — Create Dedicated Test Modules:**
   * Do not append every new test to the bottom of `tests/test_core.py`.
   * Create a focused test file `tests/test_<feature>.py` for new modules or features (following the pattern of `tests/test_cancellation.py`, `tests/test_reports.py`, and `tests/test_cache_and_cli.py`).
2. **`backend/api/routes.py` — Keep Route Handlers Thin:**
   * Place domain logic, data formatting, and calculations in dedicated backend modules (`backend/reports.py`, `backend/parsers.py`, `backend/preprocessing.py`). Route functions should only handle HTTP parsing, `track_cancellation`, and response serialization.
3. **`backend/cli.py` — Isolate Subcommands and Handlers:**
   * Implement new CLI modes as standalone helper functions rather than expanding a monolithic `main()` body.
4. **`web/public/js/app.js` & `web/public/css/style.css` — Avoid Whole-File Reformatting:**
   * Do not run global auto-formatters that alter indentation or quote styles across untouched functions, as this creates whole-file diffs that break parallel rebases.
5. **OS & Runtime Artifacts:**
   * Never commit `.DS_Store`, `__pycache__/`, `.venv/`, or temporary test outputs (`*.csv`, `report*.md`).

---

## 6. Testing & Definition of Done (DoD)

Before merging any branch into `main`, verify the following checklist:

### 1. Fast, Deterministic Unit Tests
Unit tests must run quickly and offline. When testing API routes, CLI flags, report formatting, or cancellation logic, **mock the transformer inference engine** (`unittest.mock.patch`) so tests do not require GPU hardware or live Hugging Face downloads:

```bash
source .venv/bin/activate
pytest -v
```

### 2. Editable Package & CLI Smoke Check
Ensure package metadata and console entrypoints remain valid:

```bash
pip install -e ".[dev]"
veritext --help
```

### 3. Clean Diff & Documentation
* Check `git status` and `git diff` to confirm only task-relevant files are modified.
* Include type annotations (`from __future__ import annotations`) on all new functions and Pydantic models.
* Update `README.md` if your change introduces a new CLI flag, API endpoint, or configuration variable.
