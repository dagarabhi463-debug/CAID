# CAID Phase 4.0 — Release Blockers & Findings

Date: 2026-11-12
Classification: Critical / High / Medium / Low / Informational.
All findings verified against the live repository; nothing speculative is listed as confirmed.

---

## CRITICAL

### C1 — Entire Milestone 2.4–3.2 work is uncommitted

- **Evidence:** `git status` — 31 modified tracked files, ~70 untracked paths including all of `core/streaming/` (10 modules), `app_pages/{streaming_health,detection_quality,reports}.py`, `core/{artifact_lifecycle,drift_monitor}.py`, 12 new test files, `scripts/` (6 files), all new model artifacts, `docs/`, and `data/incidents/incidents.json` modifications. HEAD is `1744782 "Remove unused files"` — i.e., everything since the cleanup snapshot is working-tree only.
- **Affected:** whole repository.
- **Impact:** a single `git checkout .`, `git clean -fdx`, disk failure, or OneDrive sync conflict could destroy the entire release candidate. There is no recoverable checkpoint.
- **Reproduction:** `git status` — observe the uncommitted set.
- **Recommended correction:** before any further work, create a checkpoint commit on `master` (or a `release/v1.0-rc` branch + tag) after deciding which generated artifacts to track (see M3). Do **not** `git add -A` blindly — `.env` and `nul` must be excluded first (H1, L1).
- **Regression test:** n/a (process control). Verify with `git status` clean and `git log` showing the checkpoint.

---

## HIGH

### H1 — `.env` not protected by `.gitignore`

- **Evidence:** `.gitignore` contains only `.venv/`, `__pycache__/`, `.pytest_cache/`, `.streamlit/secrets.toml`. `git check-ignore .env` → not ignored. `.env` exists untracked at repo root.
- **Affected:** `.gitignore`, `.env`.
- **Impact:** `git add -A` / IDE "stage all" commits `.env`. Current contents are non-secret localhost config, but the pattern invites committing real secrets later (Ollama/Redis URLs could carry credentials in other deployments).
- **Reproduction:** `git check-ignore .env; echo $?` → exit 1.
- **Recommended correction:** add `.env` and `.env.*` (with `!.env.example`) to `.gitignore` **before** the checkpoint commit.
- **Regression test:** `git check-ignore .env` returns 0; `git status` no longer lists `.env`.

---

## MEDIUM

### M1 — `.env` drift + `.env` is never loaded by the application

- **Evidence:** `.env` lacks `REDIS_URL` present in `.env.example`; `diff .env .env.example` shows only that difference. No `dotenv` import anywhere; `python-dotenv` absent from requirements; all consumers use `os.environ.get` with localhost defaults.
- **Affected:** `.env`, `.env.example`, `core/ollama_client.py:47`, `core/streaming/event_queue.py:495`, `app_pages/settings.py:62-75`.
- **Impact:** operators may believe `.env` configures the app when it does not; conversely a future `load_dotenv()` addition would silently change behavior. Documentation/config drift.
- **Reproduction:** `diff .env .env.example`; `grep -rn dotenv --include='*.py' .` → no matches.
- **Recommended correction (minimal):** add the missing `REDIS_URL` line to `.env`; add one line to README stating env vars must be exported (or `.env` sourced) — the app does not auto-load `.env`. Optionally add `python-dotenv` + `load_dotenv()` as a documented enhancement post-RC.
- **Regression test:** none required for doc fix; if dotenv is adopted later, add a test asserting env override works.

### M2 — Generated benchmark/runtime artifacts untracked and un-ignored

- **Evidence:** 27 `reports/stream_benchmark_*.json`, 29 `data/stream_results/*.jsonl`, 29 `data/stream_dlq/*.jsonl` — all untracked, none ignored (`git check-ignore` exit 1). `data/incidents/incidents.json` (3.1 MB) is tracked and grows on every dashboard analysis.
- **Affected:** `reports/`, `data/stream_results/`, `data/stream_dlq/`, `data/incidents/`.
- **Impact:** repo bloat and noisy diffs; risk of committing transient runtime output into the release tag; conversely DLQ/result files are evidence for the benchmark claims — decide deliberately.
- **Reproduction:** `git status --short reports data/stream_dlq data/stream_results`.
- **Recommended correction:** for the RC, commit the *latest* benchmark report(s) referenced by docs as evidence (or a curated subset), and add ignore rules for `data/stream_dlq/`, `data/stream_results/`, `data/stream_checkpoint/` (runtime-generated). Keep `data/incidents/incidents.json` tracked as the persistent incident store but document its growth, or snapshot it.
- **Regression test:** streaming tests already cover DLQ/result writing; add `.gitignore` entries and verify `git status` stays clean after a benchmark run.

### M3 — `nul` junk file at repository root

- **Evidence:** `nul` (103 bytes) untracked, created by a Windows shell redirect artifact.
- **Affected:** `nul`.
- **Impact:** would be committed by `git add -A`; on Windows `nul` is a reserved device name — the file can cause checkout/clone problems on Windows and confuse tooling.
- **Reproduction:** `ls -la nul; cat nul`.
- **Recommended correction:** inspect contents, then delete the file (it is not source data; verified it is not referenced anywhere — `grep -rn "nul" --include='*.py'` shows no imports). Deletion requires explicit confirmation per destructive-op policy.
- **Regression test:** n/a.

---

## LOW

### L1 — Dead module `modules/dashboard.py`

- **Evidence:** AST/import scan — no module imports `modules.dashboard`; only a docstring in `tests/test_system_reliability.py:252` references it. Prior audit (`complete_dashboard_inventory.md`) already documented it as dead.
- **Impact:** maintenance confusion only; zero runtime effect.
- **Recommended correction:** may be removed in a post-RC cleanup; keeping it is also acceptable. Do not remove in the RC commit without explicit sign-off.
- **Regression test:** full suite already guards column contracts (`test_system_reliability.py`).

### L2 — Pytest deprecation warnings (4)

- **Evidence:** `PytestRemovedIn10Warning` on class-scoped fixtures used as instance methods in 4 test classes (listed in baseline doc).
- **Impact:** warnings only; will break under a future pytest 10 major. Not a release blocker for pytest 9.x.
- **Recommended correction:** post-RC, convert fixtures to `@classmethod` or function scope. Do not change in RC (would touch tests without need).
- **Regression test:** n/a until fixed.

### L3 — `.vscode/` untracked

- **Evidence:** `.vscode/` present, untracked, not ignored.
- **Impact:** editor-specific files may be committed unintentionally.
- **Recommended correction:** add `.vscode/` to `.gitignore` (or commit deliberately if team-shared settings are wanted).
- **Regression test:** n/a.

---

## INFORMATIONAL

- **I1 — Test-count reconciliation resolved:** 528 passed/8 skipped = pre-3.2 baseline with Redis down; 543 passed = current suite (15 tests added in 3.2) with Redis up. Both accurate for their states. Definitive RC baseline: **543 passed / 0 skipped / 4 warnings**.
- **I2 — Python version unpinned:** running 3.14.6; no `requires-python` declared. Recommend documenting `>=3.11`.
- **I3 — Repo-relative paths:** all data/model/report paths assume cwd = repo root. Fine for documented `streamlit run app.py` flow; note in README for other launch methods.
- **I4 — `data/stream_checkpoint/` absent until first streaming run** — handled gracefully (verified via page smoke test).
- **I5 — `audit_probe_tmp.py`** in IDE open-files list does not exist on disk — stale editor tab, no action.
- **I6 — Dependency ranges unpinned:** audited versions recorded in `dependency_inventory.md`; consider a constraints/lock file for reproducible RC installs.
- **I7 — `incidents.json` accumulates incidents across repeated analyses** (deterministic IDs; merge/split available) — documented pre-existing behavior from M3.2, unchanged.

---

## Confirmed release blockers (must resolve before tagging v1.0-rc)

1. **C1** — create the checkpoint commit/tag so the RC is recoverable.
2. **H1** — gitignore `.env` before that commit.
3. **M1** — sync `.env`/`REDIS_URL` and document that `.env` is not auto-loaded.
4. **M2/M3** — decide tracking policy for generated artifacts; remove or exclude `nul`.

Everything else is post-RC cleanup. No code defects, test failures, missing artifacts, or threshold/model changes were found.
