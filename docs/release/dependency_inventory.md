# CAID Phase 4.0 — Dependency & Configuration Inventory

Date: 2026-11-12

## Declared dependencies (`requirements.txt`)

| Package | Constraint | Installed | Notes |
|---|---|---|---|
| streamlit | >=1.67,<2 | 1.67.0 | OK |
| pandas | >=3.1,<4 | 3.1.1 | OK |
| numpy | >=2.5,<3 | 2.5.4 | OK |
| scikit-learn | >=1.9,<2 | 1.9.1 | OK |
| joblib | >=1.4,<2 | 1.6.0 | OK |
| plotly | >=5.0,<8 | 7.1.0 | OK |
| matplotlib | >=3.9,<4 | 3.11.2 | OK |
| pytest | >=8.0,<10 | 9.2.0 | OK |
| reportlab | >=5.1,<6 | 5.1.0 | OK |
| redis | >=5,<7 | 6.4.0 | OK |

- All third-party imports found by AST scan of the codebase (`joblib, numpy, pandas, plotly, pytest, redis, reportlab, sklearn, streamlit`) are covered by requirements.txt. No undeclared dependencies.
- No upper/lower bound violations. Constraints are ranges, not pins — a fresh install resolves to newest-in-range, which may differ from the audited environment. **Recommendation:** record the audited versions above (or add a lock/pins file) for the RC.
- No `python_requires` / `requires-python` metadata anywhere; project effectively targets the Python running the venv (**3.14.6**). pandas 3.1.x / sklearn 1.9.x require Python ≥3.11 in practice; declare `Python >=3.11` in README for support clarity.

## Environment variables

| Variable | Used by | Default | .env.example | .env (actual) |
|---|---|---|---|---|
| `OLLAMA_BASE_URL` | `core/ollama_client.py`, `app_pages/settings.py` | `http://localhost:11434` | yes | yes |
| `OLLAMA_MODEL` | same | `llama3.1:8b` | yes | yes |
| `OLLAMA_TIMEOUT` | same | `60` | yes | yes |
| `REDIS_URL` | `core/streaming/event_queue.py`, `scripts/stream_benchmark.py`, tests | `redis://127.0.0.1:6379/0` | yes | **MISSING** |

**Finding (Medium): `.env` is never loaded by the application.** No `dotenv`/`python-dotenv` in requirements or code; all consumers read `os.environ` directly. `.env` and `.env.example` are inert documentation — values only take effect if exported into the real environment. Also `.env` has drifted from `.env.example` (missing `REDIS_URL`). Either (a) document that `.env` must be sourced/exported, or (b) add `python-dotenv` + `load_dotenv()`. Minimal fix: sync `.env` and clarify README.

**Finding (High): `.env` is not in `.gitignore`.** `.gitignore` covers only `.venv/`, `__pycache__/`, `.pytest_cache/`, `.streamlit/secrets.toml`. `.env` is currently untracked but `git add -A` would commit it. Today's contents are benign (localhost URLs only), but the pattern is unsafe for future secrets. Add `.env` (and `.env.*` with `!.env.example`) to `.gitignore` before the checkpoint commit.

## Hardcoded paths

- No user-specific absolute paths (`C:\Users`, `/home/`, `/Users/`) in any `.py` source. The only `C:\` matches are inside `data/mitre_attack/enterprise-attack.json` (STIX dataset text — legitimate).
- All data paths are repo-relative (`data/incidents/incidents.json`, `data/stream_dlq/`, `data/stream_results/`, `models/saved_models/calibration/`, `reports/`). Working-directory-relative — fine for `streamlit run` from repo root; running from another cwd would break paths. Informational.
- `data/stream_checkpoint/` directory referenced as default checkpoint path in `app_pages/streaming_health.py` does not exist yet; `checkpoint_meta()` handles absence (page smoke test passes). Created on demand when streaming runs.

## Model & calibration artifacts

| Artifact | Status |
|---|---|
| `models/saved_models/anomaly_detector.joblib` | tracked, present |
| `models/saved_models/behavioral_baselines.joblib` | tracked, present |
| `models/saved_models/sequence_model.joblib` | **untracked — new, must be committed** |
| `models/saved_models/session_baselines.joblib` | **untracked — new, must be committed** |
| `models/saved_models/calibration/thresholds_0cde9fd80c5f6d2d-3ab7b78cf14b2f21.joblib` | untracked, matches manifest fingerprint `3ab7b78cf14b2f21` |
| `models/saved_models/calibration/v2_0_thresholds_0cde9fd80c5f6d2d-3ab7b78cf14b2f21.joblib` | untracked, same fingerprint |
| `models/saved_models/calibration/archive/` (2 old joblibs) | untracked — superseded artifacts preserved, not deleted |
| `reports/calibration_v2_manifest.json` | untracked — fingerprint-bound to detector `0cde9fd80c5f6d2d` |

Git status shows the two previously-tracked calibration joblibs (`-631ba9b38fb99d53`, `-d735d1b3fd4c6c0e`) **deleted in working tree** but preserved under `calibration/archive/`. The manifest fingerprints match the new files — consistent, intentional artifact rotation. **Action:** ensure the deletions + new files + archive are committed together so no artifact is lost.

## Datasets

- `data/raw/` (4 CSVs), `data/processed/evaluation_data.csv`, `data/incidents/incidents.json` (3.1 MB, tracked, modified — accumulates incidents per analysis), `data/mitre_attack/enterprise-attack.json` + meta — all present.
- MITRE STIX bundle present and used by Settings/MITRE pages (706 techniques per M3.2 report).

## Redis / Ollama configuration

- Redis: `REDIS_URL` env or default `redis://127.0.0.1:6379/0`; queue adapter in `core/streaming/event_queue.py`; DLQ at `data/stream_dlq/bench_*.jsonl`; results at `data/stream_results/`; checkpoint default `data/stream_checkpoint/stream_checkpoint.json`.
- Ollama: HTTP API via stdlib `urllib` (`core/ollama_client.py`); no Python package needed; graceful degraded fallback when unavailable (tested path).

## Repository hygiene observations

- `nul` — 103-byte junk file at repo root (Windows `> nul` redirect artifact). Untracked; inspect contents then delete. Low.
- `modules/dashboard.py` — confirmed dead: zero imports outside a test docstring mention. Prior audit documented it; retained per "no deletion without proof" — proof now exists via import scan. Low.
- `data/stream_dlq/` + `data/stream_results/`: 29+29 benchmark JSONL artifacts, untracked. Decide: commit as evidence or gitignore as generated output. Medium (repo size).
- `reports/stream_benchmark_*.json`: 27 untracked benchmark reports — same decision.
- `.vscode/` untracked — editor settings; typically gitignored or committed deliberately.
- `audit_probe_tmp.py` appears in the IDE open-files list but does not exist on disk — stale tab, ignore.
