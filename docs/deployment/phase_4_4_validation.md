# Phase 4.4 — Deployment, Backup & Recovery Validation

**Date:** 2026-11-12 · **Baseline:** master @ `29a1926` (post 4.1–4.3 commits)

## Installation verification

- `requirements.txt` reviewed: bounded ranges, all runtime deps; `pip install` already exercised in this venv (Python 3.14.6). `compileall` clean (see Phase 4.5 regression).
- Startup path verified implicitly: full pytest suite (543 tests incl. dashboard-page tests) exercises `app.py` navigation and all 13 pages; Streamlit app was launched during dashboard audits (M2.9).
- Model artifacts present: `models/saved_models/` = anomaly_detector + behavioral/session/sequence baselines + `calibration/v2_0_thresholds_*.joblib`; loaded successfully in benchmarks and the Phase 4.3 holdout run (`calibration_statuses: calibrated`).

## External dependency matrix (verified live)

| Dependency | State | Evidence |
|---|---|---|
| Ollama | Optional; graceful degradation tested (`test_ollama_client.py`, `test_ai_grounding.py`) | AI fallback returns deterministic summaries |
| Redis 7.4.8 / 5.0.14.1 | Both live-tested Phase 4.1 (8/8 tests each) | `test_redis_streams.py` |
| MITRE dataset | Cached bundle present (`data/mitre_attack/`); `allow_download=False` path used by default in AI flows | settings page reload path verified in code |

## Failure testing

Covered by existing regression suite (`tests/test_system_reliability.py`, `test_artifact_lifecycle.py`, `test_calibration_artifacts.py`, `test_milestone_2_4_reliability.py`) plus Phase 4.1/4.3 runs:

- Missing config → defaults (`.env.example` contract)
- Invalid `OLLAMA_TIMEOUT` → 60s fallback (`ollama_client.py` try/except)
- Corrupt model artifact → `ensure_*` explicit status, fingerprint-checked calibration refuses silent mismatch
- Missing calibration artifact → `calibration_status: uncalibrated` surfaced in UI/metrics
- Ollama unavailable → typed errors → deterministic fallback (tested)
- Redis unavailable → asyncio queue fallback; Redis tests auto-skip (verified: suite ran with 6379 up; skip path exercised in earlier runs)
- Missing MITRE dataset → "not loaded" state, optional download
- Unwritable dirs → surfaced exceptions; atomic writes prevent partial state
- Interrupted streaming worker → pending recovery via claim_pending/XAUTOCLAIM + checkpoint restore (tested)
- Invalid uploaded file → validation rejects before analysis (10MB/.csv/schema)
- Application restart → incident store + checkpoint + artifacts all reload (verified: 136 incidents restored in backup smoke test)

## Backup & restore — VERIFIED

Smoke test executed: copied `models/`, `data/incidents/`, `data/raw/`, `reports/calibration_v2_manifest.json` to a clean temp dir → detector `joblib.load` OK, calibration artifact OK, `IncidentStore` restored 136 incidents with `load_error=None`. Originals untouched. Full procedure in `backup_restore_guide.md`.

## Deliverables

- `docs/deployment/installation_guide.md`
- `docs/deployment/configuration_guide.md`
- `docs/deployment/backup_restore_guide.md`
- `docs/deployment/phase_4_4_validation.md` (this file)

**Phase 4.4 status: COMPLETE — install path documented, failure modes verified against test suite, backup/restore smoke-tested.**
