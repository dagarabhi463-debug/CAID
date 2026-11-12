# CAID v1.0 Release Readiness Checklist

Date: 2026-11-12 — Phase 4.0 audit baseline
Legend: [x] verified pass · [ ] open · [~] partial/conditional

## 1. Repository protection

- [ ] Checkpoint commit created on `master` (or `release/v1.0-rc` branch) covering all M2.4–3.2 work — **BLOCKER C1**
- [ ] Tag `v1.0-rc1` (or chosen RC tag) on the checkpoint commit
- [ ] `.env` added to `.gitignore` before committing — **BLOCKER H1**
- [ ] `nul` file inspected and removed/excluded — **M3**
- [ ] `.vscode/` ignore-or-commit decision made — **L3**
- [ ] Generated-artifact policy decided: which benchmark reports / DLQ / result JSONL are tracked vs ignored — **M2**
- [ ] No persistent data deleted: incidents store, model artifacts, calibration archive, MITRE bundle all preserved (verified present)

## 2. Definitive baseline (verified this audit)

- [x] `pytest tests -q` → **543 passed, 0 skipped, 4 warnings** (Redis available)
- [x] `compileall` clean on `app.py app_pages modules core utils tests scripts`
- [x] `streamlit run app.py` serves HTTP 200
- [x] All 13 registered pages covered by `test_dashboard_pages.py` smoke tests + entrypoint + 2 functional/regression tests
- [x] Redis integration tests pass against live Redis (PONG)
- [x] Latest saved benchmark (`stream_benchmark_20261112T045510Z.json`): 360k events, 0 lost/duplicate/DLQ — consistent with reports
- [x] Test-count discrepancy reconciled (528+8skip pre-3.2 / 543 post-3.2; skips are Redis-availability-dependent)

## 3. Dependencies & configuration

- [x] All imports covered by `requirements.txt`; no undeclared deps
- [x] Installed versions satisfy all constraints (see `dependency_inventory.md`)
- [ ] `.env` synced with `.env.example` (`REDIS_URL` missing) — **M1**
- [ ] README states `.env` is not auto-loaded / env vars must be exported — **M1**
- [ ] Supported Python version documented (`>=3.11`; audited on 3.14.6) — **I2**
- [ ] Optional: audited-version lock/constraints file for reproducible installs — **I6**

## 4. Artifacts & data

- [x] Detector, behavioral baselines, sequence model, session baselines present
- [x] Calibration artifacts fingerprint-consistent with `reports/calibration_v2_manifest.json` (`3ab7b78cf14b2f21`)
- [x] Superseded calibration joblibs preserved under `models/saved_models/calibration/archive/`
- [x] MITRE STIX dataset present (`data/mitre_attack/enterprise-attack.json`)
- [x] Incident store present and functional (`data/incidents/incidents.json`)
- [x] DLQ/results/checkpoint paths consistent between code, scripts, and Streaming health page
- [x] Ollama integration degrades gracefully; CLI 0.40.16 present; no Python dep required

## 5. Pages & workflows (all 13 verified via AppTest in suite)

- [x] Overview — KPIs, load-and-analyze functional test passes
- [x] Events — searchable/filterable, sanitized CSV export
- [x] Anomaly Detection — Isolation Forest + rules + behavioral evidence
- [x] Streaming health — observational metrics, p50/p95/p99 latency, graceful missing-checkpoint handling
- [x] Detection Quality — renders all 4 evaluation JSONs + on-demand drift
- [x] Session Intelligence — timelines, sequence patterns, risk summaries
- [x] Correlation — cross-session findings with evidence
- [x] Investigation — incident triage/notes/merge/split, PDF/CSV export (PDF verified %PDF-1.4 in M3.2)
- [x] MITRE ATT&CK — cached STIX bundle regression test passes (706 techniques)
- [x] AI Analyst — Ollama + deterministic fallback
- [x] Log Analysis — upload→analysis path (unit-tested)
- [x] Settings — status, Ollama config, MITRE dataset management
- [x] Reports & Exports — artifact inventory + export map

## 6. Known limitations (accepted, documented — not blockers)

- Streaming health is observational only (no live Redis probing) — by design
- Incident store accumulates across repeated analyses — pre-existing, documented
- `modules/dashboard.py` dead code retained pending explicit removal decision — **L1**
- 4 pytest deprecation warnings — post-RC cleanup — **L2**
- Log Analysis upload path not AppTest-simulated — noted in M3.2
- Live Ollama generation exercised via connection check only — noted in M3.2

## 7. Sign-off gate for v1.0 RC

- [ ] All section-1 boxes checked
- [ ] Baseline re-run on the checkpoint commit reproduces 543 passed / 0 skipped (with Redis) — expect 528+8skip without Redis
- [ ] `release_blockers.md` shows zero open Critical/High items
- [ ] Human review of `phase_4_0_repository_audit.md` findings
