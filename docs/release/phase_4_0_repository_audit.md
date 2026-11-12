# CAID Phase 4.0 — Repository & Release Audit

Date: 2026-11-12
Scope: read-only audit of the existing application prior to v1.0 release candidate. **No code, data, thresholds, models, or fusion weights were modified.** Deliverables: this document + `phase_4_0_test_baseline.md`, `dependency_inventory.md`, `release_blockers.md`, `release_readiness_checklist.md`.

---

## 1. Git status & branch state

- Branch: `master` (only branch; no other local/remote branches, no stashes, no remotes configured).
- HEAD: `1744782 "Remove unused files: caches, dead code, superseded datasets/models, docs, .venv"`, preceded by `bfdca25 "Snapshot before cleanup"`.
- **The entire Milestone 2.4 → 3.2 body of work is uncommitted**: 31 modified tracked files (+35,852/−164 per `git diff --stat`) and ~70 untracked paths. See finding **C1**.

### Modified tracked files (31)

`.env.example`, `README.md`, `app.py`, `app_pages/{ai_analyst,events,mitre,overview}.py`, `core/{ai_analyst,ai_context,behavioral_baseline,feature_engineering,incident_manager,sequence_detection,session_analyzer,session_baseline}.py`, `data/generate_scenarios.py`, `data/incidents/incidents.json`, `models/anomaly_detector.py`, `modules/{correlation_dashboard,log_analysis,session_intelligence}.py`, `requirements.txt`, 6 test files, `utils/validators.py`.

Two tracked calibration joblibs show as deleted in the working tree (superseded artifacts — preserved under `calibration/archive/`, see §4).

### Untracked (selected)

- Source: `core/streaming/` (10 modules), `core/{artifact_lifecycle,drift_monitor}.py`, `app_pages/{detection_quality,reports,streaming_health}.py`, `scripts/` (6 files)
- Tests: 12 new test files (`test_redis_streams`, `test_streaming_*`, `test_artifact_lifecycle`, `test_calibration_artifacts`, `test_drift_monitor`, `test_milestone_2_4_reliability`)
- Docs: entire `docs/` tree (architecture, audit, research, roadmap, testing) + `PROJECT_DOCUMENTATION.md`
- Artifacts: new model joblibs, calibration joblibs + manifest, 27 benchmark reports
- Runtime data: `data/stream_dlq/` (29), `data/stream_results/` (29)
- Noise: `.env`, `nul`, `.vscode/`

## 2. Parallel development session identification

Evidence indicates work was done across sequential (not simultaneous) sessions on 2026-11-12, each leaving audit reports:

| Session output | Evidence |
|---|---|
| Milestone 2.4 (reliability/calibration) | `docs/audit/milestone_2_4_*`, `core/calibration.py`, calibration manifest |
| Milestone 2.5/2.5.1 (streaming ingestion + reliability) | `docs/audit/milestone_2_5*`, `core/streaming/`, `test_redis_streams`, `test_milestone_2_4_reliability` |
| Milestone 2.6–2.9 (detection quality, AI threat analysis, validation) | `docs/audit/milestone_2_*`, `scripts/evaluate_*`, `reports/*.json` |
| Milestone 3.x (dashboard completion) | `docs/audit/milestone_3_2_completion_report.md`, new pages, `test_dashboard_pages.py` expansion |

All changes appear **integrated and consistent**: the 543-test suite passes, calibration fingerprints match the manifest, benchmark reports align with `core/streaming/` code, and page tests cover all 13 registered pages. No conflicting duplicate implementations were found (`behavioural_analyzer.py` vs `behavioral_baseline.py` are distinct, both-imported modules — analyzer vs baseline, not duplicates). No evidence of lost or orphaned parallel work.

## 3. Generated / persistent data inventory (must not be lost)

| Item | Location | Tracking | Risk if lost |
|---|---|---|---|
| Incident store | `data/incidents/incidents.json` (3.1 MB) | tracked, modified | analyst work product |
| Detector model | `models/saved_models/anomaly_detector.joblib` | tracked | detection pipeline |
| Baselines | `behavioral_baselines.joblib`, `session_baselines.joblib`, `sequence_model.joblib` | 1 tracked + 2 untracked | detection pipeline |
| Calibration artifacts | `models/saved_models/calibration/` | mixed | threshold integrity (fingerprint-bound) |
| Calibration manifest | `reports/calibration_v2_manifest.json` | untracked | provenance |
| Evaluation reports | `reports/{detection_quality,fusion_evaluation,ml_benchmark_results}.json` | untracked | Detection Quality page data |
| Benchmark reports | `reports/stream_benchmark_*.json` ×27 | untracked | streaming evidence |
| Benchmark I/O | `data/stream_results/`, `data/stream_dlq/` ×29 each | untracked | benchmark evidence |
| MITRE dataset | `data/mitre_attack/enterprise-attack.json` | tracked | MITRE page |
| Datasets | `data/raw/*.csv`, `data/processed/evaluation_data.csv` | tracked | analysis scenarios |
| Audit docs | `docs/audit/*` ×13 | untracked | milestone provenance |

## 4. Backup & checkpoint recommendation

**Recommended safe sequence (requires approval — modifies git):**

1. Add `.env`, `.env.*` (except `.env.example`), `nul` handling, and runtime-output ignores (`data/stream_dlq/`, `data/stream_results/`, `data/stream_checkpoint/`, `.vscode/`) to `.gitignore`.
2. Delete or relocate `nul` after inspecting contents.
3. Stage everything else deliberately — including new source, tests, docs, model artifacts, calibration files **and** the archive, manifest, evaluation reports, and a curated set of benchmark reports.
4. Commit on `master` (or branch `release/v1.0-rc`) with message e.g. `Milestones 2.4-3.2: streaming ingestion, detection quality, dashboard completion — v1.0 RC checkpoint`.
5. Tag `v1.0-rc1`.
6. Optionally `git bundle` / zip the repo + `data/` + `models/` + `reports/` to a timestamped archive outside OneDrive as belt-and-braces backup before tagging.

**Do not** run `git clean`, `git checkout .`, or delete `data/`, `models/`, `reports/` — all contain unrecoverable work product.

## 5. Application functionality verification

All 13 pages registered in `app.py` (`st.navigation`, Monitor/Investigate/AI/Data & Admin groups) are exercised by `tests/test_dashboard_pages.py` — 13 parametrized smoke tests + entrypoint test + MITRE cached-bundle regression + Overview load-and-analyze functional test. All pass (543 total). Page inventory and per-page backend wiring confirmed against `docs/audit/complete_dashboard_inventory.md` and the live `app.py` — no unregistered or missing pages; no page file exists without registration (`app_pages/` contains exactly the 13 registered files).

Cross-checks performed:
- Calibration manifest fingerprints ↔ artifact filenames: consistent (`3ab7b78cf14b2f21`).
- Superseded calibration joblibs: tracked-deleted in worktree, preserved in `calibration/archive/` — must be committed together.
- DLQ/results/checkpoint paths consistent across `core/streaming/`, `scripts/stream_benchmark.py`, `app_pages/streaming_health.py`.
- No hardcoded user paths in code; MITRE JSON `C:\` strings are dataset content.
- Third-party imports fully covered by `requirements.txt`.

## 6. Missing or inconsistent files

- `.env` missing `REDIS_URL` vs `.env.example` (**M1**); `.env` never loaded by app code (**M1**).
- `.gitignore` incomplete: `.env`, `nul`, `.vscode/`, runtime output dirs (**H1, M2, M3, L3**).
- `data/stream_checkpoint/` absent (created on demand; handled — **I4**).
- `modules/dashboard.py` dead code, proven unused (**L1**).
- `audit_probe_tmp.py` — IDE tab references a nonexistent file; ignore (**I5**).

## 7. Definitive test baseline

See `phase_4_0_test_baseline.md`. Summary: **543 passed / 0 skipped / 4 warnings / 162.5s** on Python 3.14.6 with Redis available; compileall clean; Streamlit HTTP 200; latest soak benchmark 360k events, zero loss/duplicates/DLQ.

## 8. Audit conclusion

The application is functionally healthy: full suite green, all pages verified, artifacts consistent, no code defects found. The release risk is entirely **repository-state risk**: a large, verified body of work exists only in the working tree. The minimal path to a safe v1.0 RC is: fix `.gitignore`, remove `nul`, commit + tag, then address the Medium/Low hygiene items. No detection logic, thresholds, models, or architecture changes are required or recommended.
