# Calibration Reliability — Milestone 2.4

**Date:** 2026-11-12
**Scope:** `core/calibration.py`, `models/saved_models/calibration/`,
`models/anomaly_detector.py`, `core/detection_engine.py`,
`core/risk_engine.py`, `modules/session_intelligence.py`

## 1. Artifact inventory & live verification

Executed check: load each artifact in
`models/saved_models/calibration/` against the **current**
`anomaly_detector.joblib` (detector fingerprint
`0cde9fd80c5f6d2d-3ab7b78cf14b2f21`).

| Artifact | Version | Result |
|---|---|---|
| `thresholds_…-3ab7b78cf14b2f21.joblib` | v1 (risk) | ✅ loads — `cal-20261111T211515Z-04303141` |
| `thresholds_…-631ba9b38fb99d53.joblib` | v1 (risk) | ❌ rejected — detector fingerprint mismatch (`…-631ba9b3…` ≠ current) |
| `v2_0_thresholds_…-d735d1b3fd4c6c0e.joblib` | v2 (detection) | ❌ rejected — detector fingerprint mismatch (`…-d735d1b3…` ≠ current) |

## 2. Key finding — RESOLVED: v2 detection calibration regenerated

**Update (Milestone 2.4 closure):** the stale-artifact finding below
has been remediated. `scripts/calibrate_v2_detection.py` now provides
a controlled, reproducible workflow that loads the **current**
production artifacts (all `loaded` status) and calibrates on the
deterministic entity-disjoint `calibration` split from
`scripts/benchmark_models.py` (73 normal rows / 70 users, seed 42).

- New artifact: `v2_0_thresholds_0cde9fd80c5f6d2d-3ab7b78cf14b2f21.joblib`
  — `threshold_version=cal-20261112T020320Z-2962264f`,
  `channel_profile='v2_0'`, fingerprints bound to detector +
  feature/behavioral/session/sequence components.
- Post-save verification: `apply_detection_calibration` →
  `calibration_status='calibrated'` (was
  `uncalibrated_default_thresholds`).
- Stale artifacts archived to `models/saved_models/calibration/archive/`
  (kept for provenance; correctly rejected by fingerprint checks).
- Provenance manifest: `reports/calibration_v2_manifest.json`
  (roles, seed, split counts, user-overlap disclosure: 65/70
  calibration users share `user_id`s with `normal_activity.csv` —
  disclosed; detector trains on normal_activity rows only, never on
  calibration rows).
- Regression guard: `tests/test_calibration_artifacts.py` fails if the
  production fingerprints drift from the on-disk v2 artifact.

### Original finding (pre-closure, for the record)

`modules/session_intelligence.py` looks up
`v2_0_thresholds_{detector_fingerprint(detector)}.joblib`. The only
v2 artifact on disk was built for an **older fitted state** of the
detector (`…-d735d1b3…`); the current artifact's fingerprint is
`…-3ab7b78c…`. Consequence was:

- `apply_detection_calibration` raised `CalibrationUnavailable`
  internally → returned `None` → `DetectionEngine` stayed in
  **`uncalibrated_default_thresholds`** mode.
- The dashboard surfaced this honestly via
  `detection_calibration_status` / `threshold_version` columns —
  **no silent wrong-threshold application**.
- RiskEngine calibration (`apply_to_engine`) found its matching
  v1 artifact → `calibrated`.

**Root cause:** `anomaly_detector.joblib` was refit/re-saved after
the v2 calibration artifact was produced; the calibration-generation
scripts were removed in commit `1744782`.

## 3. Mechanism audit (verified in source)

| Guarantee | Status | Evidence |
|---|---|---|
| `dataset_role` must be `'calibration'` to persist | ✅ enforced (`save_calibration` raises otherwise) | calibration.py:170-173 |
| Detector fingerprint binding | ✅ `detector_fingerprint` = schema+hyperparams+fitted-state digest | calibration.py:43-97 |
| Component fingerprints (v2) | ✅ `fingerprints` dict checked per component | calibration.py:272-285 |
| Missing artifact → explicit uncalibrated status | ✅ `CalibrationUnavailable` → `uncalibrated_*` labels | calibration.py:290-342 |
| Wrong version → rejected | ✅ `SUPPORTED_ARTIFACT_VERSIONS={1,2}` | calibration.py:246-250 |
| Incomplete schema → rejected | ✅ required-keys check | calibration.py:251-258 |
| Corrupt joblib → rejected | ✅ load wrapped in `CalibrationUnavailable` | calibration.py:236-240 |
| Thresholds never applied silently | ✅ verified live (§1) | — |

## 4. Drift monitoring (new)

`core/drift_monitor.py` — observational distribution-drift component:

- PSI (equal-frequency bins) + two-sample KS per numeric feature;
  total-variation distance for configured categorical columns.
- Size-aware KS thresholds (α=0.01 two-sample critical value) so
  identical small samples don't false-warn.
- Explicit status: `ok` / `warning` / `critical` /
  `insufficient_reference`; per-feature evidence dicts; skipped
  columns reported, never silently dropped.
- **Never** retrains, never changes thresholds/weights/scores;
  drift ≠ malicious activity (stated in every report summary).
- Tests: `tests/test_drift_monitor.py` — 13 tests covering
  identical/shifted distributions, categorical drift, explicit
  statuses, missing columns, non-mutation, determinism.

## 5. Tests added/verified this milestone

- `tests/test_drift_monitor.py` (new, 13 tests) — all pass.
- Existing calibration coverage confirmed present:
  `test_baseline_integration.py`, `test_milestone_2_0*.py` exercise
  fingerprint rejection, missing calibration, uncalibrated fallback.
- Live artifact verification (§1) — executed, results above.

## 6. Recommendations

1. ~~Rebuild the v2 calibration workflow~~ — **DONE** (§2):
   `scripts/calibrate_v2_detection.py` regenerates the artifact on
   the benchmark calibration split; re-run it whenever any bound
   component artifact changes (`tests/test_calibration_artifacts.py`
   guards this).
2. ~~Retire stale calibration artifacts~~ — **DONE**: stale artifacts
   moved to `models/saved_models/calibration/archive/`.
3. Wire `DriftMonitor` into a future streaming/monitoring layer as
   an observational signal only (Milestone 2.5+).
