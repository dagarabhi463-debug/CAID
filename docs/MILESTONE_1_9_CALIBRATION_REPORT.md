# CAID Milestone 1.9 — Calibration Report

**Date:** 2026-11-12 · Implements Phase 4: calibration is now part of the
real application lifecycle, not an isolated method.

## Lifecycle

```
Training (train.csv normals)
  -> AnomalyDetector.fit + BehavioralAnalyzer.fit
       -> models/saved_models/v1_9/*
Calibration (calibration.csv NORMAL rows only, n=725)
  -> full channel pipeline -> RiskEngine.calibrate_thresholds
       -> models/saved_models/calibration/thresholds_<fingerprint>.joblib
Application loading (app.py, modules/log_analysis.py)
  -> core.calibration.apply_to_engine(risk_engine, detector=...)
       -> engine.fixed_thresholds + calibration_status='calibrated'
Evaluation (holdout.csv — never used above)
```

## Components

- `core/calibration.py` — `CalibrationStore`-style persistence:
  - `save_calibration()` writes a versioned joblib artifact containing
    `threshold_version` (`cal-<utc>-<hash8>`), thresholds, percentiles,
    `calibration_info` stats, `detector_fingerprint`, `dataset_role`,
    `dataset_manifest_hash`, `created_utc`.
  - `load_calibration()` validates artifact version, required keys, role,
    and detector fingerprint; raises `CalibrationUnavailable` otherwise.
  - `apply_to_engine()` sets `engine.fixed_thresholds`,
    `engine.calibration_info`, `engine.calibration_status`,
    `engine.threshold_version` — or leaves explicit
    `uncalibrated_dataset_relative` mode.
- `models/anomaly_detector.py` — additive `fingerprint()` via
  `detector_fingerprint()` (hash of artifact version + feature schema +
  hyperparameters). Artifact schema unchanged (still v2).
- `scripts/calibrate_thresholds.py` — CLI entry point. **Role guard:**
  refuses to run unless `dataset_manifest.json` declares the input file's
  role as `calibration` (verified: `--data holdout.csv` exits REFUSED).
- `core/risk_engine.py` — `calibrate_thresholds()` now sets
  `calibration_status='calibrated'`; `evaluate_calibration()` reports
  `calibration_status` + `threshold_version`; risk factors carry
  `ml_validated` (False for neutral placeholders).

## Application wiring (verified)

| Site | File | Behavior |
|------|------|----------|
| Overview pipeline | `app.py::analyze_data` | loads persisted thresholds for the loaded detector; status stored in `st.session_state.risk_calibration_status` |
| Threat-detection pipeline | `app.py::analyze_data_with_milestone_3` | same; RiskEngine now runs in the documented pipeline order (predict → risk → rules → behavioral → detection) |
| Upload pipeline | `modules/log_analysis.py::analyze_uploaded_data` | same |
| Risk display | `modules/log_analysis.py::display_risk_analysis` | best-effort calibration with production detector; explicit fallback |

Uncalibrated mode is **always explicit**: `calibration_status =
'uncalibrated_dataset_relative'` (dataset-relative percentiles) vs
`'calibrated'` (absolute cutoffs). No silent fallback exists —
`apply_to_engine` returns `None` and the engine keeps
`fixed_thresholds=None`.

## Calibration runs

Detector fingerprints are content-bound (schema + hyperparameters +
canonical digest of fitted tree/scaler arrays), so the production v2
detector and the v1_9 evaluation detector each have their own artifact —
thresholds never transfer across models.

**v1_9 evaluation detector** (`models/saved_models/v1_9/`):
- Detector fingerprint: `0cde9fd80c5f6d2d-631ba9b38fb99d53`
- Threshold version: `cal-20261111T211520Z-e7935b64`
- Thresholds: medium=0.0904, high=0.1491, critical=0.2150
- Calibration rows: 725 (normal-labeled calibration split only)
- Artifact: `models/saved_models/calibration/thresholds_0cde9fd80c5f6d2d-631ba9b38fb99d53.joblib`

**Production detector** (`models/saved_models/anomaly_detector.joblib`):
- Detector fingerprint: `0cde9fd80c5f6d2d-3ab7b78cf14b2f21`
- Threshold version: `cal-20261111T211515Z-04303141`
- Thresholds: medium=0.1484, high=0.2041, critical=0.2517
- Calibration rows: 800 (full calibration split via CLI)
- Artifact: `models/saved_models/calibration/thresholds_0cde9fd80c5f6d2d-3ab7b78cf14b2f21.joblib`

Both verified loading via `apply_to_engine` → `calibration_status =
'calibrated'`.

Holdout data never influenced thresholds: the calibrate script enforces
the manifest role, the evaluation script calibrates before touching
holdout, and `evaluate_calibration`/`process_events` do not mutate
`fixed_thresholds` (regression-tested).

## Dataset-relative fallback distinction

When no artifact matches the detector fingerprint (or none exists),
`classify_risk` uses the legacy percentile mode normalized by the
current dataset's risk-score range — reported as
`uncalibrated_dataset_relative`. This is intentionally NOT called
"calibrated" anywhere in outputs.
