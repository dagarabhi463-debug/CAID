# CAID Milestone 2.0.1 — Final Fixes Summary

Scope: synthetic defensive research only. No real Canvas connectivity.

## Changes

- Added deterministic `FeatureBaselineManager.fingerprint()` covering fitted user/IP/event/global baseline state.
- Calibration verification now includes `feature_baseline`; the saved v2.0 calibration artifact was updated with the missing fingerprint metadata.
- `apply_detection_calibration()` now leaves `DetectionEngine` explicitly `uncalibrated_default_thresholds` when no compatible artifact is available.
- The session-intelligence dashboard path now attempts fingerprint-verified v2 detection-calibration loading before final `DetectionEngine` scoring.
- Common-IP evaluation now includes mixed shared-IP rows and normal-only shared-IP rows in addition to positive-only rows.
- Added `tests/test_milestone_2_1.py` covering feature-baseline fingerprints, calibration rejection, calibrated DetectionEngine thresholds, shared-IP user separation, and independent-v2 mixed-IP coverage.

## Verification

- `.venv/Scripts/python.exe -m pytest tests -q` → **361 passed**, 221 warnings in 62.67s.
- Focused v1.9/v2.0, baseline, causal, calibration, and DetectionEngine tests → **90 passed**.
- New Milestone 2.0.1 tests → **7 passed**.
- Manual v2 calibration load with all component fingerprints → passed.
- Dashboard smoke check → `operating_mode=full_v2_0`, final `risk_level` present, `risk_calibration_status=calibrated`, `detection_calibration_status=uncalibrated_default_thresholds` for the production/sample artifacts.

## Remaining limitations

- The dashboard uses production/sample artifacts; the independent-v2 calibration artifact is fingerprint-bound to the v2.0 evaluation artifacts, so dashboard detection calibration correctly remains uncalibrated unless a compatible artifact set is supplied.
- `BehavioralBaselineManager` still uses the existing generic component fingerprint path rather than a dedicated full-state fingerprint; this was left unchanged to avoid expanding scope.
- Common-IP metrics are synthetic evaluation coverage, not a guarantee against all real-world IP-sharing ambiguity.
