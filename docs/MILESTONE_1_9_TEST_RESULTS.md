# CAID Milestone 1.9 — Test Results

**Date:** 2026-11-12 · Command: `.venv/Scripts/python.exe -m pytest tests -q`

## Result

**327 passed, 0 failed** (48.0s) — up from 301 at Milestone 1.8.1.

| Suite | Tests | Status |
|-------|------:|--------|
| tests/test_anomaly_detector.py | — | pass |
| tests/test_baseline_integration.py | — | pass |
| tests/test_behavioral_analyzer.py | — | pass |
| tests/test_behavioral_baseline.py | — | pass |
| tests/test_detection_engine.py | — | pass (1 updated for insufficient-evidence contract) |
| tests/test_detection_rules.py | — | pass (custom suspicious_ip_class rule still addable; removed from defaults) |
| tests/test_log_analysis.py | — | pass |
| tests/test_milestone_17.py | — | pass (incl. formula-injection export fix) |
| tests/test_milestone_18_1.py | — | pass |
| tests/test_milestone_1_9.py | **26 new** | pass |
| tests/test_ml_integration.py | — | pass |
| tests/test_risk_engine.py | — | pass |
| tests/test_score_consistency.py | — | pass (3 updated for channel-evidence contract) |
| tests/test_system_reliability.py | — | pass (1 updated for degraded-mode contract) |

## New coverage (`tests/test_milestone_1_9.py`)

- **Independent datasets**: splits exist; labels don't determine IP
  prefix; holdout-only entities absent from train yet present in holdout;
  shared entities span splits; zero cross-split duplicates; manifest
  records seed/roles.
- **Leakage guards**: `suspicious_ip_class` absent from default rules and
  no `ip_class` in any default condition; `label`/`ip_class` not ML
  features; `is_sensitive_event` overlap verified on holdout.
- **Calibration lifecycle**: artifact round-trip preserves
  thresholds/version/manifest hash/fingerprint; fingerprint mismatch and
  non-calibration roles rejected; `apply_to_engine` sets calibrated status
  + version and fixed-threshold classification; missing artifact →
  explicit `uncalibrated_dataset_relative`; CLI refuses holdout data.
- **Operating modes**: full / degraded (exact renormalized score, no
  neutral-0.5 term) / insufficient_evidence (NaN, `RiskLevel.INSUFFICIENT`,
  no High/Critical); mode surfaced in explanations.
- **Structured evidence**: required fields present per object; channels
  {ml, rules, behavioral, combined}; missing channels marked excluded;
  legacy explanation string preserved.
- **Artifacts**: v2 load + deterministic predict + save/load prediction
  consistency; incompatible version fails; feature-schema mismatch
  rejected; default calibration status uncalibrated.

## Pre-existing tests intentionally updated

| Test | Reason |
|------|--------|
| `test_detection_engine.py::test_process_event_fallback_values` | bare row now yields `insufficient_evidence`/NaN per new contract |
| `test_score_consistency.py` (3 tests) | frames given real rule/behavioral channel inputs so intended scoring path is exercised |
| `test_system_reliability.py::test_missing_score_propagation` | rules engine added so ml+rules form a degraded 2-channel mode |

## Pre-existing environment fix

`test_milestone_17.py::test_get_download_button_csv` was failing before
this milestone (pandas 3.x `str` dtype bypassed the `dtype == object`
check in `_sanitize_csv_frame`). Fixed in `modules/export.py`; all
formula-injection export tests now pass.

## Warnings (non-blocking)

Pandas4Warnings (`dt.dayofweek` deprecation, `fillna` dtype notices) and
PytestRemovedIn10Warnings (class-scoped fixture style) — cosmetic, no
functional impact.
