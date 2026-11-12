# CAID Milestone 1.8.1 — Test Results

**Date:** 2026-11-12 · All results actually executed; nothing fabricated.

## New regression tests — `tests/test_milestone_18_1.py`

**27 tests, all passing** (`python -m pytest tests/test_milestone_18_1.py -q`
→ `27 passed in 17.42s`).

| Class | Cases covered |
|-------|---------------|
| `TestKnownEntities` (3) | known user/IP/event-type use stored per-entity baselines |
| `TestUnseenEntities` (6) | unseen user typical-user fallbacks (no population counts), population-z semantics, unseen IP per-entity-mean fallback, unseen event-type fallback, multiple unseen entities in one batch without cross-contamination |
| `TestDegenerateBaselines` (6) | unfitted predict raises, empty fit rejected (both managers), single-row baseline, zero-variance baseline, missing/malformed entity values (None/NaN/numeric), missing feature columns |
| `TestDeterminismAndImmutability` (3) | repeated `predict()` identical, repeated `engineer_features` identical, baselines unmutated after predict |
| `TestLeaveOneOutTrainingZ` (3) | LOO excludes self (hand-verified value −2.1213), identical-event users → z=0, predict-time known-user z unchanged |
| `TestEndToEndScores` (3) | unseen normal user gets no population-wide count and score < 0.9; abnormal event still flagged with score > benign; eval-normal FPR < 0.30 and anomaly recall > 0.80 |
| `TestCalibrationIntegrity` (4) | calibration uses baseline feature space, thresholds fixed during holdout eval, eval input not mutated, missing-ML → explicit neutral 0.5 + `ml_score_missing` |

Required-case coverage vs milestone list: known user ✓, unseen user ✓,
known IP ✓, unseen IP ✓, known event type ✓, unseen event type ✓,
multiple unseen in batch ✓, empty/incomplete baseline ✓, zero-variance ✓,
missing/malformed entities ✓, repeated-prediction determinism ✓,
prediction-doesn't-mutate-baselines ✓, feature values AND final scores ✓,
unseen-normal-no-population-count ✓, abnormal-still-detectable ✓.

## Updated existing test

`tests/test_milestone_17.py::test_unseen_user_gets_global_stats` — updated
to assert the corrected typical-user semantics
(`user_mean_api == typical_mean_api`), matching the M1.8 precedent of
fixing stale assertions that encoded buggy fallback values. No other
assertions weakened.

## Full suite

`python -m pytest tests/ -q` → **301 passed, 0 failed, 0 skipped, 0 errors**
(274 pre-existing + 27 new). Runtime ≈ 80 s.

## Before/after false-positive measurement (evaluation_data.csv)

| Metric | Before 1.8.1 (artifact v1) | After 1.8.1 (artifact v2) |
|--------|---------------------------|---------------------------|
| eval-normal `anomaly_score_normalized` mean | 0.585 | **0.354** |
| eval-normal `is_anomaly` rate (record) | 0.68 | **0.12** |
| eval-suspicious `is_anomaly` rate | 1.00 | 1.00 |
| eval-high_risk `is_anomaly` rate | 1.00 | 1.00 |
| binary-mode record FPR (IF flag) | ~0.68 | **0.12** |
| binary-mode record precision/recall (IF flag) | — | 0.893 / 1.000 |
| binary-mode ROC-AUC / PR-AUC (normalized score) | — | 1.000 / 1.000 |

Detection metrics detail: `MILESTONE_1_8_1_SCORE_DISTRIBUTION_REPORT.md`.
Note the IF flag is a binary anomaly detector; "high-risk mode" metrics
treat suspicious as negative and are reported for completeness — the IF
flag legitimately fires on suspicious events.
