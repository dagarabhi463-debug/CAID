# CAID Milestone 1.8.1 — Audit Report

**Date:** 2026-11-12 · Builds on Milestone 1.8 · Synthetic defensive research
prototype only — no real Canvas infrastructure.

## Executive summary

The residual false-positive elevation (eval-normal mean
`anomaly_score_normalized` ≈ 0.585 after the M1.8 user-count fix) was **not**
caused by the suspected IP/event-type pooled-total fallbacks — those were
verified to already use per-entity mean counts. Two genuine defects were
found and fixed:

| ID | Defect | Status |
|----|--------|--------|
| B1 | **Self-referential training z-scores**: `user_*_z` at fit time scored each event against a baseline that *included the event itself*, producing a near-degenerate distribution (~0 ± 0.16). At predict time every honest deviation — known *or* unseen user — produced z values far outside the learned range, so the IsolationForest treated routine unseen-user activity as maximally anomalous. This was the dominant residual FP driver. | **Fixed** — leave-one-out z at fit (`leave_one_out=True`), model retrained |
| B2 | **Pooled-statistic unseen-user aggregates**: `user_mean_*`/`user_max_*`/`user_total_*` fell back to pooled population mean (and pooled std for `user_max`), so unseen users presented `user_max_api` ≈ 78.6 vs a training mean of 57.0 — a systematic ~0.8σ shift on every unseen row. | **Fixed** — typical-user statistics (mean across per-user feature distribution) |
| B3 | `FeatureBaselineManager.fit()` accepted empty DataFrames, producing NaN baselines silently. | **Fixed** — `ValueError` guard |

Verified **non-defects** (audited, no change):

- `ip_event_count` unseen fallback = mean events per IP (2.31) — already a
  per-entity typical count, not a pooled total.
- `event_type_frequency` unseen fallback = mean events per type (100.0) —
  same, already correct.
- Behavioral analyzer unseen-user fallback = pooled per-event mean/std —
  appropriate for per-event deviation scoring (pooled std is conservative);
  `event_count`/`user_count` are stored but unused in scoring.
- `ip_class` is not an ML feature; the `suspicious_ip_class` rule cannot
  fire on eval-normal rows (verified rule_score = 0.000 for all 100).
- `user_total_accounts` remains permanently 0.0 — documented schema quirk,
  zero-variance, retained for artifact compatibility.

## Measured effect (evaluation_data.csv, n=200)

| Label | anomaly_score_normalized mean (before → after) | is_anomaly rate (before → after) |
|-------|------------------------------------------------|----------------------------------|
| normal | 0.585 → **0.354** | 0.68 → **0.12** |
| suspicious | 0.835 → 0.831 | 1.00 → 1.00 |
| high_risk | 0.837 → 0.853 | 1.00 → 1.00 |

Detection on labeled anomalies is fully preserved; normal FPR dropped by
~82% relative. See `MILESTONE_1_8_1_SCORE_DISTRIBUTION_REPORT.md` for full
statistics and `MILESTONE_1_8_1_TEST_RESULTS.md` for the regression suite.

## Phase 1 — Audit scope

Files audited: `core/feature_engineering.py`, `models/anomaly_detector.py`,
`core/behavioral_baseline.py`, `core/behavioural_analyzer.py`,
`core/ml_integration.py`, `core/detection_engine.py`,
`core/detection_rules.py`, `core/risk_engine.py`, all test callers, and
`scripts/`. Empirical verification: engineered features for
`normal_activity.csv` vs `evaluation_data.csv` were compared per feature,
split by known/unseen user/IP/event-type, in both raw and scaler-transformed
space. Full inventory: `MILESTONE_1_8_1_FEATURE_FALLBACK_MATRIX.md`.

Key empirical evidence for B1: scaled-space mean difference between
training and eval-normal was dominated by `user_*_z` features
(+1.78, −1.64, +1.36, +0.74σ) — an order of magnitude larger than any
other feature. Ablation confirmed causality: zeroing unseen-user z dropped
the IF anomaly rate to ~5% but also destroyed detection (z carries the
model's discriminative power), proving the fix must repair *training*
semantics, not mask the feature.

## Phase 2 — Corrections implemented

`core/feature_engineering.py`:
- `FeatureBaselineManager.fit()` now stores `typical_mean_*`,
  `typical_max_*`, `typical_total_*`, `typical_event_count` in
  `global_stats` (mean across training users of each per-user feature).
- `transform_user_features(df, leave_one_out=False)`: unseen-user
  aggregate fallbacks use typical-user stats (`.get()` fallbacks preserve
  compatibility with pre-1.8.1 artifacts); `leave_one_out=True` recomputes
  `user_*_z` excluding each event from its own user's baseline.
- `_loo_zscores()`: per-user leave-one-out z, std ddof=1 floored to 1.0
  (same zero-variance guard as `_safe_std`), clip ±10, non-finite → 0,
  single-event users → 0.
- `fit()` rejects empty DataFrames.

`models/anomaly_detector.py`:
- `fit()` calls `engineer_features(..., leave_one_out=True)`.
- `ARTIFACT_VERSION` 1 → **2** (feature semantics changed; v1 artifacts are
  rejected with a clear error rather than silently loaded).
- `predict()` unchanged — known users still score against their stored
  baseline; unseen users get documented population-z / typical-user
  fallbacks. Feature schema and column order unchanged.

`models/saved_models/anomaly_detector.joblib` retrained on
`normal_activity.csv` and resaved. `behavioral_baselines.joblib`
unchanged (behavioral semantics untouched).

## Phase 6 — Calibration integrity audit

`RiskEngine.calibrate_thresholds()` / `evaluate_calibration()`:

- **Feature space:** calibration consumes `risk_score` rows produced by
  `calculate_risk_score`, which uses `anomaly_score_normalized` when
  present — the intended space. Verified: calibrated on pipeline-processed
  `normal_activity.csv` (n=500, all with ML scores).
- **No holdout leakage:** thresholds derive only from the calibration
  frame; `evaluate_calibration` does not mutate `fixed_thresholds`
  (asserted in tests and in the validation script).
- **Missing ML scores:** handled explicitly — neutral 0.5 contribution
  with `ml_score_missing=True` / `score_source='unavailable'`. **Caveat
  found:** a neutral 0.5 contributes 0.20 to `risk_score`, which exceeds
  the calibrated `high` threshold (~0.13–0.19), so evaluating data with ML
  scores stripped yields degenerate FPR ≈ 1.0. This is documented, not
  "fixed" — changing the neutral value or thresholds is out of scope and
  would be threshold tuning.
- **Unseen representation:** 186/198 eval users are unseen in training —
  good coverage for fallback semantics.
- **Not generalized:** only two CSVs exist; there are no user-disjoint
  train/calibration/holdout splits. No generalized-calibration claim is
  made. The app pipeline (`modules/log_analysis.py`) never calls
  `calibrate_thresholds` — it uses the legacy dataset-relative percentile
  mode; calibrated mode is exercised only via tests/scripts.

## Phase 7 — Artifact integrity

- `feature_columns`: 30 columns, order unchanged (verified against
  `engineer_features` output and persisted artifact).
- `ARTIFACT_VERSION=2`; v1 artifacts raise `ValueError` on `load()` —
  no silent incompatible load. `ensure_model_exists()` falls back to
  retraining on `normal_activity.csv`.
- Baseline manager persists inside the artifact; save→load→predict
  consistency verified (identical predictions and normalized scores).
- `behavioral_baselines.joblib` (v1) still valid — unchanged semantics.
- New `global_stats` keys are additive; `.get()` fallbacks mean a v2
  baseline manager loaded from an older dict would degrade gracefully,
  but version gating makes this moot.

## Defects found & fixed (summary)

| ID | Defect | Fix | Evidence |
|----|--------|-----|----------|
| B1 | Self-referential training `user_*_z` (event in own baseline) → degenerate z distribution → systematic FP on any honest deviation | LOO z at fit + retrain (artifact v2) | eval-normal is_anomaly 0.68→0.12; suspicious/high_risk stay 1.00 |
| B2 | Unseen-user aggregates used pooled mean/std → inflated `user_max_*` etc. | typical-user statistics in `global_stats` | scaled diff on `user_max_api` +0.82σ → ~0 |
| B3 | Empty-DataFrame fit produced NaN baselines | `ValueError` guard | new regression test |

## Remaining limitations & roadmap

1. **Residual normal elevation (~0.354 mean, 12% FPR)** — remaining gap is
   small distribution shift (eval-normal raw features differ slightly from
   training) plus IsolationForest contamination=0.1 by design; not a
   fallback defect. No threshold tuning performed.
2. **User/IP-disjoint dataset splits** still absent — required before any
   calibration-generalization claim (carried from M1.8).
3. **Missing-ML neutral 0.5 vs calibrated thresholds** — degenerate FPR
   when ML scores absent; needs a documented policy decision, not a
   silent fix.
4. **Structured evidence objects** for explainability — still pending.
5. **`user_total_accounts`** zero-variance column — still retained for
   schema compatibility.
6. Diverse (non-tiled) benchmark — still pending.
7. Eval label↔IP-prefix perfect correlation is a synthetic-data artifact;
   detection metrics partly reflect rule-channel separation, not pure ML.
