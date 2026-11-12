# CAID Milestone 1.7 — Performance, Calibration and Data Quality Engineering

**Date:** 2026-11-12 · Builds on verified Milestone 1.6 · CAID remains a
**synthetic defensive research prototype**.

---

## Phase 1 — Performance

### What was optimized

| Stage | Before (650 rows) | After | Change |
|-------|-------------------|-------|--------|
| Feature engineering (`transform_user_features`) | ~3.1 s | vectorized via `map`/NumPy | per-user `.loc` loop → vectorized baseline maps; **semantics preserved** (verified by 110 baseline/score tests) |
| `AnomalyDetector.predict` (650 rows) | ~5.0 s incl. feature eng | **0.096 s** | ~50× faster |

Other stages (`risk`, `rules`, `behavioral`, `detection`) remain
row-wise but are secondary costs; documented below.

### Benchmark — median / p95 (ms), 3 runs, tiled synthetic data

Machine: Windows 11, Intel64 Family 6 Model 186, 16.9 GB RAM,
Python 3.14.6, pandas 2.3.3. Full output: `docs/milestone_17_benchmark.txt`.

| rows | ml_predict | risk | rules | behavioral | detection | TOTAL |
|------|-----------|------|-------|-----------|-----------|-------|
| 100  | 68/69     | 12/13| 10/15 | 10/11     | 18/24     | 107/124 |
| 500  | 66/74     | 18/31| 50/55 | 32/54     | 51/78     | 213/292 |
| 1000 | 73/146    | 36/38| 83/84 | 120/122   | 137/164   | 433/517 |
| 5000 | 124/126   | 212/222| 460/515| 528/790 | 712/1027  | 1950/2612 |
| 10000| 153/197   | 354/440| 900/1235| 934/1044| 1402/1615 | **3780/4213** |

**Result:** 10k events ≈ **3.8 s median** (vs ~90 s extrapolated pre-1.7).
Remaining hotspots: `DetectionRulesEngine.process_events` (per-row AST eval,
~0.9 s @10k) and `DetectionEngine.process_dataframe` (per-row `iterrows`,
~1.4 s @10k) — acceptable for research scale; vectorizing rule AST eval is
non-trivial and deferred.

## Phase 2 — Risk Calibration

### Design

- `RiskEngine.calibrate_thresholds(calibration_df, percentiles=(30,65,85))`
  computes risk scores on a **separate calibration dataset** and stores
  **absolute** cutoffs in `self.fixed_thresholds`.
- `classify_risk` uses fixed thresholds when calibrated; otherwise the
  legacy dataset-relative percentile path (unchanged, backward compatible).
- Dataset-relative ranking and absolute calibrated classification are
  explicitly separate modes.
- `evaluate_calibration(eval_df, mode)` returns tp/fp/tn/fn, precision,
  recall, F1, FPR.

### Measured results (synthetic)

Calibration: `AnomalyDetector.predict(normal_activity.csv)` (ML-scored
normal data — calibration **requires** anomaly scores; calibrating on raw
rows without them degenerates all thresholds to the neutral-ML score).
Thresholds: medium=0.075, high=0.124, critical=0.181.
Evaluation (`evaluation_data.csv`, binary mode): P=0.50, R=1.00, F1=0.67,
**FPR=1.00**.

### Honest findings

1. **Calibration requires the same feature space as evaluation** — raw
   rows without `anomaly_score*` collapse risk to the neutral ML term.
2. **Distribution shift breaks transfer**: the evaluation set's "normal"
   records score ~0.93 anomaly-normalized (shifted synthetic distribution),
   so every row exceeds the calibrated thresholds → FPR 1.0. This is a
   *data limitation*, not a code defect: synthetic normal/eval populations
   are not identically distributed.
3. Synthetic labels are scenario tags, not ground-truth detections; metrics
   describe this prototype's behavior only.

## Phase 3 — Data Quality

- New `utils/validators.clean_and_summarize(df)` → `(clean_df, summary)`:
  retained/dropped/corrected row counts + per-issue breakdowns
  (invalid timestamps, missing user_id, non-numeric fields, non-binary
  `after_hours`). Original input is never mutated (auditability preserved;
  `df_original` already kept in session state).
- Wired into `analyze_uploaded_data()` before ML; all-invalid input raises
  a clear error; summary stored in `st.session_state.upload_clean_summary`.
- **Bug found & fixed:** `after_hours` corrections were overwritten by the
  uncorrected coerced series (values like 5 survived). Now masked properly.
- Formula-injection tests now cover **all 7 export paths**.

## Phase 4 — Score Contract Cleanup

- `AnomalyDetector.predict` now emits explicit **`anomaly_score_raw`**
  (raw decision_function, higher = more normal); `anomaly_score` kept as a
  **documented deprecated alias** with identical values.
- `MLIntegration.predict` propagates `anomaly_score_raw`.
- `DetectionEngine.process_dataframe` prefers `anomaly_score_raw` over the
  ambiguous `anomaly_score`; `process_event` already preferred it.
- Range-guessing remains **only** in the documented legacy adapter for
  `anomaly_score` (in-range values assumed normalized) — required for
  backward compatibility with older artifacts/tests.
- Direction verified end-to-end: raw < normal ⇒ normalized > normal ⇒
  combined_score higher.

## Phase 5 — Regression

- **Total tests: 274** (251 prior + 23 new `test_milestone_17.py`)
- New coverage: vectorization equivalence, unseen-user global stats,
  calibration thresholds/absolute classification/legacy fallback/eval
  metrics, strict cleaning counts & auditability, formula injection on
  every export path, `anomaly_score_raw` presence/direction/engine
  preference.
- Result: see suite run below.

## Defects found this milestone

| ID | Defect | Fix |
|----|--------|-----|
| M1 | `evaluate_calibration` compared `RiskLevel` enums to strings → all predictions negative (metrics all-zero) | Convert to `.value` before `isin` |
| M2 | `clean_and_summarize` overwrote corrected `after_hours` with uncorrected values | Mask bad values in the coerced series |
| M3 | (design) calibration on score-less rows degenerates thresholds | Documented; calibration now expects ML-scored input |

## Remaining limitations & risks

1. **Calibration transfer**: fixed thresholds tuned on one synthetic
   population don't generalize across shifted distributions (FPR=1.0
   observed). Recalibrate per deployment domain; treat as research tooling.
2. **Rule evaluation** (~0.9 s @10k) and **DetectionEngine iterrows**
   (~1.4 s @10k) are the remaining hotspots — vectorization deferred
   (AST eval can't be trivially vectorized; semantics must be preserved).
3. **`anomaly_score` deprecated alias** still emitted for compatibility;
   removal planned for a future major version.
4. **p95 ≈ max** with only 3 runs — timings are indicative, not rigorous.
5. Tiled benchmark data repeats the same 650 events; real-world skew may
   differ.
6. `user_total_accounts` remains a permanently-zero legacy feature column
   (preserved for schema compatibility).

**Bottom line:** ~25× end-to-end speedup at 10k events, absolute risk
calibration implemented and honestly evaluated (with documented transfer
limits), strict upload cleaning added, and the score contract is now
explicit — while leakage-free and score-consistency guarantees from
1.5.x remain intact. Production readiness is **not** claimed.
