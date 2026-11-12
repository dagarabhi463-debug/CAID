# Phase 4.6 — Detection Fusion Recalibration (precision fix)

**Date:** 2026-11-12 · **Scope:** v2_0 channel weights + detection operating
point · **Trigger:** Phase 4.3 independent holdout showed combined
precision 0.50 / FPR 0.39 — the dominant v1.0 weakness.

## 1. Problem

Phase 4.3 (`phase_4_3_independent_validation.md`) reported on the
independent_v2 holdout (n=1500): combined_v2_0 **P=0.497, R=0.940,
F1=0.650, FPR=0.389**. Per-channel metrics showed the failure was a
*fusion* problem, not a channel problem:

| Channel | Precision | FPR | Recall |
|---|---|---|---|
| behavioral | 0.885 | 0.038 | 0.706 |
| rules | 0.889 | 0.001 | 0.018 |
| sequence | 0.710 | 0.124 | 0.743 |
| session | 0.838 | 0.017 | 0.214 |
| ml | 0.558 | 0.272 | 0.841 |

False-positive anatomy: among 414 holdout FPs, the ML channel scored
≥0.5 on 197 rows and sequence on 130, while behavioral/session/rules
stayed near zero — FPs were predominantly **single-channel noise**
averaging ~0.93 fired channels vs ~0.15 for true negatives.

Two root causes:

1. **Weights amplified noisy channels.** v2_0 weights gave ml+sequence
   40% of the vote; on OOD synthetic traffic those channels inflate.
2. **Threshold didn't transfer.** `high=0.20` was the p99 of *internal*
   calibration-normal combined scores. Independent_v2 normals score
   systematically higher (p65 ≈ 0.20), so ~39% of normals crossed it.
   Under the new weights this gap widens: internal calibration normals
   score near zero (p99.9 = 0.147), so percentile-derived cutoffs cannot
   express a valid operating point at all.

## 2. Method (leakage-safe)

- Channel scores for internal (`data/raw` + `evaluation_data.csv`
  entity-disjoint splits) and independent_v2 (calibration/validation/
  holdout) corpora were computed once through the production component
  stack and cached for offline analysis.
- **Weight sets and decision thresholds were selected on VALIDATION
  splits only** (internal_validation n=123 + independent_v2 validation
  n=800, pooled n=923). Holdout data was not used for selection.
- Candidates compared: weight re-grids, corroboration gates
  (≥k channels ≥0.5), threshold sweeps, and a logistic combiner.
  The logistic combiner reached holdout F1 0.855 but was rejected in
  favor of the simpler reweight+recalibrate (F1 0.844) — no learned
  artifact to version or audit.

## 3. Change

`CHANNEL_PROFILES['v2_0']` in `core/detection_engine.py`:

| Channel | Old | New |
|---|---|---|
| ml | 0.25 | 0.05 |
| rules | 0.25 | 0.15 |
| behavioral | 0.15 | 0.35 |
| session | 0.20 | 0.30 |
| sequence | 0.15 | 0.15 |

Detection thresholds (absolute `combined_score` cutoffs, applied via
`risk_level`): **medium=0.20, high=0.30, critical=0.45**, derived from
pooled-validation operating points — medium = recall≥0.95 band,
high = F1-max/precision≥0.90 point, critical = precision plateau.

`scripts/calibrate_v2_detection.py` gained `--detection-thresholds
MEDIUM,HIGH,CRITICAL` for explicit operating points (percentile
derivation remains the default); the manifest records
`detection_thresholds_source`. New artifact:
`cal-20261112T055222Z-2962264f` (same detector fingerprint, verified
`calibrated` on load).

## 4. Results — independent_v2 holdout (n=1500), post-change

Evaluated once via `scripts/evaluate_independent_detection.py`
(new repeatable script; Phase 4.3 used an uncommitted temp script).
Full machine-readable output: `reports/independent_detection_evaluation_v2.json`.

| Metric | Phase 4.3 (v1.0) | Phase 4.6 |
|---|---|---|
| TP / FP / TN / FN | 409 / 414 / 651 / 26 | 351 / 46 / 1019 / 84 |
| Precision | 0.497 | **0.884** |
| Recall | 0.940 | 0.807 |
| F1 | 0.650 | **0.844** |
| FPR | 0.389 | **0.043** |
| ROC-AUC / PR-AUC | 0.922 / 0.833 | 0.929 / 0.836 |

Entity-cohort slices (consistent — not a memorization artifact):

| Slice | P | R | F1 | FPR |
|---|---|---|---|---|
| holdout_only_users (n=350, user-disjoint) | 0.916 | 0.831 | 0.871 | 0.039 |
| shared_users (n=1150) | 0.872 | 0.798 | 0.834 | 0.044 |

Internal entity-disjoint holdout (n=133): P=1.000, R=0.886, F1=0.939,
FPR=0.000 at the same operating point — internal metrics remain strong.

Pooled holdout (internal + independent_v2, n=1633): P=0.893, R=0.813,
F1=0.851, FPR=0.040.

## 5. Honest caveats

- **Second look at the holdout.** Phase 4.3 consumed the holdout once;
  this phase evaluated it again after a validation-driven change. The
  improvement is large and consistent across cohorts, but the holdout
  is no longer "untouched" — treat v2 numbers as post-hoc evidence.
- **Still synthetic OOD data.** No real-world claims; the absolute
  numbers bound performance on *similar* synthetic traffic only.
- **Recall traded for precision** (0.94 → 0.81). This is the intended
  operating point — analyst workload was the dominant cost — but
  high-recall mode remains available by lowering `high` toward
  `medium` (0.20 gives recall 0.95 at precision 0.69 on validation).
- **Thresholds are operating-point constants**, not probabilities;
  per-deployment retuning is not supported (unchanged limitation).
- `reports/fusion_evaluation.json` regenerated under the new profile;
  it uses uncalibrated default engine thresholds, so its absolute
  numbers differ from the calibrated production operating point.

## 6. Verification

- `pytest`: 573 passed.
- Calibration artifact loads with fingerprint verification
  (`cal-20261112T055222Z-2962264f`, status `calibrated`).
- Batch/stream parity unaffected — streaming uses the same
  `DetectionEngine` + artifact path.
