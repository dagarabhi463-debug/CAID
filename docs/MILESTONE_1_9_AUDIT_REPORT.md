# CAID Milestone 1.9 — Audit Report

**Date:** 2026-11-12 · Builds on Milestone 1.8.1 · CAID remains a
**synthetic defensive research prototype**. No production-readiness claim.

## Phase 1 — Independent dataset generation: DONE

New `data/generate_independent_datasets.py` produces four seeded splits
(`data/independent/`): train 3000 / calibration 800 / validation 800 /
holdout 1500 rows. Entity-disjoint cohorts (140 shared + 60 holdout-only
users; 120 shared + ~40 holdout-only IPs), per-event labels, overlapping
behavioral distributions, no label→identifier mapping. Seeds, sizes and
overlap matrices recorded in `dataset_manifest.json`;
`entity_cohorts.json` sidecar maps entities to cohorts. See
`MILESTONE_1_9_DATASET_SPLIT_REPORT.md`.

## Phase 2 — Leakage audit: DONE

`scripts/leakage_audit_1_9.py` → `leakage_audit.json` +
`MILESTONE_1_9_LEAKAGE_AUDIT.md`. Confirmed legacy leaks (IP-prefix label
encoding, `suspicious_ip_class` rule, hour/token-age separation,
exclusive sensitive event types) and verified all independent-dataset
channels clean (prefix NMI≈0, no ordering/duplicate/entity leakage,
`label` never reaches features).

## Phase 3 — Generalization testing: DONE

`scripts/evaluate_holdout_1_9.py` evaluates 10 scenarios + overall on the
fixed holdout. Headline (binary, detection view): precision 1.000,
recall 0.410, FPR 0.000, ROC-AUC 0.9177; unseen-user and unseen-IP
subsets hold AUC ≥0.94; weakest scenario `suspicious_common_ips`
(recall 0.156 — genuine overlap difficulty). Full table in
`MILESTONE_1_9_DETECTION_METRICS.md`.

## Phase 4 — Calibration integration: DONE

`core/calibration.py` + `scripts/calibrate_thresholds.py` + wiring in
`app.py` and `modules/log_analysis.py`. Thresholds are versioned
(`cal-20261111T210735Z-e7935b64`), fingerprint-bound to the detector,
calibrated on calibration normals only (n=725), and the app loads them
via `apply_to_engine` with explicit `calibration_status`
(`calibrated` vs `uncalibrated_dataset_relative`). The calibrate CLI
refuses non-calibration roles; detector fingerprints are content-bound so
production and v1_9 artifacts each carry their own thresholds. See
`MILESTONE_1_9_CALIBRATION_REPORT.md`.

## Phase 5 — Missing-ML policy: DONE

`OperatingMode` in `detection_engine.py`: full / three degraded variants
/ `insufficient_evidence` (<2 channels with real evidence → NaN score,
`RiskLevel.INSUFFICIENT`, no classification). Neutral placeholders are
excluded and weights renormalized — never scored as observations.
`ml_validated` flag added to risk factors. See
`MILESTONE_1_9_MISSING_ML_POLICY.md`.

## Phase 6 — Detection performance: DONE

All required metrics computed on holdout only (never used for
fit/calibration/tuning): confusion matrix, precision, recall, F1, FPR,
specificity, ROC-AUC, PR-AUC — overall, per-scenario, per-user (200),
per-IP (160), plus score distributions by label. Scores explicitly not
probabilities. Validation split used as sanity only.

## Phase 7 — Explainability: DONE (additive)

`Detection.evidence` + `structured_evidence` column: per-channel objects
with channel, evidence_name, observed_value, expected baseline/threshold,
contribution/severity, availability, explanation. Legacy `explanation`
strings preserved; missing channels marked `excluded`.

## Phase 8 — Regression/artifact validation: DONE

- Artifact v2 loads; wrong `artifact_version` and missing keys fail
  explicitly (tested).
- Train/predict feature-schema equality enforced (`predict` raises on
  mismatch; tested).
- LOO training z-scores deterministic; baselines immutable during
  prediction (existing + new tests).
- Calibration metadata survives save/load; fingerprint mismatch rejected
  (tested).
- Predictions identical after save→load round-trip (tested).
- **Full suite: 327 passed, 0 failed** (was 301; +26 Milestone-1.9 tests,
  5 pre-existing tests updated for the new operating-mode contract, 1
  pre-existing pandas-3.x CSV-sanitization bug fixed in `modules/export.py`).

## Defects found & fixed this milestone

| ID | Defect | Fix |
|----|--------|-----|
| B1 | Legacy IP prefixes encoded labels; `suspicious_ip_class` rule consumed them | Rule removed from defaults; independent generator decorrelates prefixes |
| B2 | `calibrate_thresholds()` never called by the app | `core/calibration.py` + lifecycle wiring + role-guarded CLI |
| B3 | Missing ML → neutral 0.5 scored as real observation | Operating modes; channel exclusion + renormalization; insufficient-evidence mode |
| B4 | No user/IP-disjoint evaluation data | Entity-cohort generator + 4 splits + manifest |
| B5 | pandas 3.x `str` dtype bypassed CSV formula-injection sanitizer | `modules/export.py` dtype check widened (pre-existing env failure) |
| B6 | IP cohort pools could collide across cohorts | global exclusion set in generator |

## Honest headline result

On independent, entity-disjoint, label-decorrelated data the system
achieves **precision 1.000 / FPR 0.000** with binary recall ≈0.41
(detection view) and ROC-AUC ≈0.92 — versus the leaky 1.000/0.893/1.000
figures from Milestone 1.8.1. The drop is the expected cost of removing
label-correlated identifiers, not a regression.

## Remaining limitations & remediation roadmap

1. **Recall on subtle suspicious behavior is limited** (suspicious +
   common IPs recall 0.156) — consider richer session-level features or
   sequential models; do NOT tune on holdout.
2. `user_total_accounts` zero-variance column still in feature schema
   (deferred since 1.8; removal needs artifact v3 migration).
3. Per-user/per-IP metrics use modal label for mixed-label entities —
   documented approximation.
4. `ip_class`/`is_sensitive_event` remain engineered (unused by default
   rules for ip_class) — schema cleanup candidate.
5. App UI does not yet display `operating_mode`/calibration status
   prominently — surfaced via session state only.
6. Metrics remain synthetic-only; real-world validation requires
   non-synthetic data (out of scope).
7. Consider chronological/streaming evaluation path
   (`evaluate_chronological` exists) as a future milestone.
