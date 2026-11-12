# CAID Milestone 1.9 — Dataset Split Report

**Date:** 2026-11-12 · Builds on Milestone 1.8.1 · Synthetic defensive research only.

## Generator

`data/generate_independent_datasets.py` (seed `1900`, `numpy.default_rng`).

Design guarantees:

- **Labels are per-event**, drawn from each user's behavioral propensity
  blended with split-level label mixtures. Labels never determine user IDs,
  IP addresses, IP prefixes, or row order.
- **Entity cohorts**: `shared` users/IPs appear in every split;
  `holdout_only` users/IPs appear exclusively in `holdout.csv`
  (sidecar: `data/independent/entity_cohorts.json`).
- **IP prefixes carry no label**: all three RFC 5737 TEST-NET ranges are
  used by every label in every split.
- **Behavioral overlap**: suspicious events occur during business hours
  from common IPs; normal events occur after hours and occasionally use
  sensitive event types; high-risk reuses previously-seen users/IPs.
- Timestamps share one 30-day window across all splits (no temporal split
  artifact). `after_hours` is derived from the timestamp hour.

## Splits (`data/independent/`)

| Split | File | Rows | Users | IPs | Label counts (n/s/hr) |
|-------|------|-----:|------:|----:|------------------------|
| train | train.csv | 3,000 | 140 | 120 | 2413 / 389 / 198 |
| calibration | calibration.csv | 800 | 140 | 120 | 725 / 48 / 27 |
| validation | validation.csv | 800 | 140 | 120 | 599 / 132 / 69 |
| holdout | holdout.csv | 1,500 | 200 | 159 | 1073 / 293 / 134 |

(Exact counts in `dataset_manifest.json`; n=normal, s=suspicious,
hr=high_risk.)

## Entity overlap (from manifest)

| Comparison | Shared users | Shared IPs | holdout-only users present | holdout-only IPs present |
|------------|-------------:|-----------:|---------------------------:|-------------------------:|
| train vs calibration | 140 | 120 | 0 | 0 |
| train vs validation | 140 | 120 | 0 | 0 |
| train vs holdout | 140 | 120 | 60 | 39 |

Holdout contains 60 unseen users and 39 unseen IPs not present in train —
the basis for generalization scenarios 2–4.

## Usage contract

| Split | Permitted use |
|-------|---------------|
| train | Fit `AnomalyDetector` + `BehavioralAnalyzer` (normal-labeled rows only) |
| calibration | `RiskEngine.calibrate_thresholds` (normal-labeled rows only, 725 rows) |
| validation | Sanity metrics only — never tuned on |
| holdout | ALL reported Milestone 1.9 metrics — never used for fitting, feature fitting, calibration, or threshold tuning |

## Quality checks (manifest `quality` block)

- Exact duplicate rows across all splits: **0**
- Missing required fields: **0**
- Labels: `normal`, `suspicious`, `high_risk` only

## Label-independence evidence (manifest `label_crosstabs`, holdout)

- `label × ip_prefix`: all three prefixes appear under all three labels
  (e.g. high_risk: 42/49/43 across prefixes vs normal 390/365/330) —
  NMI(prefix, label) ≈ 0 (verified by `scripts/leakage_audit_1_9.py`).
- `label × hour_band`: every label spans night/morning/afternoon/evening.
- `label × event_type`: sensitive types (`grade_download`,
  `student_export`, `course_export`, `bulk_download`) appear under all
  labels; holdout-only types (`lti_launch`, `mastery_check`, `wiki_review`)
  appear under all labels in holdout only.

## Artifacts produced

- `data/independent/{train,calibration,validation,holdout}.csv`
- `data/independent/dataset_manifest.json`
- `data/independent/entity_cohorts.json`
- `data/independent/leakage_audit.json` (Phase 2 output)
- `data/independent/eval_results.json` (Phase 6 output)
- `models/saved_models/v1_9/anomaly_detector.joblib`
- `models/saved_models/v1_9/behavioral_baselines.joblib`
- `models/saved_models/calibration/thresholds_<fingerprint>.joblib`
