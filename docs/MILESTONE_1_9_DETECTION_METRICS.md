# CAID Milestone 1.9 — Detection Metrics (Independent Holdout)

**Date:** 2026-11-12 · Data: `data/independent/holdout.csv` (n=1500;
1073 normal / 293 suspicious / 134 high_risk) · Positives (binary mode):
suspicious+high_risk = 427; (high-risk mode): high_risk = 134.

> **Synthetic labeled data only. These are NOT real-world detection
> accuracy figures. `anomaly_score_normalized`, `risk_score` and
> `combined_score` are calibrated anomaly scores, NOT probabilities.**

Pipeline: v1_9 `AnomalyDetector` (fit on train normals) → calibrated
`RiskEngine` (thresholds from calibration normals only) →
`DetectionRulesEngine` → `BehavioralAnalyzer` (v1_9 baselines) →
`DetectionEngine`. Operating mode on all 1500 rows:
`full_ml_rules_behavioral`.

Two decision views are reported: `detection_risk_level` (combined-score
classification: binary pos = High/Critical, high-risk pos = Critical) and
`is_anomaly_flag` (raw IsolationForest flag).

## Overall — holdout

### Binary mode (suspicious+high_risk = positive)

| View | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Specificity | ROC-AUC | PR-AUC |
|------|---:|---:|---:|---:|----------:|-------:|---:|----:|------------:|--------:|-------:|
| detection_risk_level | 175 | 0 | 1073 | 252 | 1.000 | 0.410 | 0.581 | 0.000 | 1.000 | 0.9177 | 0.8065 |
| is_anomaly_flag | 295 | 96 | 977 | 132 | 0.755 | 0.691 | 0.721 | 0.090 | 0.911 | 0.9177 | 0.8065 |

### High-risk mode (high_risk = positive)

| View | TP | FP | TN | FN | Precision | Recall | F1 | FPR | Specificity | ROC-AUC | PR-AUC |
|------|---:|---:|---:|---:|----------:|-------:|---:|----:|------------:|--------:|-------:|
| detection_risk_level | 79 | 0 | 1366 | 55 | 1.000 | 0.590 | 0.742 | 0.000 | 1.000 | 0.9581 | 0.6236 |
| is_anomaly_flag | 129 | 262 | 1104 | 5 | 0.330 | 0.963 | 0.491 | 0.192 | 0.808 | 0.9581 | 0.6236 |

The detection layer is deliberately conservative (precision 1.000, FPR
0.000) at the cost of recall (~0.41 binary / ~0.59 high-risk); the raw IF
flag is more sensitive (recall 0.69 / 0.96) with measurable FPR. AUC/PR-AUC
are identical across views because they rank by the same
`anomaly_score_normalized`.

## Per-scenario — binary mode, detection_risk_level

| Scenario | n | pos | Precision | Recall | F1 | FPR | Spec | ROC-AUC | PR-AUC |
|----------|--:|----:|----------:|-------:|---:|----:|-----:|--------:|-------:|
| all_holdout | 1500 | 427 | 1.000 | 0.410 | 0.581 | 0.000 | 1.000 | 0.9177 | 0.8065 |
| known users + known IPs | 945 | 264 | 1.000 | 0.417 | 0.588 | 0.000 | 1.000 | 0.9217 | 0.7992 |
| unseen users + known IPs | 404 | 122 | 1.000 | 0.377 | 0.548 | 0.000 | 1.000 | 0.9602 | 0.8946 |
| known users + unseen IPs | 109 | 29 | 1.000 | 0.483 | 0.651 | 0.000 | 1.000 | 0.9431 | 0.8427 |
| unseen users + unseen IPs | 42 | 12 | 1.000 | 0.417 | 0.588 | 0.000 | 1.000 | 0.9500 | 0.8858 |
| new event-type combinations | 71 | 20 | 1.000 | 0.500 | 0.667 | 0.000 | 1.000 | 0.9010 | 0.7895 |
| normal + unusual hours | 44 | 0 | — | — | — | 0.000 | 1.000 | n/a | n/a |
| suspicious + common IPs | 269 | 269 | 1.000 | 0.156 | 0.270 | n/a | n/a | n/a | n/a |
| high_risk + seen users | 94 | 94 | 1.000 | 0.989 | 0.995 | n/a | n/a | n/a | n/a |
| normal distribution shift | 55 | 0 | — | — | — | 0.000 | 1.000 | n/a | n/a |
| multiple anomalies / session | 825 | 306 | 1.000 | 0.487 | 0.655 | 0.000 | 1.000 | 0.9102 | 0.8460 |

High-risk-mode scenario detail is in `eval_results.json` (e.g.
high_risk+seen_users recall 0.553 at Critical; unseen_users+known_ips
recall 0.667).

### Scenario observations (honest)

- **No false positives anywhere**: FPR = 0.000 on every scenario with
  normal rows, including unusual-hours normals and distribution-shifted
  normals — the Milestone 1.8.1 FP fix holds on independent entities.
- **Unseen users do not degrade AUC** (0.96 on unseen_users+known_ips) —
  typical-entity fallbacks work as designed.
- **Weakest scenario**: `suspicious_common_ips` recall 0.156 — suspicious
  events on shared/common IPs during business hours are genuinely hard;
  this is the realistic overlap cost, not a leakage artifact.
- `high_risk_seen_users` recall 0.989 — persistent per-user behavioral
  baselines catch repeat high-risk users effectively.

## Per-user / per-IP

`eval_results.json` → `per_user` / `per_ip`: entity score = max event
`anomaly_score_normalized`; entity prediction = any event High/Critical
(detection) or any `is_anomaly` (IF flag); entity label = modal label.
200 users / 160 IPs evaluated in each mode. Mixed-label users are kept
with modal label (documented limitation).

## Score distributions by label (holdout, means)

| Label | anomaly_score_normalized | risk_score | behavioral_deviation | combined_score |
|-------|-------------------------:|-----------:|---------------------:|---------------:|
| normal | 0.277 | 0.113 | 0.004 | 0.110 |
| suspicious | 0.654 | 0.349 | 0.471 | 0.427 |
| high_risk | 0.941 | 0.828 | 0.943 | 0.739 |

Clean ordinal separation normal < suspicious < high_risk on every channel.
Full per-label stats (mean/median/std/min/max/p25/p75) in
`eval_results.json → score_distributions`.

## Validation-split sanity (not tuned on)

Binary detection: P=1.000 R=0.463 FPR=0.000 AUC=0.9312;
high-risk detection: P=1.000 R=0.638 FPR=0.000 AUC=0.9654 — consistent
with holdout, confirming results are not holdout-specific.

## Comparison to Milestone 1.8.1 (different data — not comparable)

1.8.1 reported suspicious recall 1.000 / precision 0.893 / ROC-AUC 1.000
on the legacy leaky evaluation set. On independent entity-disjoint data
the honest numbers are materially lower (binary recall 0.41–0.69
depending on view, ROC-AUC ~0.92). **This is the expected and correct
outcome of removing label-correlated identifiers** — the earlier metrics
were inflated by IP-prefix and generator-pattern leakage.
