# CAID Milestone 2.0.1 — Statistical Comparison

Method: paired bootstrap over the same v2 holdout events. Event bootstrap resamples rows (n=400); user-clustered bootstrap resamples users and includes all their events (n=200). Differences are `v2_variant - reference`. CIs are percentile 95%. Results: `data/independent_v2/audit_2_0_1/milestone_2_0_1_audit.json` → `bootstrap`.

## v1_9_baseline vs v2_complete (default thresholds)

| Metric | Event mean diff | Event 95% CI | User-clustered mean diff | User-clustered 95% CI | Meaningful? |
|---|---:|---:|---:|---:|---|
| F1 | -0.0584 | [-0.0860, -0.0301] | -0.0580 | [-0.0980, -0.0207] | v2_complete F1 significantly lower |
| ROC-AUC | +0.0363 | [0.0194, 0.0527] | +0.0366 | [0.0108, 0.0608] | v2_complete ranking significantly better |
| PR-AUC | +0.0207* | see JSON | +0.0215* | see JSON | ranking improvement |

*Exact PR-AUC intervals are in the JSON; direction is positive.

## v1_9_baseline vs v2_sequence (default thresholds)

| Metric | Event mean diff | Event 95% CI | User-clustered mean diff | User-clustered 95% CI | Meaningful? |
|---|---:|---:|---:|---:|---|
| F1 | +0.0009 | [-0.0186, 0.0194] | +0.0006 | [-0.0252, 0.0210] | not distinguishable from zero |
| ROC-AUC | +0.0106 | [-0.0036, 0.0243] | +0.0116 | [-0.0180, 0.0380] | not distinguishable from zero |

## v2_base vs v2_complete

- F1 diff (complete - base): event -0.0297 [-0.0515, -0.0060]; user -0.0300 [-0.0555, -0.0091] → complete F1 significantly lower.
- ROC-AUC diff: event +0.0649 [0.0491, 0.0809]; user +0.0637 [0.0398, 0.0885] → ranking improvement is significant.

## v2_session vs v2_complete and v2_sequence vs v2_complete

- complete vs session: F1 +0.0698 [0.0473, 0.0938] event / +0.0692 [0.0402, 0.0917] user → adding sequence to session significantly improves F1; ROC-AUC +0.0138/+0.0135 also significant.
- complete vs sequence: F1 -0.0593 [-0.0818, -0.0405] event / -0.0587 [-0.0924, -0.0334] user → sequence-only has significantly higher F1; ROC-AUC +0.0257/+0.0250 → complete has significantly better ranking.

## Interpretation

- The v2.0 complete pipeline's ranking improvement over v1.9 is statistically meaningful, but its default-threshold F1/recall regression is also meaningful.
- Sequence-only vs v1.9 differences are within sampling uncertainty; the earlier "F1 0.6401 vs 0.6392" should be treated as a tie, not a proven improvement.
- Session-aware ML has the best ranking metrics but its classification benefit depends heavily on threshold choice; calibrated thresholds improve F1 while sharply increasing FPR.
