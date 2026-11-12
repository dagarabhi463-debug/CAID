# CAID Milestone 2.0.1 — Channel Ablation Report

Evaluation code: `scripts/milestone_2_0_1_audit.py`; results: `data/independent_v2/audit_2_0_1/milestone_2_0_1_audit.json`.
Protocol: identical v2 holdout population, labels (`suspicious`+`high_risk` positive), DetectionEngine `risk_level` High/Critical as alert positive, `combined_score` as ranking score. No holdout tuning.

## Default detection thresholds (apples-to-apples)

| Variant | Precision | Recall | F1 | FPR | Specificity | ROC-AUC | PR-AUC | ML ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1_9_baseline (A reference) | 0.8893 | 0.4989 | 0.6392 | 0.0254 | 0.9746 | 0.8809 | 0.8011 | 0.8779 |
| A: v2_base ML+rules+behavioral | 0.8899 | 0.4644 | 0.6103 | 0.0235 | 0.9765 | 0.8522 | 0.7669 | 0.8281 |
| B: + session | 0.8864 | 0.3586 | 0.5106 | 0.0188 | 0.9812 | 0.9036 | 0.8057 | 0.8281 |
| C: + sequence | 0.8930 | 0.4989 | 0.6401 | 0.0244 | 0.9756 | 0.8920 | 0.7917 | 0.8281 |
| D/F: + session + sequence (v2_0) | 0.8868 | 0.4322 | 0.5811 | 0.0225 | 0.9775 | 0.9177 | 0.8224 | 0.8281 |
| E: session-aware ML complete | 0.9026 | 0.4046 | 0.5587 | 0.0178 | 0.9822 | 0.9334 | 0.8431 | 0.9134 |

## Calibrated-threshold sensitivity (calibration-normal percentiles, not holdout-tuned)

| Variant | Precision | Recall | F1 | FPR | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1_9_baseline | 0.5171 | 0.8690 | 0.6484 | 0.3315 | 0.6685 | 0.8809 | 0.8011 |
| v2_base | 0.4629 | 0.8598 | 0.6018 | 0.4075 | 0.5925 | 0.8522 | 0.7669 |
| v2_session | 0.4800 | 0.9379 | 0.6350 | 0.4150 | 0.5850 | 0.9036 | 0.8057 |
| v2_sequence | 0.4789 | 0.9126 | 0.6282 | 0.4056 | 0.5944 | 0.8920 | 0.7917 |
| v2_complete | 0.4807 | 0.9448 | 0.6372 | 0.4169 | 0.5831 | 0.9177 | 0.8224 |
| v2_session_aware_complete | 0.5080 | 0.9540 | 0.6629 | 0.3775 | 0.6225 | 0.9334 | 0.8431 |

Calibrated thresholds raise recall but FPR rises to ~0.33-0.42; this is an operating-point tradeoff, not a free improvement.

## Key scenarios (default thresholds)

| Scenario | v1_9 F1/ROC | v2_sequence F1/ROC | v2_complete F1/ROC | session-aware F1/ROC |
|---|---:|---:|---:|---:|
| common_ip_positives | 0.6539 / n/a* | 0.6596 / n/a* | 0.6187 / n/a* | 0.5880 / n/a* |
| unseen_users | 0.6738 / 0.9018 | 0.6595 / 0.9342 | 0.5412 / 0.9433 | 0.5301 / 0.9527 |
| normal_to_suspicious | 0.6242 / 0.7863 | 0.6271 / 0.8299 | 0.5657 / 0.8445 | 0.5421 / 0.8515 |
| suspicious_to_normal | 0.7692 / 0.9822 | 0.6667 / 0.9544 | 0.5882 / 0.9891 | 0.5161 / 0.9970 |

*n/a because the subset contains positives only, so ROC-AUC is undefined.

## Findings

- Session-only (B) improves ranking (ROC-AUC 0.9036 vs v2_base 0.8522) but sharply reduces recall/F1 under default thresholds.
- Sequence-only (C) is the only v2 variant that matches/slightly exceeds v1.9 F1 (0.6401 vs 0.6392) while keeping precision/FPR comparable; its ROC-AUC is higher (0.8920 vs 0.8809).
- Complete v2_0 (D) has the best default-threshold ROC-AUC/PR-AUC among non-session-aware variants (0.9177/0.8224) but lower F1/recall than v1.9 and sequence-only.
- Session-aware ML (E) gives the strongest ranking (ROC-AUC 0.9334, PR-AUC 0.8431, ML ROC-AUC 0.9134) but not the best default-threshold F1; under calibrated thresholds it has the highest F1 (0.6629) at high FPR.
- Common-IP suspicious positives remain precision=1.0 for all variants; recall differences are modest and sequence-only is best among v2 variants.
