# Milestone 2.7 — Detection Quality Report

Date: 2026-11-12  
Scope: standalone defensive research prototype; synthetic data only.
Validation scope remains **internal reproducibility**, not independent validation.

## What changed

- Added `scripts/evaluate_detection_quality.py`, a reproducible harness that reports:
  - unseen-user holdout metrics on entity-disjoint users,
  - validation-only threshold selection applied once to holdout,
  - calibrated risk/detection status and threshold version,
  - score separation by label,
  - synthetic attack-scenario detection/FP metrics.
- Updated `scripts/calibrate_v2_detection.py` so v2 detection thresholds use a higher tail of the calibration-normal `combined_score` distribution:
  - risk thresholds unchanged: percentiles `(30, 65, 85)`,
  - detection thresholds changed from `(30, 65, 85)` to `(90, 99, 99.9)`.
- Regenerated `models/saved_models/calibration/v2_0_thresholds_0cde9fd80c5f6d2d-3ab7b78cf14b2f21.joblib` and `reports/calibration_v2_manifest.json`.

New calibrated detection thresholds:

| Level | Old threshold | New threshold |
|---|---:|---:|
| medium | 0.0588 | 0.1332 |
| high | 0.0805 | 0.2007 |
| critical | 0.1192 | 0.2321 |

## Results

Report: `reports/detection_quality.json`

### Unseen-user holdout

- Holdout rows/users: 133 rows / 125 users, disjoint from all training/calibration users.
- Score separation remains strong: ROC-AUC `1.0`, PR-AUC `0.9714`.
- Default calibrated operating point after recalibration:
  - precision `0.9211`, recall `1.0`, F1 `0.9589`, FPR `0.0306`.
- Validation-selected threshold (`0.327077`) applied to holdout:
  - precision `1.0`, recall `0.9714`, F1 `0.9855`, FPR `0.0`.
- Calibration status: `risk=calibrated`, `detection=calibrated`,
  `threshold_version=cal-20261112T035403Z-2962264f`.

### Synthetic scenarios — default High/Critical FPR before → after

| Scenario | Before FPR | After FPR |
|---|---:|---:|
| baseline | 0.8333 | 0.0467 |
| insider_exfiltration | 0.7879 | 0.1061 |
| credential_stuffing_ring | 0.7708 | 0.1250 |
| after_hours_lab | 0.8542 | 0.1667 |
| low_and_slow | 0.8261 | 0.1304 |
| mixed_campaign | 0.8095 | 0.1299 |

The recalibration materially reduced false-positive High/Critical classifications while preserving holdout recall. Scenario FPR remains nonzero because scenario normal traffic includes shared-IP/after-hours patterns that legitimately look anomalous under the current features.

## Verification

```powershell
.venv/Scripts/python.exe scripts/calibrate_v2_detection.py
.venv/Scripts/python.exe scripts/evaluate_detection_quality.py --out reports/detection_quality.json
.venv/Scripts/python.exe -m pytest tests/test_streaming_queue.py tests/test_streaming_ordering.py tests/test_streaming_duplicates.py tests/test_streaming_pipeline.py tests/test_streaming_health.py -q
```

Streaming parity/reliability tests: `28 passed`.

## Limitations

- Metrics are internal reproducibility on the current labeled CSVs and generated scenarios; independent validation is still not performed.
- The labeled corpus is small (~850 rows), so metrics are indicative rather than statistically powerful.
- Scenario FPR is still above zero; further reduction would require feature/label review or scenario-specific baseline work, not silent threshold tuning on holdout.
- The validation-selected threshold outperforms the calibrated default on holdout FPR, but it was selected on validation users only and is reported separately rather than auto-applied.
