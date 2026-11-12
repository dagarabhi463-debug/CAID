# CAID Milestone 2.0 — Architecture Audit

## Scope audited

Repository: `C:\Users\dagar\OneDrive\Documents\web-project` (not a Git repo). Python/Streamlit synthetic defensive research prototype. Audited before Milestone 2.0 code changes using filesystem inspection, source reads, existing reports, and tests.

## Existing pipeline (v1.9)

Batch-oriented flow:

```
CSV -> utils/validators + core/log_processor
    -> models/anomaly_detector.AnomalyDetector.predict
    -> core/risk_engine.RiskEngine.process_events
    -> core/detection_rules.DetectionRulesEngine.process_events
    -> core/behavioural_analyzer.BehavioralAnalyzer.analyze_user_behavior
    -> core/detection_engine.DetectionEngine.process_dataframe
    -> modules/dashboard.py / modules/threat_detection.py / modules/export.py
```

Evaluation path (`scripts/evaluate_holdout_1_9.py`) fits detector and behavioral artifacts on train normals only, calibrates on calibration normals only, and reports holdout metrics without holdout tuning.

## Contracts preserved

- `anomaly_score_raw`: IsolationForest `decision_function`, higher = more normal.
- `anomaly_score_normalized` / `ml_anomaly_score`: calibrated 0–1, higher = more anomalous.
- `rule_score` / `rule_detection_score`: 0–1 rule evidence.
- `behavioral_deviation_score`: 0–1 deviation or NaN + `behavioral_score_missing`.
- `risk_score` / `combined_score`: 0–1 higher = riskier.
- Missing channels are explicit and excluded; weights renormalize over available channels; fewer than two evidence channels → `insufficient_evidence` and NaN score.
- Artifacts are joblib, versioned, schema-validated, and fingerprint-bound for calibration.
- v1.9 artifacts under `models/saved_models/v1_9/` and `data/independent/` remain untouched.

## Key modules and ownership

| Area | Module | Role |
|---|---|---|
| Timestamp/time features | `core/feature_engineering.py` | hour/day/weekend/after-hours |
| Event/user/IP/event baselines | `core/feature_engineering.py::FeatureBaselineManager` | persisted inside detector artifact |
| Behavioral baselines | `core/behavioral_baseline.py`, `core/behavioural_analyzer.py` | per-user/global stats, NaN missing contract |
| Rules | `core/detection_rules.py` | safe configurable AST rule engine |
| Risk | `core/risk_engine.py` | heuristic risk score + calibrated fixed thresholds |
| Detection | `core/detection_engine.py` | channel weighting, operating modes, evidence |
| Calibration | `core/calibration.py` | versioned artifacts, detector fingerprint, role guard |
| ML artifact | `models/anomaly_detector.py` | IsolationForest v2 artifact |
| Logs | `core/log_processor.py`, `utils/validators.py` | load/validate/clean/timeline |
| Independent data | `data/generate_independent_datasets.py` | v1.9 train/calibration/validation/holdout |
| Dashboard | `app.py`, `modules/*.py` | Streamlit UI |

## Gaps found during audit

1. The pipeline had no session reconstruction or sequence intelligence; individual-event channels could miss subtle multi-event behavior.
2. `RiskEngine.calibrate_thresholds` calibrates `risk_score` thresholds, but `DetectionEngine` had fixed `combined_score` thresholds. Milestone 2.0 adds explicit detection-threshold calibration for v2 while preserving legacy defaults.
3. `utils/validators.py` allowed only `label` as optional; independent generators already used event types outside `VALID_EVENT_TYPES`. Milestone 2.0 adds optional `session_id`/`auth_result` and expands the synthetic event-type list.
4. No causal session-prefix features existed; completed-session summaries alone would leak future events into earlier alerts.

## Milestone 2.0 dependency map

```
raw CSV
  -> session_reconstruction (session_id + causal prefix features + summaries)
  -> [optional SessionAwareAnomalyDetector] AnomalyDetector.predict
  -> RiskEngine -> DetectionRulesEngine -> BehavioralAnalyzer
  -> SessionAnalyzer -> SequenceAnalyzer
  -> DetectionEngine(channel_profile)
  -> dashboard/export/evaluation
```

Baselines fit only on train-normal data; thresholds calibrate only on calibration-normal data; validation is sanity-only; holdout is report-only.

## Risks and mitigations

- **Future leakage:** per-event scores use causal prefix/window features only; completed-session summaries are dashboard/session-level eval only.
- **Missing evidence:** new channels emit NaN + missing flags; DetectionEngine excludes and renormalizes.
- **Schema drift:** optional `session_id`/`auth_result` are validated as optional; absence degrades sequence evidence explicitly.
- **Artifact compatibility:** `AnomalyDetector` remains artifact v2; only `SessionAwareAnomalyDetector` uses v3. Calibration accepts v1 and writes v2.
- **Synthetic scope:** all data remains synthetic/defensive; docs and UI state scores are not probabilities and no real-world accuracy is claimed.
