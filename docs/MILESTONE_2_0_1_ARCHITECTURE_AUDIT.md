# CAID Milestone 2.0.1 — Architecture & Code Audit

Scope: synthetic defensive research only. No real Canvas connectivity.

## Channel trace (raw event -> final risk)

1. `core/session_reconstruction.py` reconstructs deterministic `session_id` (explicit > timeout > isolated missing-ts), flags duplicates by scalar content hash, and emits causal `session_prefix_*` features. Prefix features include only current/prior events in the same session.
2. Optional `models/session_anomaly_detector.py` (`SessionAwareAnomalyDetector`, artifact v3) adds the same causal prefix features to the IsolationForest schema. It is a separate experiment artifact, not the default detector.
3. `models/anomaly_detector.py` emits `anomaly_score_raw` (higher=normal), `anomaly_score_normalized`/`ml_anomaly_score` (0-1, higher=anomalous), `ml_confidence`.
4. `core/risk_engine.py` computes `risk_score`/`risk_factors` and can apply fixed calibrated thresholds. Important: `RiskEngine.process_events()` calls `self.fit(df)` on the frame it processes, so its dataset-relative min/max are input-dependent unless `fixed_thresholds` are set. In the detection pipeline, `DetectionEngine.process_dataframe()` overwrites `risk_level`/`combined_score`; therefore the final alert classification is DetectionEngine output, not RiskEngine `risk_level`.
5. `core/detection_rules.py` emits `triggered_rules`, `rule_score`/`rule_detection_score` (0-1 evidence).
6. `core/behavioural_analyzer.py` emits `behavioral_deviation_score` plus `behavioral_score_missing`.
7. `core/session_analyzer.py` emits `session_deviation_score`, `session_score_missing`, `session_baseline_scope`, `session_evidence`/`session_factors`.
8. `core/sequence_detection.py` emits `sequence_deviation_score`, `sequence_score_missing`, `sequence_baseline_scope`, `sequence_factors`, `sequence_evidence`, `sequence_triggered_patterns`.
9. `core/detection_engine.py` fuses configured channels, renormalizes weights over available channels, emits `combined_score`, `risk_level`, `operating_mode`, `channels_available`, `missing_channels`, `structured_evidence`, `detection_explanation`.

## Score direction contracts verified

- ML normalized: higher = more anomalous.
- Rules/behavioral/session/sequence: 0-1 deviation/evidence, higher = more anomalous.
- `combined_score`: 0-1, higher = riskier; NaN only for insufficient evidence.
- Raw IsolationForest `decision_function` is inverted before use; `anomaly_score` remains a deprecated alias.

## Missing-channel behavior verified

- Missing session/sequence scores are NaN + explicit missing flags, never neutral placeholders.
- `DetectionEngine` excludes missing configured channels and renormalizes weights; fewer than two available channels -> `insufficient_evidence` and NaN combined score.
- Legacy profile ignores session/sequence columns even if present.

## Calibration/artifact dependencies verified

- v2 calibration artifact `models/saved_models/calibration/v2_0_thresholds_*.joblib` is artifact_version=2, dataset_role=calibration, channel_profile=v2_0, and fingerprint-bound to detector, behavioral baseline, session baseline, and sequence model. Audit fingerprint comparison: all non-None fingerprints match. `feature_baseline` fingerprint is currently None and therefore not independently verified (limitation).
- v1.9 artifacts under `models/saved_models/v1_9/` and `data/independent/` were not modified.
- v2.0 artifacts under `models/saved_models/v2_0/` include detector, behavioral, session, sequence, and the optional session-aware detector experiment.

## Issues found and fixed during audit

- `DetectionEngine` explanation/evidence labeled every non-legacy operating mode as "degraded", including `full_v2_0`. Fixed so full v2 profiles report full mode and only degraded/insufficient modes carry the degraded note.
- Same-timestamp session ordering previously depended on input row order. `_add_prefix_features` now uses a deterministic content-hash tie-break (excluding label/session metadata), making shuffled input deterministic for equal timestamps.
- Dashboard session page showed a generic calibration sentence and an unlabeled retrospective session summary. Patched to display actual risk/detection calibration status + threshold version and to label completed-session aggregates as retrospective, not causal alerts.

## Remaining limitations

- `feature_baseline` fingerprint is not populated by `component_fingerprint` for `FeatureBaselineManager`; calibration binding for that component is weaker than for detector/behavioral/session/sequence.
- Dashboard uses production/sample artifacts and default detection thresholds unless a compatible calibration artifact is explicitly applied; the v2.0 independent calibration artifact is bound to the v2.0 evaluation artifacts.
