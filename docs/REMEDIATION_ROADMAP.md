# CAID Remediation / Improvement Roadmap

## Milestone 2.0 status (implemented in this workspace)

- Session reconstruction with explicit/timeout session IDs, missing-timestamp isolation, duplicate flags, causal prefix features, and completed-session summaries.
- Persisted `SessionBaselineManager` / `SessionAnalyzer` trained on train-normal data only.
- Configurable `SequenceAnalyzer` with transition rarity, rate/sensitive deviations, and pattern definitions.
- `DetectionEngine` channel profiles: `legacy` (unchanged), `session`, `sequence`, `v2_0`; missing channels excluded and weights renormalized.
- Calibration artifact v2 with fingerprints for detector, feature/behavioral/session baselines, and sequence model; detection-threshold calibration added.
- Optional `SessionAwareAnomalyDetector` (artifact v3) as a separate ML feature-schema experiment.
- New `data/independent_v2/` generator with session_id/auth_result optional fields, entity cohorts, behavior phases, and leakage audit.
- `scripts/evaluate_holdout_2_0.py`, `scripts/leakage_audit_2_0.py`, dashboard page `Session Intelligence`.

## Known limitations / next steps

- Evaluation and performance numbers must be read from generated `data/independent_v2/eval_results_v2.json` and `performance_results_v2.json`; do not quote unrun results.
- Channel weights and detection thresholds are fixed/versioned, not tuned on holdout; consider calibration-only weight selection only if a separate calibration objective is defined.
- Sequence patterns are statistical/configurable; richer temporal models or deep sequence models are deferred until simpler methods are fully evaluated.
- Production streaming would require an incremental session store; current implementation is batch/casual-prefix over uploaded or split files.
- Real Canvas integration is out of scope; all claims remain synthetic-only.
