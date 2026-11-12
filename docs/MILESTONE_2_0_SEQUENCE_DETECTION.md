# CAID Milestone 2.0 — Sequence Detection

## Module

`core/sequence_detection.py` — `SequenceConfig`, `SequenceAnalyzer`, persisted `sequence_model.joblib`.

## Approach

Interpretable statistical sequence analysis only (no deep learning in v2.0):

- Per-user/global event-type transition frequencies with smoothing.
- Causal window per session (configurable `window_seconds`, `max_window_events`).
- Session-prefix event-rate and sensitive-operation deviations.
- Configurable pattern definitions evaluated on the causal window only.

Default patterns (configurable, not label/IP hard-coded):
- `login_failure_burst_then_access` — repeated auth failures then success/sensitive access.
- `sensitive_operation_burst` — multiple sensitive operations in-window.
- `export_after_broad_access` — sensitive/export after broad course/account access.
- `ip_switch_burst` — multiple source-IP switches in-window.

## Score contract

`sequence_deviation_score` ∈ [0,1], higher = more anomalous; **not a probability**. Missing inputs → NaN + `sequence_score_missing`. Components: transition rarity, event-rate deviation, sensitive-sequence deviation, pattern match. `sequence_baseline_scope` reports `user`/`global`/`insufficient`.

Outputs include `sequence_factors`, `sequence_evidence`, `sequence_triggered_patterns` for explainability.

## Leakage rules

- Fit only on train-normal data.
- Scoring uses only events up to and including the current event within the same session/window.
- Missing `auth_result` makes auth-dependent patterns inactive rather than fabricating failures/successes.
- Duplicates may be ignored from transition windows via config.
