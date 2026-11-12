# CAID Milestone 2.0 — Session Engine

## Module

`core/session_reconstruction.py` — deterministic session reconstruction and feature extraction.
`core/session_baseline.py` — persisted training-only `SessionBaselineManager`.
`core/session_analyzer.py` — thin inference wrapper (`SessionAnalyzer`).

## Session identity

- Explicit non-empty `session_id` wins when present; IDs are scoped by `user_id` (`explicit:<user>:<session_id>`).
- Otherwise per-user inactivity timeout grouping (`timeout:<user>:<start>:<ordinal>`), default 1800 seconds, configurable via `SessionConfig`.
- Missing timestamps are isolated (`missing_ts:<user>:<pos>`) with `session_timestamp_missing=True`; no duration/rate is fabricated.
- Duplicates are preserved and flagged by deterministic content hash over scalar event fields (`is_duplicate_event`).
- Sorting is chronological per user/session with deterministic tie-breaks; original records are preserved.

## Feature views

Causal prefix features (per event, past/current only):
`session_prefix_event_index`, `session_prefix_event_count`, `session_prefix_duration_seconds`, `session_prefix_event_rate`, `session_prefix_unique_event_types`, `session_prefix_api_requests`, `session_prefix_data_volume_mb`, `session_prefix_unique_ips`, `session_prefix_ip_switch_count`, `session_prefix_sensitive_event_count`, `session_prefix_after_hours_count`, `session_prefix_auth_failures`, `session_prefix_auth_successes`, `session_prefix_seconds_since_previous`, plus event/transition context fields.

Completed-session summaries (retrospective only): `session_event_count`, `session_unique_event_types`, `session_api_requests`, `session_data_volume_mb`, `session_unique_ips`, `session_sensitive_event_count`, `session_after_hours_count`, `session_ip_switch_count`, `session_duration_seconds`, `session_event_rate`, `session_auth_failures`, `session_auth_successes`.

## Baseline contract

- `SessionBaselineManager.fit()` reconstructs sessions on training-normal data only and stores per-user/global mean/std for canonical session features plus completed-session stats.
- `score_row()`/`transform_events()` score causal prefix vectors; `score_session_row()`/`transform_sessions()` score completed summaries.
- Unseen users fall back to global stats; insufficient inputs → NaN + missing; zero variance uses safe std; prediction never mutates fitted state.
- `save()`/`load()` are deterministic joblib artifacts with artifact version, feature schema, config, thresholds, and fit counts; `fingerprint()` supports calibration binding.

## Score contract

`session_deviation_score` ∈ [0,1], higher = more deviation; NaN when missing. `session_score_missing` and `session_baseline_scope` (`user`/`global`/`insufficient`) are explicit.
