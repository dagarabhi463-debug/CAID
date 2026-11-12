# CAID Milestone 2.0.1 — Causal Validation

Tests: `tests/test_mileestone_2_0_1_causal.py` — **9 passed**.

## Verified behaviors

- **Future-event exclusion:** `session_prefix_api_requests`/`session_prefix_event_count` at event *i* include only events 1..i in the same session.
- **Same-timestamp ordering:** identical timestamps are ordered by deterministic content hash, not input position; shuffled rows preserve per-event prefix indices.
- **Shuffled input determinism:** for unique timestamps, session IDs and prefix features are identical after row-order shuffling when aligned by event identity.
- **Timeout boundaries:** gap == timeout stays in-session; gap == timeout+1s starts a new session.
- **Missing timestamps:** isolated as `missing_ts:*`, `session_timestamp_missing=True`, prefix count 1, no fabricated transition.
- **Duplicate events:** preserved, flagged by scalar content hash, and excludable from sequence windows via config.
- **Multiple sessions per user:** explicit session IDs reset prefix counters per session.
- **Late-arriving events:** input order does not matter; chronological timestamp order determines prefix position.
- **Sequence truncation:** removing later events does not change earlier `sequence_deviation_score`, `sequence_triggered_patterns`, or `sequence_baseline_scope`.

## Interpretation

Per-event session and sequence scores are causal with respect to the reconstructed session. Completed-session summaries (`session_summaries`) are retrospective and are not used as causal per-event evidence.
