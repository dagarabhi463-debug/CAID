# CAID Milestone 2.1 — Cross-Session Threat Correlation

## Scope

Retrospective correlation layer that links sessions, accounts, and
shared-IP activity across time, on top of the existing v2.0 pipeline
(session reconstruction, session/sequence deviation, DetectionEngine
risk levels and structured evidence). Synthetic/defensive research
data only — no real Canvas connectivity.

## Modules

- `core/cross_session_correlation.py` — `CrossSessionCorrelator` +
  `CorrelationConfig`.
- `modules/correlation_dashboard.py` — Streamlit page module.
- `app.py` — new "Cross-Session Correlation" navigation page.
- `tests/test_milestone_2_1_correlation.py` — correlation tests.

## Design contract

- **Retrospective only.** Correlation never feeds back into per-event
  causal scores, DetectionEngine channel weights, or risk levels. The
  v1.9 and v2.0 score contracts are unchanged.
- **Evidence-first.** Every finding carries `evidence`: a list of
  `{signal, observed_value, expected_baseline_or_threshold,
  explanation}` dicts naming the sessions/users/IPs involved.
- **Deterministic.** Finding IDs are `corr_<sha1[:12]>` over
  (type, sorted entities); output ordering is stable.
- **Session reuse.** If `session_id` is absent/empty, events are
  re-grouped with `SessionReconstructor` (explicit IDs win; otherwise
  per-user inactivity timeout) before correlating.

## Correlation types

| Type | Meaning | Key evidence |
|---|---|---|
| `repeated_pattern_across_sessions` | Same user triggers the same sequence pattern, or repeated elevated deviation, in ≥2 sessions | pattern name + session list |
| `similar_account_behavior` | Two suspicious accounts with cosine-similar aggregated behavioral profiles AND ≥1 shared triggered pattern | similarity score + shared patterns |
| `repeated_sensitive_operations` | Sensitive event types (grade/bulk export & download) on ≥2 distinct days | per-day counts |
| `long_term_behavior_change` | Early-vs-late window aggregate shift beyond threshold | before/after profile values |
| `shared_ip_activity` | ≥2 users on one `source_ip` where ≥1 account is suspicious | IP, user list, max score on IP |

## Shared-IP policy

Shared IP is **never sufficient evidence** of coordination:

- Only IPs involving at least one suspicious account are reported
  (routine NAT/lab IPs are skipped).
- Uncorroborated findings are capped at `shared_ip_max_score` (0.4),
  flagged `shared_ip_only=True`, and carry an explicit operating note.
- A finding escalates only when ≥1 user pair on the IP is independently
  linked by another correlation type (e.g. `similar_account_behavior`).

## Account profiles

`account_profiles` exposes the per-user aggregates used for similarity:
session/event counts, sensitive/auth-failure/after-hours/IP-switch
rates, elevated & suspicious session counts, and the event-type
distribution over a fixed vocabulary.

## Limitations

- **Synthetic data only.** Findings reflect generator behavior; no
  real-world validation.
- **Retrospective, not causal.** Correlations describe the whole
  analyzed window; they cannot raise per-event alerts in real time.
- **Similarity ≠ coordination.** `similar_account_behavior` measures
  profile resemblance; shared tooling, roles, or coursework can produce
  similar legitimate behavior. It is a lead, not a conclusion.
- **Heuristic thresholds.** All cutoffs live in `CorrelationConfig`
  and are documented defaults, not calibrated values.
- **No correlation ground truth.** The synthetic datasets label events,
  not account relationships, so precision/recall of correlations
  cannot be measured directly.
- **IP ambiguity.** NAT, shared labs, and VPN egress make shared IPs
  weak signals by design; corroboration is required for escalation.
- **Session dependency.** Explicit `session_id` quality affects
  repeated-pattern findings; timeout reconstruction is a fallback.

## Verification

- `pytest tests/test_milestone_2_1_correlation.py -q` — 15 tests.
- Regression: `test_milestone_2_0.py`, `test_milestone_1_9.py`,
  `test_milestone_2_0_1_causal.py`, `test_baseline_integration.py`,
  `test_milestone_2_1.py` — all pass unchanged.
