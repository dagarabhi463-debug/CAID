# CAID Milestone 1.8.1 — Feature Fallback Matrix

**Date:** 2026-11-12 · Companion to `MILESTONE_1_8_1_AUDIT_REPORT.md`
**Scope:** synthetic defensive research prototype only.

This matrix documents every ML feature produced by `engineer_features()`
(30 columns, order preserved in `anomaly_detector.joblib`), plus the
non-ML channels (rules, behavioral, risk) that consume the same entities.

Fallback legend:
- **self** — computed from the row's own entity baseline (known entity).
- **typical-user** — mean across training users of the per-user feature
  (unseen user; Milestone 1.8.1 semantics).
- **per-entity-mean** — mean count per entity in training (unseen IP/type).
- **population** — pooled training-wide statistic.
- **raw** — per-event column, no entity aggregation.
- **constant** — schema column with no populated source.

| # | Feature | Definition | Train-time | Predict-time | Unseen entity | Scope | Artificial-outlier risk |
|---|---------|------------|------------|--------------|---------------|-------|------------------------|
| 1 | `courses_accessed` | raw count | passthrough | passthrough | n/a (raw) | per-event | none |
| 2 | `accounts_accessed` | raw count | passthrough | passthrough | n/a | per-event | none |
| 3 | `api_requests` | raw count | passthrough | passthrough | n/a | per-event | none |
| 4 | `data_mb` | raw MB | passthrough | passthrough | n/a | per-event | none |
| 5 | `token_age_minutes` | raw minutes | passthrough | passthrough | n/a | per-event | none |
| 6 | `after_hours` | raw 0/1 flag | passthrough | passthrough | n/a | per-event | none (zero-variance in train; real signal at predict) |
| 7 | `hour` | timestamp hour | `extract_time_features` | same | n/a | per-event | none |
| 8 | `day_of_week` | timestamp dow | same | same | n/a | per-event | none |
| 9 | `is_weekend` | dow >= 5 | same | same | n/a | per-event | none |
| 10 | `is_after_hours` | hour<6 or >22 | same | same | n/a | per-event | none (zero-variance in train) |
| 11 | `user_total_api` | Σ api per user | user baseline totals | known: stored total; unseen: `typical_total_api` (mean of per-user totals) | typical-user | per-user | **was** pooled-total defect (M1.8, fixed); now typical-user |
| 12 | `user_mean_api` | mean api per user | user baseline mean | known: stored mean; unseen: `typical_mean_api` (mean of per-user means) | typical-user | per-user | **was** pooled mean; now typical-user (≈equal, documented) |
| 13 | `user_max_api` | user mean + 1 std | user baseline mean+std | known: stored; unseen: `typical_max_api` (mean of per-user max) | typical-user | per-user | **was** pooled std → inflated (~78.6 vs 57.0 train mean); fixed |
| 14–16 | `user_total/mean/max_data` | same pattern for `data_mb` | same | same | typical-user | per-user | same as api; `user_max_data` was inflated, fixed |
| 17–19 | `user_total/mean/max_courses` | same for `courses_accessed` | same | same | typical-user | per-user | same; fixed |
| 20 | `user_total_accounts` | — | always 0.0 | always 0.0 | constant | per-user | none (zero-variance; schema quirk, documented since M1.8) |
| 21 | `user_mean_accounts` | mean accounts per user | user baseline mean | known: stored; unseen: `typical_mean_accounts` | typical-user | per-user | was pooled mean; now typical-user |
| 22 | `user_max_accounts` | user mean + 1 std | user baseline mean+std | known: stored; unseen: `typical_max_accounts` | typical-user | per-user | was mean+pooled std; now typical-user |
| 23 | `user_api_z` | (event − user mean)/user std, clip ±10 | **leave-one-out** at fit (excludes scored event); single-event users → 0 | known: stored baseline; unseen: population z vs `typical_mean_api`/`std_api` | population (unseen) / self (known) | per-user | **was** self-referential train z ≈ degenerate 0±0.16 → any honest predict z was a massive outlier; fixed via LOO at fit + retrain |
| 24–26 | `user_data_z`, `user_courses_z`, `user_accounts_z` | same pattern | LOO at fit | same | same | per-user | same defect/fix as `user_api_z` |
| 27 | `user_event_count` | # training events for user | user totals count | known: stored count; unseen: `per_user_event_count` (mean events/user ≈1.03) | typical-user | per-user | **was** pooled total 500 (M1.8 defect, fixed) |
| 28 | `ip_event_count` | # training events for IP | `ip_frequency` value_counts | known: stored count; unseen: `_global_ip_freq` = **mean events per IP** (2.31) | per-entity-mean | per-IP | none — verified NOT pooled-total; already typical-per-IP |
| 29 | `event_type_frequency` | # training events of type | `event_type_frequency` value_counts | known: stored count; unseen: `_global_event_freq` = **mean events per type** (100.0) | per-entity-mean | per-event-type | none — verified NOT pooled-total; already typical-per-type |
| 30 | `is_sensitive_event` | event_type ∈ sensitive list | constant rule | same | n/a | per-event-type | zero-variance in train (all 0); cannot be split on by IF; retained for schema |

## Non-ML channels

| Channel | Unseen-entity behavior | Scope | Notes |
|---------|----------------------|-------|-------|
| `ip_class` (derived, not an ML feature) | prefix-derived label (TEST-NET ranges) | per-IP | consumed by `suspicious_ip_class` rule only; not in `feature_columns` |
| Rules (`DetectionRulesEngine`) | raw-column conditions; `suspicious_ip_class` fires on 198.51.100/203.0.113 prefixes | per-event | eval label↔IP-prefix correlation is a synthetic-data artifact; eval-normal rows never trigger rules (verified: rule_score mean 0.000) |
| `behavioral_deviation_score` | unseen user → pooled `global_stats` per-event mean/std | per-user→population | pooled std is *larger* than per-user std → conservative (smaller deviations); `event_count`/`user_count` stored but unused in scoring; verified no FP contribution (eval-normal mean 0.008) |
| `risk_score` / `combined_score` | consume ML normalized score, rule_score, behavioral deviation | per-event | missing ML → neutral 0.5 flagged `ml_score_missing`; missing behavioral → NaN + weight renormalization |

## Unfitted-baseline fallback path

`engineer_features(df)` without a fitted `baseline_manager` uses the legacy
`extract_*_features()` functions that aggregate over the **current**
DataFrame. This path is only reachable when no fitted manager exists
(prediction always requires one — `AnomalyDetector.predict` raises
otherwise). Documented as non-baseline behavior; not used in production
flow.

## Fit-time vs predict-time asymmetry (fixed in 1.8.1)

| Aspect | Before 1.8.1 | After 1.8.1 |
|--------|--------------|-------------|
| Training `user_*_z` | event included in own baseline → z ≈ 0±0.16 (degenerate) | leave-one-out → honest deviation, std ≈ 2.28 (api) |
| Unseen `user_mean_*` | pooled event mean | typical-user mean (mean of per-user means) |
| Unseen `user_max_*` | pooled mean + pooled std (inflated) | mean of per-user (mean+std) |
| Unseen `user_total_*` | global mean × per-user count | mean of per-user totals |
| Unseen `user_*_z` | population z (unchanged) | population z vs typical mean / pooled std |
| `ip_event_count` unseen | mean events/IP | unchanged (already correct) |
| `event_type_frequency` unseen | mean events/type | unchanged (already correct) |
