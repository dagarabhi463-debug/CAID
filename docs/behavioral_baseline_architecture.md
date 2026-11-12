# CAID Behavioral Baseline Architecture (Milestone 1.5.4)

## Overview

Behavioral baselines are **fit once on designated training data** (e.g.
`data/raw/normal_activity.csv`), persisted to
`models/saved_models/behavioral_baselines.joblib`, and reused for all
inference. Inference never fits or updates baselines.

Components:

- `core/behavioral_baseline.py` — `BehavioralBaselineManager`: per-user and
  global statistics, row scoring, chronological evaluation, joblib
  persistence with artifact-version and feature-schema validation.
- `core/behavioural_analyzer.py` — `BehavioralAnalyzer`: thin wrapper
  exposing `fit()`, `analyze_user_behavior()`,
  `analyze_user_behavior_chronological()`, `save()`/`load()`,
  `ensure_fitted_analyzer()`.

## Baseline contents

Per user and globally, for each feature in `BEHAVIORAL_FEATURES`
(`api_requests`, `data_mb`, `courses_accessed`, `accounts_accessed`,
`token_age_minutes`, `after_hours`):

- `avg_<key>` / `std_<key>` (legacy schema keys, e.g. `avg_api`,
  `std_data_mb`), `event_count`, and per-user `role`.

Guards:

- Missing/non-finite values are coerced explicitly; they never silently
  count as normal.
- Zero/NaN variance uses a safe std floor (`_safe_std` → 1.0).
- Unseen users fall back to global training statistics.
- Feature definitions and ordering are stored in the artifact and
  validated on load.

## Evaluation modes (do not conflate)

1. **Historical batch analysis** — `analyze_user_behavior(df)`.
   Fixed training baselines applied to a batch of events. Appropriate for
   offline/research evaluation of already-collected logs. Claims nothing
   about real-time detection latency.

2. **Chronological event detection** — `analyze_user_behavior_chronological(df)`.
   Expanding-window research evaluation: each event is scored only against
   strictly-prior events (per-user history, global-history fallback,
   `min_history` guard). Used to verify that future events cannot leak
   into baselines. Not the production path.

3. **Future production streaming detection** — would score each incoming
   event against the persisted baseline artifact in near-real-time.
   Requires streaming infrastructure not present in this prototype; the
   persisted `BehavioralBaselineManager` artifact is the intended store.

## Score contract

All scores are 0–1 unless noted. **Higher = more anomalous/deviant/risky.**

| Field | Range | Direction | Source | Calibration |
|---|---|---|---|---|
| `anomaly_score_raw` | unbounded (≈[-0.5, 0.5]) | higher = MORE NORMAL | IsolationForest `decision_function` | raw; must be inverted downstream |
| `anomaly_score` | unbounded or [0,1] | ambiguous (legacy) | legacy alias of raw decision_function | legacy range-guess adapter only |
| `anomaly_score_normalized` | [0,1] | higher = MORE ANOMALOUS | calibrated from training score percentiles | fixed training calibration (batch-invariant) |
| `ml_anomaly_score` | [0,1] | higher = MORE ANOMALOUS | alias of `anomaly_score_normalized` | same |
| `behavioral_deviation_score` | [0,1] or NaN | higher = MORE DEVIATION | `BehavioralBaselineManager` vs stored baselines | weighted z-deviation, thresholds ×3 std |
| `behavioral_score_missing` | bool | — | True when no usable baseline/input | NaN score, never silent normal |
| `rule_score` | [0,1] | higher = more rule evidence | `DetectionRulesEngine` weight sum | clamped [0,1] |
| `combined_score` | [0,1] or NaN | higher = riskier | weighted ml/rules/behavioral | missing behavioral → weights renormalized over ml+rules |
| `risk_score` | [0,1] | higher = riskier | `RiskEngine` weighted heuristic | percentile classification relative to dataset (documented limitation) |

Ambiguous inputs are rejected or explicitly adapted: `anomaly_score` is
only consumed through a documented legacy adapter; new producers should
emit `anomaly_score_raw` or `anomaly_score_normalized`.
