# CAID Milestone 1.9 — Missing-ML Policy

**Date:** 2026-11-12 · Phase 5 deliverable.

## Problem resolved

Previously a missing ML score silently contributed a neutral `0.5` to the
combined score — indistinguishable from a genuine mid-range ML
observation — and missing behavioral scores were the only renormalized
channel.

## Operating modes (`core/detection_engine.py::OperatingMode`)

A channel "has evidence" when it produced a **real** observation:

| Channel | Evidence condition |
|---------|--------------------|
| `ml` | a genuine anomaly score exists (`ml_score_missing == False`) |
| `rules` | rule evaluation produced `rule_score` (0 is a genuine "no triggers") |
| `behavioral` | `behavioral_deviation_score` is non-NaN and not flagged missing |

| Mode | Condition | Behavior |
|------|-----------|----------|
| `full_ml_rules_behavioral` | all 3 channels have evidence | weights 0.35/0.35/0.30 as designed |
| `degraded_rules_behavioral` | ML missing; rules+behavioral present | ML **excluded**; weights renormalize over rules+behavioral (0.35+0.30) |
| `degraded_ml_rules` | behavioral missing; ML+rules present | behavioral excluded; renormalize over 0.35+0.35 |
| `degraded_ml_behavioral` | rules missing; ML+behavioral present | rules excluded; renormalize over 0.35+0.30 |
| `insufficient_evidence` | fewer than 2 channels have evidence | `combined_score = NaN`, `risk_level = "Insufficient Evidence"`, **no HIGH/CRITICAL classification**, recommended action = manual review |

## Guarantees

- **Neutral placeholders are never scored**: missing ML contributes
  nothing to `combined_score` (previously it added `0.5 × 0.35`).
- **Missing-channel flags preserved**: `ml_score_missing`,
  `behavioral_score_missing`, plus new `operating_mode` and
  `channels_available` columns on every processed row; `Detection` carries
  the same fields.
- **No false validation claims**: rows in degraded/insufficient modes
  cannot reach CRITICAL without full-channel evidence paths; the
  explanation string states the operating mode, and structured evidence
  marks excluded channels `availability='missing'`,
  `contribution_or_severity='excluded'`.
- **RiskEngine parity**: `risk_factors` ML entry now carries
  `ml_validated` (False for placeholders) alongside `ml_score_missing`;
  `calibration_status` distinguishes calibrated vs
  `uncalibrated_dataset_relative` thresholds.

## Degraded-mode classification behavior (documented)

- Degraded modes still emit LOW/MEDIUM/HIGH/CRITICAL from the
  renormalized score — but the mode is always reported, so consumers can
  distinguish "fully validated" from "partial-channel" assessments.
- `insufficient_evidence` emits **no** risk level classification
  (`RiskLevel.INSUFFICIENT`, score NaN) rather than guessing from a
  single channel.

## Tests (`tests/test_milestone_1_9.py::TestOperatingModes`)

- full mode: all channels → `full_ml_rules_behavioral`, finite scores.
- degraded: ML missing → `degraded_rules_behavioral`, exact renormalized
  score verified numerically (no 0.5 ML term).
- insufficient: bare row → `insufficient_evidence`, NaN score,
  `RiskLevel.INSUFFICIENT`, zero High/Critical predictions.
- missing ML flagged not counted (`ml_score_missing`, excluded).
- mode surfaced in `detection_explanation`.

Pre-existing tests updated where they encoded the old neutral-0.5 /
absent-channel-as-zero semantics (`test_detection_engine.py`,
`test_score_consistency.py`, `test_system_reliability.py`) — each now
supplies real channel inputs or asserts the new insufficient-evidence
contract.
