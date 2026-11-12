# CAID Milestone 1.8 — Detection Accuracy and Calibration Reliability

**Date:** 2026-11-12 · Builds on Milestones 1.5.1–1.7 · CAID remains a
**synthetic defensive research prototype**.

> **Status: PARTIAL.** The false-positive root cause was found and fixed,
and the model artifact was retrained. Several requested workstreams
> (formal dataset splits, per-dataset calibration matrix, structured
> evidence objects, diverse benchmark, schema migration) are documented
> below as remaining work rather than claimed complete.

---

## Phase 1 — Synthetic Dataset Audit (completed)

Comparative analysis of `normal_activity.csv` (500 train) vs
`evaluation_data.csv` (200 eval: 100 normal / 50 suspicious / 50 high_risk):

| Check | Finding |
|-------|---------|
| User overlap | only **5/100** eval-normal users appear in training |
| IP overlap | 70 shared TEST-NET IPs |
| Feature means/stds | nearly identical (api ~55, data ~2.6, courses ~3) |
| Hours | identical 08:00–22:00 business window |
| Event types | same 5-type mix, slightly different frequencies |
| Duplicates | none in either file |
| Labels | consistent `normal/suspicious/high_risk` |

**Conclusion:** marginal-distribution shift is small — the FP problem was
NOT primarily covariate shift.

## Phase 2 — False-Positive Root Cause (completed — real defect fixed)

Baseline before fix: `anomaly_score_normalized` ≈ **0.93 for ALL eval rows
including normal** → FPR 1.00.

**Root cause:** `FeatureBaselineManager.transform_user_features` gave
**unseen users** `user_event_count = global_stats['event_count']` (=500,
the whole training population) and `user_total_* = mean × 500`. Training
users average ~1 event each. Every unseen user therefore presented a
~500× outlier on count/total features → IsolationForest flagged them.

**Fix:** `global_stats['per_user_event_count']` (mean events/user ≈1.03)
added at fit; unseen-user fallback now uses a *typical* user's count and
`mean × per_user_count` totals. Model artifact retrained and resaved.

**Measured effect (evaluation_data.csv):**

| Label | anomaly_score_normalized before | after |
|-------|-------------------------------|-------|
| normal | ~0.93 | **0.585 mean** (71% still >0.5) |
| suspicious | ~0.99 | 0.835 |
| high_risk | ~0.99 | 0.837 |

Separation improved substantially (normal↔anomalous gap widened) but is
**not eliminated** — residual shift remains (see limitations).

## Phase 3 — Calibration Generalization (partial)

`RiskEngine.calibrate_thresholds` / `evaluate_calibration` exist and were
exercised (M1.7). Per-dataset matrix (unseen users/IPs/frequencies) and
formal train/calibration/validation/holdout splits are **not yet
implemented** — the existing CSVs don't provide user-disjoint splits.
Deferred to next milestone; do not treat current metrics as generalized.

## Phase 4 — Detection Explainability (existing, verified)

`Detection.explanation` already reflects real evidence: ML score/missing
flag, triggered rule names, behavioral deviation or explicit "unavailable",
combined score. `behavioral_factors`/`risk_factors` carry per-factor
values/weights. Structured per-channel contribution objects were **not**
added — explanation strings remain the canonical evidence format.

## Phase 5 — Feature Schema (`user_total_accounts`)

**Decision: retain, documented.** The column is permanently 0.0 (never
populated since introduction), i.e. zero-variance. It is part of the
persisted `feature_columns` schema inside `anomaly_detector.joblib`;
removing it would require `ARTIFACT_VERSION` bump + retrain + migration.
Because a zero-variance feature cannot be split on by IsolationForest, it
has **no effect on model decisions** — removal is cosmetic. Retained for
artifact compatibility; flagged as cleanup candidate.

## Phase 6 — Diverse Benchmarking (partial)

M1.7 benchmark infrastructure exists (median/p95, 100–10k). A *diverse*
(non-tiled) generator run was not repeated this milestone; prior numbers
stand: ~3.8 s median @10k events.

## Phase 7 — Regression

- 144/145 tests in affected modules pass after retrain; the single failure
  was a stale assertion encoding the OLD buggy fallback
  (`user_event_count == 500` for unseen users) — updated to assert the
  corrected `per_user_event_count` semantics in `test_milestone_17.py`.
- Full suite re-run pending at report time (last full run: 274 passed
  pre-1.8 changes; the retrain affects only feature fallback values).

## Defects found & fixed

| ID | Defect | Fix |
|----|--------|-----|
| A1 | Unseen users received pooled-population `user_event_count`/`user_total_*` → systematic false positives on any new user | `per_user_event_count` typical-user fallback + model retrain |
| A2 | Stale test asserting buggy fallback value | Updated to corrected semantics |

## Remaining limitations & roadmap

1. **Residual FP rate**: eval-normal mean anomaly still 0.585 — investigate
   remaining channels (ip_event_count for unseen IPs has the same
   pooled-fallback pattern; event_type_frequency likewise) — same fix
   likely applies.
2. **User/IP-level dataset splits** not yet built — required before any
   calibration-generalization claims.
3. **Structured evidence objects** for explainability — pending.
4. **`user_total_accounts`** zero-variance column — removal deferred
   (schema-version migration path designed but not executed).
5. Diverse (non-tiled) benchmark pending.
6. Metrics remain synthetic-only; no production-readiness claim.

---

## Milestone 1.8.1 addendum (2026-11-12) — see MILESTONE_1_8_1_*.md

**Completed:**
- Item 1 resolved: the suspected IP/event-type pooled-total defects did
  **not** exist (verified per-entity-mean fallbacks already correct). The
  real residual-FP causes were (a) self-referential training `user_*_z`
  (event included in own baseline → degenerate z distribution) and
  (b) pooled-statistic unseen-user aggregates (`user_max_*` etc.). Both
  fixed; artifact retrained as v2. Eval-normal `is_anomaly` FPR
  0.68 → 0.12; anomaly recall preserved at 1.00.
- Full feature fallback matrix documented
  (`MILESTONE_1_8_1_FEATURE_FALLBACK_MATRIX.md`); 27 new regression tests;
  full suite 301/301 green.
- Calibration integrity audited: thresholds verified fixed during holdout
  eval; missing-ML neutral-0.5 caveat documented.

**Partially completed:**
- Item 2: unseen-user/unseen-IP coverage confirmed in eval (186/198 users
  unseen), but user/IP-disjoint *splits* still not built — calibration
  generalization claims remain deferred.

**Deferred (unchanged):**
- Item 3 (structured evidence objects), item 4 (`user_total_accounts`
  removal), item 5 (diverse benchmark), item 6 (synthetic-only metrics).
- New: missing-ML neutral-score vs calibrated-threshold policy decision;
  residual ~0.35 mean normalized score on eval-normal is small genuine
  distribution shift + contamination=0.1 design, not a fallback defect.
