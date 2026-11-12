# Reliability Report — Milestone 2.4

**Date:** 2026-11-12
**Scope:** ML validation, calibration, fusion, security, performance.

## 1. ML detection reliability

Executed `scripts/benchmark_models.py` on entity-disjoint splits
(unsup_train 179 / calibration 73 / sup_train 227 / validation 123
/ holdout 133; ~850 labeled rows, seed 42). Scope: **internal
reproducibility** on current data — not independent validation;
`data/independent_v2` recovery from git `bfdca25` is documented in
`docs/research/ml_benchmark_methodology.md`.

- **Isolation Forest (production):** holdout binary P=0.946,
  R=1.000, F1=0.972, FPR=0.020, ROC-AUC=1.0 — remains the baseline.
- **LOF:** rejected — FPR≈1.0 at default offset.
- **RF/GBM (supervised reference):** perfect binary scores but
  require labeled attack data; not deployable candidates.
- **High-risk-only mode is the weak spot** for all models
  (best F1≈0.57) — documented limitation for Milestone 2.6.
- Full numbers: `docs/research/ml_benchmark_results.md`;
  raw: `reports/ml_benchmark_results.json`.

## 2. Calibration reliability

- Mechanism verified sound (fingerprint binding, role enforcement,
  explicit uncalibrated fallback) — `docs/research/calibration_reliability.md`.
- **Resolved at closure:** `scripts/calibrate_v2_detection.py`
  regenerated the v2_0 artifact on the benchmark calibration split
  (73 normal rows, entity-disjoint, seed 42) →
  `v2_0_thresholds_…-3ab7b78cf14b2f21.joblib`,
  `cal-20261112T020320Z-2962264f`, verified `calibrated`.
  Stale artifacts archived; `tests/test_calibration_artifacts.py`
  guards against future drift. RiskEngine v1 calibration loads fine.
- **New:** `core/drift_monitor.py` — observational PSI/KS/TV
  drift with explicit status; 13 tests; never mutates detection.

## 3. Detection fusion validation

Executed `scripts/evaluate_fusion.py` →
`reports/fusion_evaluation.json`:

- Combined v2_0 holdout: P=1.0, R=0.343, F1=0.511 (conservative at
  default thresholds); all rows `full_v2_0`.
- Behavioral channel strongest single contributor (R=0.943);
  rules high-precision/low-recall; sequence moderate; session
  produced no usable scores on this small corpus.
- Ablation shows score compensation via weight renormalization;
  no weight changes made (no holdout tuning).
- Post-scenario-fix rerun: channel/combined/ablation metrics are
  unchanged (holdout combined P=1.0, R=0.343, F1=0.511; ablation
  F1 range 0.511–0.571); only timing fields differ.

## 4. Security findings (Phase 7)

Verified by audit + new tests:

- Upload: 10MB cap, `.csv`-only, 5-row probe — enforced
  (`validate_upload`); oversized/non-CSV/empty rejected (tested).
- Strict UTF-8 decode failure surfaces as page-level error —
  documented; friendly-message improvement is a minor UX item.
- Formula injection neutralized in CSV exports (tested).
- No eval/exec/subprocess; rule engine uses restricted AST.
- joblib artifacts trusted-local-path only (documented boundary);
  MITRE bundle download has no checksum — documented (AUD-008).
- Secrets: `.env` pattern; settings page never displays secrets.
- AuthN/AuthZ/audit-log design: `docs/architecture/
  real_time_ids_architecture.md` §9 — PLANNED.

## 5. Performance findings

- Benchmark train times: IF 1.68s, LOF 0.08s, RF 2.34s, GBM 0.24s
  on ~200 rows; peak traced memory 2.54MB.
- Known hotspots (documented, unchanged): per-row `iterrows` in
  `DetectionEngine.process_dataframe`, per-row AST rule eval,
  `RiskEngine` `apply(axis=1)`. M1.7 baseline ~3.8s @10k events.
- Streaming targets (100 ev/s sustained, 500 ev/s burst, 60-min
  soak, p95 <1s) are **PLANNED** — no load harness exists yet;
  per-row processing is the identified bottleneck to address in
  Milestone 2.5.

## 6. Status summary

| Area | Status |
|---|---|
| Test suite | TESTED — **486 passed / 0 failed** at closure (see `docs/audit/milestone_2_4_closure_report.md`) |
| ML benchmark | TESTED — executed, results documented |
| Drift monitor | IMPLEMENTED + TESTED (observational) |
| Calibration mechanism | TESTED — v2_0 artifact regenerated via controlled workflow, verified `calibrated` |
| Fusion validation | TESTED — executed on entity-disjoint splits |
| Artifact lifecycle | IMPLEMENTED + TESTED — unified loaded/refit policy, corrupt-artifact quarantine (AUD-002) |
| Upload pipeline | TESTED — now runs the same v2_0 detection pipeline as scenarios (AUD-003) |
| Real-time streaming | PLANNED — architecture + ADR + detailed M2.5 plan; no real-time claims |
| Auth/audit logging | PLANNED — design only |
| Prometheus metrics | PLANNED — catalog defined, nothing implemented |
