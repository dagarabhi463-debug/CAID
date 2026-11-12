# Regression Report — Milestone 2.4

**Date:** 2026-11-12
**Suite:** `pytest tests -q`

## Test counts

| Stage | Passed | Failed | Notes |
|---|---|---|---|
| Baseline (start of milestone) | 439 | 0 | 3 documented historical failures did not reproduce |
| After Phase 2 additions | 441 | 0 | +2 PDF edge-case regressions |
| After Phase 4 additions | 454 | 0 | +13 drift monitor tests |
| Closure Phase 1 (artifact lifecycle) | 477 | 0 | +11 lifecycle tests + 2 upload tests |
| Closure Phase 3 (AUD-001 pins) | 482 | 0 | +5 legacy-alias tests (targeted run) |
| Closure Phase 4 (calibration) | 484 | 0 | +2 calibration-artifact guards |
| Mid-closure full run | 485 | **1** | `test_baseline_is_quiet` — dispositioned in closure report §4.2 |
| **Final full run** | **486** | **0** | 120.2s, **3 warnings** (pytest fixture notices only; all 212 pandas deprecations fixed) |

## Regression tests added this milestone

| File | Tests | Covers |
|---|---|---|
| `tests/test_milestone_2_2_incidents.py` | +2 | PDF unicode/markup content; empty findings + oversized evidence |
| `tests/test_drift_monitor.py` | 13 (new) | drift identical/shifted/categorical, explicit statuses, missing columns, non-mutation, determinism |
| `tests/test_milestone_2_4_reliability.py` | 10+2 (new) | entity-disjoint split integrity, normal-only unsup splits, deterministic splits, upload size/extension/encoding validation, formula-injection sanitization, upload→v2_0 pipeline channels |
| `tests/test_artifact_lifecycle.py` | 11 (new) | AUD-002: loaded/missing/corrupt lifecycle, quarantine, persist, normal-data-only fitting |
| `tests/test_score_consistency.py` | +5 | AUD-001: pinned legacy `anomaly_score` divergence + explicit-field precedence |
| `tests/test_calibration_artifacts.py` | 2 (new) | v2_0 artifact fingerprint match + `calibrated` status guard |
| `tests/test_milestone_2_3_scenarios.py` | +1, strengthened | mixed_campaign canonical composition; asserts `long_term_behavior_change` appears; baseline-quiet now asserts no corroborated coordination (calibrated-threshold semantics) |

## Historical failures disposition

See `docs/audit/milestone_2_4_test_remediation.md`: the 3
documented failures (`test_pdf_is_valid`,
`test_pdf_shared_ip_caveat`,
`test_scenarios_produce_multiple_incidents`) were environment/
reportlab drift, not code defects; all pass now and PDF edge-case
regressions were added.

## Determinism verification

- Benchmark splits asserted entity-disjoint and deterministic
  (`test_milestone_2_4_reliability.py`).
- Drift reports verified deterministic modulo timestamp.
- Existing determinism coverage (scenario generator, correlation
  IDs, incident IDs) unchanged and passing.

## Milestone 2.4 exit status

- PDF regressions: verified — `TestReports` 6/6 passed; both
  historical PDF failures pass in the current environment.
- Scenario correlation: verified/fixed — default `mixed_campaign`
  now composes canonical component seeds; diagnostic produced
  50 correlations (all five types present) and 5 multi-finding
  incidents; `test_scenarios_produce_multiple_incidents` passes.
- Full suite: **486 passed / 0 failed** (final closure run above).
- Drift monitor: implemented and tested (13 tests); observational
  only, not wired into detection.
- Calibration/fusion: audited; benchmark/fusion JSON now mark
  `validation_scope=internal_reproducibility` and document
  `independent_v2` recovery from git `bfdca25`.
- **Closure:** v2_0 detection calibration artifact regenerated via
  `scripts/calibrate_v2_detection.py` (controlled, fingerprint-bound,
  benchmark calibration split) → verified `calibrated`
  (`cal-20261112T020320Z-2962264f`). See
  `docs/audit/milestone_2_4_closure_report.md`.
- **Milestone 2.5:** unblocked — calibration gate satisfied;
  detailed implementation plan in
  `docs/roadmap/milestone_2_5_streaming_ingestion.md`. No streaming
  code started before calibration regeneration (gate honored).
