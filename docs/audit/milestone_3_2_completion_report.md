# CAID Milestone 3.2 — Completion Report

Date: 2026-11-12
Scope: complete audit and finalization of the existing CAID Streamlit dashboard. Synthetic/defensive data only; no Canvas, SaaS, or enterprise features. No detection scores, thresholds, models, or v2_0 semantics were changed; no refitting performed.

## Final regression

```powershell
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Result: **543 passed** in ~113s (4 pre-existing pytest deprecation warnings about class-scoped fixtures; no failures). `compileall` clean on `app.py`, `app_pages/`, `modules/`, `core/`, `utils/`. `streamlit run app.py` serves HTTP 200.

## Deliverables

- `docs/audit/complete_dashboard_inventory.md` — all 13 pages: purpose, backends, controls, exports, coverage, gaps, dead code, unsurfaced capabilities.
- `docs/audit/dashboard_functionality_audit.md` — per-page verification results and defects.
- `docs/audit/dashboard_ui_ux_audit.md` — theme/layout/state consistency review.
- `docs/audit/dashboard_test_report.md` — test coverage added and probe-verified behaviors.
- This report.

## What was broken → fixed

1. **MITRE ATT&CK page false "not loaded" state** — `MitreAttackStore.available` was checked before any load, so first visit always showed the download prompt despite a valid cached STIX bundle. Now calls `ensure_data(allow_download=False)` first. Regression test added.
2. **Incident store save flaky on Windows/OneDrive** — `IncidentStore.save()` `os.replace` intermittently failed with `WinError 5` (file locked by sync client), which surfaced as "Analysis failed" on the overview Load-and-analyze flow. Added bounded retry (5 attempts, linear backoff) around the atomic replace.
3. **`streaming_health.py` absent from page smoke tests** — added; page coverage now includes all 13 pages.
4. **Unsanitized CSV exports** — Events, Session Intelligence, and Correlation downloads bypassed `modules/export.py` formula-injection sanitization; all now use `export_csv()`.
5. **Streaming health observability gaps** — latency metric now shows p50/p95/p99; queue-depth/pending/Redis-live gauges explicitly labeled unavailable in observational mode.
6. **Overview checklist gaps** — added data-freshness caption (min→max timestamp + scenario) and detection-channel-contribution chart from real scored-event columns.

## What was added (minimal, read-only)

- **Monitor → Detection Quality** (`app_pages/detection_quality.py`): renders `reports/detection_quality.json`, `fusion_evaluation.json`, `ml_benchmark_results.json`, `calibration_v2_manifest.json`, plus an on-demand observational `DriftMonitor` measurement (normal baseline vs selected scenario). Missing/malformed artifacts degrade gracefully.
- **Data & Admin → Reports & Exports** (`app_pages/reports.py`): artifact inventory with download buttons (evaluation reports, all streaming benchmarks, calibration artifacts, incident store) and an export-location map. Generates nothing itself.

## UI/UX finalization

- Overview now matches other pages with a header + synthetic-data/calibrated-score caption.
- Verified consistent: dark theme (`.streamlit/config.toml`), wide layout, shared sidebar scenario selector, KPI grid, Plotly chart styling, `hide_index`/`width='stretch'` tables, empty/error/loading states, download-button conventions.
- No light theme exists; none added. No animations or decorative components introduced.

## Tests added

- `tests/test_dashboard_pages.py`: `PAGES` now covers all 13 pages; new `test_mitre_page_uses_cached_bundle` (regression) and `test_overview_load_and_analyze_produces_metrics` (functional load→KPIs, with bounded rerun drain and diagnostic output).

## Verified manually (AppTest probes)

- All 13 pages render without exceptions; analysis-dependent pages produce real metrics after "Load and analyze" (e.g., Overview: 650 events, 155 high/critical, 866 correlation findings, 132+ incidents on default scenario).
- Events filter functionally verified (score slider 0.0→0.5: 500→51 rows).
- Incident PDF export produces valid `%PDF-1.4` bytes; CSV exports valid.
- AI Analyst: Ollama connected, all 6 tabs render; degraded fallback path intact.
- MITRE: 706 techniques, 7 confirmed / 12 potential mappings on first visit.
- Streamlit app starts and serves on localhost.

## Known limitations / not changed

- Streaming health remains observational (benchmark files + checkpoint + DLQ); no live Redis/producer probing — by design, now explicitly labeled.
- Incident store accumulates incidents across repeated analyses of the same scenario (deterministic IDs, analyst merge/split available) — pre-existing behavior, noted in audit.
- Dead code documented but not removed: `modules/dashboard.py`, standalone `render_*_page` functions, unused `threat_detection` helpers.
- Log Analysis upload→analysis path verified via unit tests, not an AppTest file-upload simulation.
- Live Ollama generation exercised only via connection check (model `llama3.1:8b` installed); deterministic fallback remains the tested path.
- Remaining 8-skipped tests are unchanged pre-existing skips.
