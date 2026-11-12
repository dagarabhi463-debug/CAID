# CAID Milestone 3.2 — Dashboard Functionality Audit

Date: 2026-11-12
Method: static review of every `app_pages/*.py` + backing `modules/`/`core/` code, plus live `streamlit.testing.v1.AppTest` probes of all 13 registered pages (load, "Load and analyze" flow, widget interaction, export presence). Baseline before changes: 528 passed / 8 skipped; page smoke suite covered 10 of 11 pages.

## Verified working (probed end-to-end)

| Page | Result |
|---|---|
| Overview | Load→analyze produces 14 real metrics, 5+ charts, incident table; no exceptions. Added data-freshness caption + channel-contribution chart (both from real columns). |
| Events | Filters verified functionally: min-score slider 0.0→0.5 reduced table 500→51 rows; detail expander, CSV download present (now sanitized via `modules.export`). |
| Anomaly Detection | Operating modes, missing-channel rate, calibration status, threshold version, metrics/charts/findings all render; no exceptions. |
| Session Intelligence | Session risk summary, sequences, user timeline, top anomalies + evidence, CSV export — all render. |
| Correlation | 866 findings on default scenario; type/score filters, evidence expanders, profiles, sensitive tracks, CSV export render. |
| Investigation | 132→136 incidents (store accumulates across analysis runs — see observations), status/severity filters, detail, notes, merge/split controls, MITRE mappings, **PDF verified `%PDF-1.4`**, CSV verified. |
| MITRE ATT&CK | **Bug found & fixed** (below). After fix: 706 techniques, 7 confirmed / 12 potential mappings on first visit. |
| AI Analyst | Ollama connected (1 model); all 6 tabs render with working controls; degraded-fallback path covered by backend tests. |
| Streaming health | Benchmark telemetry renders; p50/p95/p99 now shown; unavailable gauges explicitly labeled. |
| Log Analysis | Upload section renders; backend validation/cleaning/export functions covered by `test_log_analysis.py` + reliability tests. Full upload→analysis path not re-exercised in this audit (covered by existing unit tests). |
| Settings | Env/model/MITRE/incident-store status all render; Ollama test + MITRE reload buttons present. |
| Detection Quality (new) | Renders all four report JSONs + on-demand drift measurement; no exceptions. |
| Reports & Exports (new) | 32 artifact download buttons + export map; no exceptions. |

## Defects found and fixed

1. **MITRE page first-visit bug** (`app_pages/mitre.py`): `MitreAttackStore.available` is only set by `ensure_data()`/`_load()`, but the page checked `store.available` without ever loading — so the page always showed "data is not loaded" + a download button even with a valid cached STIX bundle. Fixed by calling `store.ensure_data(allow_download=False)` before the availability check. Regression test added (`test_mitre_page_uses_cached_bundle`).
2. **`streaming_health.py` missing from page smoke tests**: added to `PAGES` plus the two new pages (13 pages total now covered).
3. **Streaming health latency**: only p95 was surfaced as a metric; p50/p99 existed in benchmark JSON but were buried. Now shows `p50 · p95 · p99`. Queue-depth/pending gauges are always empty in observational mode — now explicitly labeled unavailable rather than silently blank.
4. **Unsanitized CSV exports**: Events, Session Intelligence, and Correlation downloads used raw `df.to_csv()` bypassing the formula-injection sanitization in `modules/export.py`. Switched all three to `export_csv()`.
5. **Overview gaps vs checklist**: added data-freshness caption (min→max timestamp + scenario) and a detection-channel-contribution chart (mean raw channel outputs + missing-channel rate) — both computed from real scored-event columns, labeled as non-probability channel scores.

## New pages added (minimal, read-only, no new detection logic)

- **`app_pages/detection_quality.py`** (Monitor group): renders `reports/detection_quality.json` (unseen-user holdout metrics, synthetic-scenario FP checks), `reports/fusion_evaluation.json` (per-channel metrics), `reports/ml_benchmark_results.json`, `reports/calibration_v2_manifest.json` (thresholds/fingerprints), and an on-demand `DriftMonitor` measurement (normal baseline vs selected scenario, observational only). Missing/malformed reports degrade to `st.info`/`st.error`, never crash.
- **`app_pages/reports.py`** (Data & Admin group): inventory + download buttons for evaluation reports, latest/all streaming benchmarks, calibration artifacts, incident store; plus an export-location map. Generates nothing itself.

## Checklist items confirmed N/A or covered elsewhere

- "Threat Hunting" and "Attack chain" are tabs inside AI Analyst (per Milestone 2.8 design) — covered, not separate pages.
- "Behavior Analytics" / "Session timelines" / "Unusual login & after-hours activity" — covered by Session Intelligence page + triggered-rule evidence on Anomaly Detection + Events filters.
- "Producer/consumer status, live Redis connection, backpressure" — intentionally absent: streaming page is observational over benchmark artifacts only (Milestone 2.6 design). Now explicitly labeled.
- Dark/light themes: only dark theme is configured (`.streamlit/config.toml`); no light theme exists to audit.

## Observations (not changed)

- `build_incidents` accumulates incidents across repeated analyses of the same scenario (store grew 132→136 during probing). Incidents are persisted research artifacts; dedup/merge is analyst-driven via the Investigation page. Flagging for awareness, not a defect per current design.
- `modules/dashboard.py` (`render_dashboard` + helpers) and the standalone `render_*_page` functions in `session_intelligence.py`/`correlation_dashboard.py`/`investigation_dashboard.py` are dead code under the multipage app — documented in the inventory, left in place (no removal without need).
- `modules/threat_detection.py` contains unused helpers (`display_detection_summary_stats`, `display_detection_explanation`, `get/save_detection_result`) — harmless, documented.
- Log Analysis runs the same v2_0 pipeline but stores results in its own session keys — intentionally isolated from scenario analysis pages.
- Widget-key hygiene: `display_filter_controls` uses auto-generated keys; no collisions observed.
