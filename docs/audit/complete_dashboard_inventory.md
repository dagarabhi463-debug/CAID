# CAID Milestone 3.2 — Complete Dashboard Inventory

Date: 2026-11-12
Scope: every registered Streamlit page in `app.py` (`st.navigation`, position="sidebar"), its backend integrations, data dependencies, interactive controls, exports, and test coverage.

Baseline: 528 tests passed, 8 skipped (`pytest tests -q`, 143s). Dark theme configured in `.streamlit/config.toml`.

## Navigation structure (as registered)

| Group | Page | File | Icon |
|---|---|---|---|
| Monitor | Overview | `app_pages/overview.py` | dashboard |
| Monitor | Events | `app_pages/events.py` | table |
| Monitor | Anomaly Detection | `app_pages/anomaly_detection.py` | query_stats |
| Monitor | Streaming health | `app_pages/streaming_health.py` | monitor_heart |
| Investigate | Session Intelligence | `app_pages/session_intelligence.py` | monitoring |
| Investigate | Correlation | `app_pages/correlation.py` | account_tree |
| Investigate | Investigation | `app_pages/investigation.py` | troubleshoot |
| Investigate | MITRE ATT&CK | `app_pages/mitre.py` | policy |
| AI | AI Analyst | `app_pages/ai_analyst.py` | smart_toy |
| Data & Admin | Log Analysis | `app_pages/log_analysis.py` | upload_file |
| Data & Admin | Settings | `app_pages/settings.py` | settings |

Sidebar (all pages): scenario selector (`modules/scenario_data.py`, keys from `data/generate_scenarios.py` + default sample dataset) and analyzed-scenario status via `modules/analysis_state.py` (`caid_analysis` session key).

## Shared data flow

`analysis_or_prompt()` → `run_analysis()` → `analyze_correlation(raw)` (`modules/correlation_dashboard.py`) → `analyze_session_intelligence(raw)` (`modules/session_intelligence.py`) → detector + RiskEngine + rules + behavioral + session + sequence + DetectionEngine (v2_0 calibrated). Result cached in `st.session_state['caid_analysis']` with keys `scored_events`, `correlations`, `account_profiles`, `sensitive_tracks`, `scenario_key`, `scenario_label`. `get_manager()` provides the JSON-backed `IncidentManager` (`data/incidents/incidents.json`).

## Page inventory

### 1. Overview — `app_pages/overview.py` (185 lines)

- **Purpose:** platform KPIs, system status, activity charts for the analyzed scenario.
- **Backend:** `analysis_or_prompt`, `get_manager().list_incidents()`, `OllamaClient.list_models()`, `MitreAttackStore.ensure_data()`.
- **Data deps:** `scored_events` (events, users, sessions, anomalies, risk levels, combined_score, calibration status columns), incidents store.
- **Controls:** none beyond the shared "Load and analyze" button; expander "System status".
- **Exports:** none.
- **Checklist coverage:** events analyzed ✓, anomalies ✓, high/critical ✓, open/critical incidents ✓, risk distribution ✓, detection trend (events/day) ✓, top suspicious users/IPs ✓, recent incidents ✓, recent high-risk events ✓, calibration status ✓, Ollama/MITRE status ✓.
- **Gaps:** no explicit data-freshness indicator (max timestamp); no detection-channel-contribution chart (`channel_weights`, `channels_available`, `missing_channels` columns exist in scored events); "Top suspicious users" is a bar chart, fine.
- **Tests:** `test_dashboard_pages.py::test_page_loads[overview]` smoke only.

### 2. Events — `app_pages/events.py` (166 lines)

- **Purpose:** searchable/filterable view of every scored event with per-event evidence.
- **Backend:** `analysis_or_prompt`; pure pandas filtering.
- **Controls:** text search, min combined-score slider, risk-level/user/event-type multiselects, date range, event-detail selectbox.
- **Exports:** "Download filtered events (CSV)" (`st.download_button`).
- **Checklist coverage:** ML/rule/behavioral/session/sequence deviation columns ✓, operating mode ✓, risk classification ✓, detection explanation ✓, structured evidence ✓, missing channels ✓.
- **Gaps:** CSV export bypasses `modules/export.py` sanitization (uses raw `to_csv` — formula-injection sanitization not applied; note for audit).
- **Tests:** smoke only.

### 3. Anomaly Detection — `app_pages/anomaly_detection.py` → `modules/threat_detection.py`

- **Purpose:** Isolation Forest results, rule triggers, behavioral deviations, per-event evidence.
- **Backend:** `display_milestone_3_dashboard(df)` — metrics, risk donut, top suspicious users, daily detection timeline, triggered-rules chart, filter controls (risk levels, score threshold ≥0.5 default, users), detection-findings expanders.
- **Page-level additions:** operating-mode counts, missing-channel rate, risk/detection calibration status, threshold version.
- **Controls:** three filter widgets inside `display_filter_controls` (note: no `key=` args — potential widget-key collisions if page rerun state persists; verify in audit).
- **Exports:** none.
- **Checklist coverage:** all section-B items present in some form; calibration ✓, missing-channel indicators ✓, explanations ✓, evidence ✓.
- **Gaps:** hardcoded `combined_score >= 0.5` "suspicious" threshold in `display_detection_metrics`/`display_detection_details`/`display_detection_summary_stats` — verify this doesn't contradict calibrated risk thresholds (audit item, not necessarily a bug since it's labeled "combined suspicious").
- **Tests:** smoke only.

### 4. Streaming health — `app_pages/streaming_health.py` (75 lines)

- **Purpose:** observational streaming telemetry — never feeds detection.
- **Backend:** `core/streaming/health.py` (`latest_benchmark_report`, `checkpoint_meta`, `dlq_summary`, `health_snapshot`); reads `reports/stream_benchmark_*.json`, `data/stream_checkpoint/`, `data/stream_dlq/bench_*.jsonl`.
- **Controls:** checkpoint-path and DLQ-path text inputs.
- **Shows:** pipeline status, effective eps, events lost, DLQ depth, processed/dead-lettered/rejected/duplicates, latency p95, benchmark JSON, queue impl, checkpoint meta, DLQ reasons, registry counters.
- **Gaps:** p50/p99 only inside raw benchmark JSON expander; `queue.depth`/`queue.pending` come from an in-process `MetricsRegistry` that is never attached → always empty; no live Redis/producer/consumer status (by design — benchmark files only); no explicit "unavailable vs degraded vs healthy" labeling beyond a single status metric.
- **Tests:** `tests/test_streaming_health.py` covers `health.py` unit-level; **page missing from `test_dashboard_pages.py::PAGES`** — confirmed gap.

### 5. Session Intelligence — `app_pages/session_intelligence.py` → `modules/session_intelligence.py`

- **Purpose:** session timelines, sequence patterns, retrospective session risk.
- **Backend:** `render_session_results(scored_events)` — KPIs, operating modes, missing-channel rate, calibration + artifact statuses, retrospective session summary, suspicious sequences, per-user timeline chart, top anomalies with structured evidence.
- **Controls:** user selectbox for timeline; "Download Session Intelligence Results" CSV (raw `to_csv`, unsanitized).
- **Checklist coverage:** user activity profiles (via summary), session timelines ✓, sequence deviations ✓, session-level risk ✓, behavioral/session baselines via deviation columns ✓, calibration/artifact status ✓.
- **Gaps:** "unusual login activity"/"after-hours activity" only visible indirectly via triggered rules in Anomaly Detection — no dedicated filter here; audit will note whether checklist items are satisfied elsewhere.
- **Tests:** smoke only.

### 6. Correlation — `app_pages/correlation.py` → `modules/correlation_dashboard.py`

- **Purpose:** cross-session correlation findings with explicit evidence.
- **Backend:** `render_correlation_results(result)` — finding counts by type, shared-IP-only vs corroborated counts, filters (type multiselect, min score slider), findings table (top 100), evidence expanders (top 25), account behavioral profiles, sensitive-operation tracks.
- **Exports:** "Download Correlation Findings" CSV (raw `to_csv`).
- **Checklist coverage:** cross-session correlation ✓, shared-IP relationships ✓, finding details + evidence ✓, shared-IP-only caveat ✓.
- **Gaps:** no incident-creation shortcut here (lives on Investigation page — acceptable); evidence expanders capped at 25 rows.
- **Tests:** smoke only; `test_milestone_2_1_correlation.py` covers backend.

### 7. Investigation — `app_pages/investigation.py` → `modules/investigation_dashboard.py`

- **Purpose:** incident management — triage, notes, merge/split, PDF/CSV reports.
- **Backend:** `IncidentManager`/`IncidentStore` (`data/incidents/incidents.json`), `core/incident_report.py` (`generate_incident_pdf` via reportlab, `generate_incident_csv`), `_mitre_mappings` via `MitreAttackStore.map_incident`.
- **Controls:** status/severity/shared-IP-only filters, incident selectbox, set-status, split multiselect, add-note, merge multiselect, manual incident creation from findings.
- **Shows:** incident metrics, linked users/IPs detail, findings table + per-finding evidence JSON, evidence timeline chart, MITRE mappings with potential-vs-confirmed caption, PDF + CSV download buttons.
- **Checklist coverage:** incident creation ✓ (auto via `build_incidents` + manual), severity ✓, lifecycle (Open/Investigating/etc.) ✓, supporting evidence ✓, MITRE mapping ✓, exports ✓.
- **Gaps:** PDF validity must be verified in audit (reportlab path exercised only if MITRE/incidents exist); store load warning shown but page continues.
- **Tests:** smoke only; `test_milestone_2_2_incidents.py` covers backend.

### 8. MITRE ATT&CK — `app_pages/mitre.py` (156 lines)

- **Purpose:** ATT&CK framework browser, CAID coverage, live mappings.
- **Backend:** `MitreAttackStore` (`data/mitre_attack/enterprise-attack.json` STIX bundle), `CAID_MITRE_MAP`, `map_finding`, `map_incident`.
- **Controls:** "Download MITRE ATT&CK data" (allow_download=True) when unloaded, tactic filter, technique selectbox, "Open on MITRE ATT&CK site" (webbrowser), mappings CSV download.
- **Checklist coverage:** technique coverage ✓, confirmed vs potential distinction ✓ (curated `state` field), unmapped rules reported not guessed ✓, incident→technique mappings ✓.
- **Gaps:** page calls `store.ensure_data(allow_download=False)` inside `@st.cache_resource` store — download button path needs verification; `webbrowser.open` is a no-op-ish side effect in headless contexts (harmless).
- **Tests:** smoke only; `test_mitre_attack.py` covers backend.

### 9. AI Analyst — `app_pages/ai_analyst.py` → `core/ai_analyst.py`, `core/ollama_client.py`, `core/ai_context.py`

- **Purpose:** Ollama-grounded explanations, incident summaries, attack-chain investigation, safe threat hunting, Q&A.
- **Backend:** `OllamaClient` (stdlib urllib → local Ollama), `explain_event`, `explain_finding`, `summarize_incident`, `investigate_attack_chain`, `generate_investigation_summary`, `hunt`, `ask`.
- **Controls:** 6 tabs (Event explanation, Finding explanation, Incident summary, Attack chain, Threat hunting, Ask a question); per-tab selectboxes/buttons; hunt CSV download.
- **Checklist coverage:** connection status banner ✓, model selection (via Settings `ollama_model` session key) ✓, incident/attack-chain selection ✓, evidence-used expander ✓, degraded fallback caption ✓, predefined safe hunts ✓.
- **Gaps:** model selection lives on Settings page — cross-page dependency to verify; live Ollama untested in this environment (deterministic fallback is the tested path).
- **Tests:** smoke only; `test_ai_grounding.py` (16 tests), `test_ollama_client.py` cover backend.

### 10. Log Analysis — `app_pages/log_analysis.py` → `modules/log_analysis.py` (897 lines)

- **Purpose:** upload raw security-log files, validate, clean, run the same v2_0 pipeline, explore results.
- **Backend:** `validate_upload`, `process_uploaded_file`, `get_validation_summary`, `clean_and_summarize` (`utils/validators.py`), `analyze_session_intelligence` (same pipeline as scenarios — AUD-003), `modules/export.py` sanitized CSV exports.
- **Controls:** file uploader, proceed/cancel on validation issues, search/filters, user/IP investigation panels, timeline, CSV exports.
- **Checklist coverage:** upload → analysis → exports ✓; runs identical detection pipeline ✓.
- **Gaps:** analysis result stored in `df_processed` session state but **not** fed into `caid_analysis` — Log Analysis output is isolated from the scenario-analysis pages (by design? verify intent in audit); page is the largest and most complex — highest bug surface.
- **Tests:** smoke only; `test_log_analysis.py`, `test_milestone_2_4_reliability.py` cover backend functions.

### 11. Settings — `app_pages/settings.py` (159 lines)

- **Purpose:** environment info, ML detector artifact status, Ollama config, MITRE dataset management, incident store status.
- **Backend:** `models.anomaly_detector.get_model_path`, `OllamaClient`, `MitreAttackStore`, `get_manager().store`.
- **Controls:** Ollama base URL/timeout inputs, "Test connection & refresh models", model selectbox/text input, "Reload MITRE dataset".
- **Checklist coverage:** model selection/config ✓, connection test ✓, calibration status from last analysis ✓, store path/errors ✓.
- **Gaps:** `ollama_models`/`ollama_conn` session keys only populate after clicking test — first visit shows neither success nor warning (minor UX gap); no Redis/streaming config surfaced (REDIS_URL unused by UI — acceptable, streaming is benchmark-file-based).
- **Tests:** smoke only.

## Dead / unused code identified

| Item | Status |
|---|---|
| `modules/dashboard.py` (`render_dashboard` + 10 `display_*` functions) | **No callers, no tests** — legacy Milestone-1 dashboard superseded by `app_pages/overview.py`. Candidate for removal or documentation. |
| `modules/session_intelligence.render_session_intelligence_page` | Unused by multipage app (only `render_session_results` is called). |
| `modules/correlation_dashboard.render_correlation_page` | Same — unused standalone entry point. |
| `modules/investigation_dashboard.render_investigation_page` + `_manager()`/`_ensure_analysis()` | Same — unused standalone entry point. |
| `modules/threat_detection.display_detection_summary_stats`, `display_detection_explanation`, `get/save_detection_result` | Not called by `display_milestone_3_dashboard` or pages (verify during audit). |

## Unsurfaced backend capabilities (no UI today)

| Capability | Location | Artifact |
|---|---|---|
| Detection-quality evaluation | `scripts/evaluate_detection_quality.py` | `reports/detection_quality.json` |
| Fusion evaluation | `scripts/evaluate_fusion.py` | `reports/fusion_evaluation.json` |
| Model benchmark | `scripts/benchmark_models.py` | `reports/ml_benchmark_results.json` |
| Drift monitor | `core/drift_monitor.py` (`DriftMonitor`, tested in `test_drift_monitor.py`) | none — compute-on-demand only |
| Calibration manifest | `scripts/calibrate_v2_detection.py` | `reports/calibration_v2_manifest.json`, `models/saved_models/calibration/*.joblib` |
| Stream benchmarks | `scripts/stream_benchmark.py` | `reports/stream_benchmark_*.json` (surfaced on Streaming health) |

## Test coverage summary

- `tests/test_dashboard_pages.py`: AppTest smoke test for 10 pages + `app.py` navigation. **Missing: `streaming_health.py`** (11th page) — confirmed gap.
- Backend coverage is strong (528 tests) but page-level functional tests (filters, buttons, exports, fallback states) are essentially absent — all page tests are "renders without exception" only.
