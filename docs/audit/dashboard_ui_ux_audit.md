# CAID Milestone 3.2 — Dashboard UI/UX Audit

Date: 2026-11-12
Scope: navigation, layout, visual consistency, states (loading/empty/error), responsiveness, and theming across all 13 registered pages.

## Theme & layout

- Dark security theme configured centrally in `.streamlit/config.toml` (bg `#0a0a15`, primary `#00d9ff`, text `#e8e8f0`). No light theme exists — consistent single-theme app; nothing to reconcile.
- `app.py` sets `layout="wide"`, expanded sidebar, global `page_icon`, and a shared sidebar containing the CAID caption, scenario selector, and analyzed-scenario status — identical on every page.
- Navigation uses `st.navigation` with four groups (Monitor / Investigate / AI / Data & Admin), icons and help text on every entry. All 13 `st.Page` entries verified to resolve to existing files; `app.py` itself loads via AppTest with no exceptions.

## Consistency findings

| Area | Status | Notes |
|---|---|---|
| Page headers | Fixed | Overview lacked the `st.subheader` + scope caption every other page has; added ("All metrics are computed from real pipeline output — scores are calibrated anomaly scores, not probabilities"). |
| Risk colors | Consistent | `RISK_COLORS` dark-theme palette used on Overview; Anomaly Detection donut uses a fixed 4-color map — visually consistent enough; no clash observed. |
| Chart styling | Consistent | Plotly charts use `width='stretch'`, fixed heights (280–400px), titles and axis labels throughout. |
| Tables | Consistent | `st.dataframe(..., hide_index=True, width='stretch')` pattern used everywhere; `alt` text present on key tables. |
| Captions/disclaimers | Consistent | Every analysis page carries a caption stating synthetic data + calibrated-score semantics; AI page adds advisory-only + untrusted-content notes; MITRE page explains potential-vs-confirmed. |
| Empty states | Consistent | `analysis_or_prompt()` renders a primary "Load and analyze" button + info prompt on all analysis-dependent pages; pages render `st.info` for empty results (no incidents, no findings, no correlations). |
| Error states | Consistent | Analysis failures → `st.error`; Ollama down → warning banner + deterministic fallback; MITRE missing → info + download button; incident store load failure → warning while page continues; report files missing → per-section `st.info`/`st.error` on the new Detection Quality page. |
| Loading indicators | Adequate | `st.spinner` on analysis load, AI generation, MITRE download; buttons that trigger long work are primary-type. |
| Exports | Consistent | `st.download_button` with explicit filenames/MIME types everywhere; CSVs now uniformly sanitized via `modules.export.export_csv`. |
| Filters | Consistent | Multiselect/slider patterns reused; Events page additionally has search + date range. |

## Fixed during this milestone

1. Overview header/caption added for parity with other pages.
2. Streaming health: unavailable queue/pending/Redis-live gauges now explicitly labeled instead of silently empty; latency metric expanded to p50/p95/p99.
3. MITRE page no longer shows a false "not loaded" state when the STIX cache exists.

## Remaining minor observations (accepted, not changed)

- Module renderers keep "(Milestone 2.x)" labels in their subtitles — informational, harmless.
- `st.columns` KPI grids assume ≥5 columns; on very narrow viewports Streamlit wraps gracefully — acceptable for a desktop research dashboard.
- No tooltips/help text on individual metrics beyond `st.caption` context and widget `help=` on the scenario selector; pages are self-explanatory for the target audience.
- No animations or decorative components present — matches the "minimal visual language" requirement.
- Dead code (`modules/dashboard.py`, standalone `render_*_page` functions) documented in the inventory; removal deferred to avoid unrelated churn.
