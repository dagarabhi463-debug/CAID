# CAID Milestone 2.0.1 — Dashboard Audit

Audited `modules/session_intelligence.py` and `app.py` wiring against the installed Streamlit reference (`developing-with-streamlit`, data-display reference). The page is reachable as **Session Intelligence** in `app.py`.

## Required elements

| Requirement | Status | Evidence |
|---|---|---|
| User session timelines | Present | `_session_timeline()` + `st.dataframe`/`st.line_chart` per selected user |
| Suspicious event sequences | Present | filters `sequence_triggered_patterns` and shows pattern/deviation/score/risk |
| Session risk summaries | Present, relabeled | now titled **Retrospective Session Risk Summary** with caption stating aggregates are not causal alerts |
| Behavioral deviations | Present | session/sequence/behavioral deviation columns in timelines and top-anomaly evidence |
| Detection evidence | Present | `structured_evidence` JSON expander for top anomalies |
| Operating mode | Present | `operating_mode` value counts and per-row column |
| Calibration status | Fixed | now displays actual `risk_calibration_status`, `detection_calibration_status`, and `threshold_version` from analyzed columns |
| Missing-channel indicators | Present | missing-channel rate plus per-row `missing_channels` |
| Event timestamps | Present | timeline/top-anomaly tables include `timestamp` |

## Fixes applied

- Replaced deprecated `use_container_width=True` with `width="stretch"`.
- Stopped overwriting detection calibration status with a misleading `risk_calibrated_detection_default`; risk and detection calibration are now separate columns.
- Labeled completed-session aggregates as retrospective so they are not presented as causal historical alerts.

## Remaining caveats

- The dashboard analyzes the sample/production artifact path (`ensure_model_exists`, `ensure_fitted_*`). The independent v2.0 evaluation artifacts are fingerprint-bound to `data/independent_v2`; applying that calibration artifact to sample-data baselines would be a fingerprint mismatch, so the page correctly shows default/uncalibrated detection status unless a compatible artifact is supplied.
- Retrospective session summaries remain available for context but are explicitly separated from causal per-event evidence.
