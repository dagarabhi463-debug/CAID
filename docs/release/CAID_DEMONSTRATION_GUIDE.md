# CAID v1.0 — Demonstration Guide (15–20 minutes)

**Prereqs:** `.venv` built, `streamlit run app.py`, Ollama optional (fallback demo works without it).

| # | Step | What to show | Technical explanation |
|---|------|--------------|------------------------|
| 1 | Launch | `streamlit run app.py` → http://localhost:8501 | Single-process Streamlit app; `st.navigation` exposes 13 pages grouped Monitor / Investigate / AI / Data & Admin. |
| 2 | Overview | Sidebar → scenario selector → **Load and analyze** | `run_analysis()` executes the full v2_0 pipeline once per scenario and caches it in `st.session_state` — every page shares identical scored data. |
| 3 | Synthetic scenario | Pick e.g. *insider exfiltration* or *credential stuffing ring* | `data/generate_scenarios.py` constructs labeled attack patterns; labels are stripped before detection — scores are unsupervised. |
| 4 | Anomaly Detection | Channel score distributions, operating mode | Five channels (ML Isolation Forest, rules, behavioral, session, sequence) each emit 0–1 deviation scores; `DetectionEngine` fuses with fixed weights + calibrated thresholds (`cal-20261112T035409Z-2962264f`). |
| 5 | Suspicious sessions | Session Intelligence page | `SessionReconstructor` groups events into timeout/explicit sessions; prefix features + session baseline produce `session_deviation_score`. |
| 6 | Correlation findings | Correlation page | `cross_session_correlation.py` links findings across sessions/users/IPs with evidence dicts; shared-IP-only findings are capped and labeled low-confidence. |
| 7 | Incident | Investigation page → open an incident | `IncidentManager` union-finds findings into incidents (deterministic `inc_<sha1>` IDs), persists to `data/incidents/incidents.json`; analyst can set status/add notes/merge/split. |
| 8 | MITRE ATT&CK | MITRE page | Curated `CAID_MITRE_MAP` resolves findings/events to technique IDs; cached official STIX bundle supplies names/tactics; potential vs confirmed confidence is explicit. |
| 9 | AI Analyst | AI Analyst (Beta) → explain event / summarize incident / threat hunt | `ai_context` builds grounded evidence bundles, sanitizes log strings, wraps them as `<<UNTRUSTED_DATA>>`; Ollama generates read-only narrative; offline → deterministic fallback. Threat hunting uses 10 predefined queries only. |
| 10 | Streaming health | Streaming health page | Observational telemetry: latest benchmark report, queue impl, DLQ reasons, checkpoint presence — never feeds detection scores. |
| 11 | Detection Quality | Detection Quality page | `reports/detection_quality.json`: entity-disjoint holdout metrics per channel + combined — internal reproducibility scope, honestly labeled. |
| 12 | Exports | Reports & Exports + per-page CSVs/PDF | Incident PDF/CSV via reportlab + sanitized CSV exports (formula-injection safe). |

**Talking points:** calibrated-not-probabilistic scores; synthetic-data scope; streaming parity (batch/stream scores match at 1e-9); honest FP rate on independent holdout (precision ~0.50).
