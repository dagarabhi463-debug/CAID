# Phase 4.2 — Application Security Audit

**Date:** 2026-11-12 · **Commit:** `f2a3997` baseline (master) · **Auditor role:** application security audit per roadmap.
**Method:** full manual source review of all first-party Python (~80 files), targeted grep sweeps for dangerous sinks, live dependency audit (`pip-audit`), and regression testing of the one confirmed defect.

## Executive summary

CAID is a **local, single-operator, synthetic-data research prototype**. The audit found **no critical or high-severity vulnerabilities**. One low-severity confirmed defect (CSV formula injection via export metadata newlines) was fixed with a regression test. The remaining findings are low/informational and mostly reflect the localhost deployment model. AI security posture is strong: prompt-injection sanitization, untrusted-data delimiters, predefined read-only queries, and graceful degradation are all implemented and tested.

## A. Application security

- **Streamlit config:** `.streamlit/config.toml` is theme-only; no server config overrides. App uses `st.set_page_config` + `st.navigation` (13 pages). No `unsafe_allow_html` on master (the SOC-UI branch uses `st.html(SOC_CSS)` with static CSS only).
- **Session management:** Streamlit session state holds analysis cache + incident manager; no auth (see SEC-005). Incident mutations (status/notes/merge/split) are analyst actions persisted to `data/incidents/incidents.json` — appropriate for the tool's purpose.
- **Input validation:** CSV upload capped at 10 MB, `.csv` only, UTF-8 decode guarded, schema validation before analysis. AI questions are sanitized before prompting. Streaming-health path inputs are read-only (SEC-007).
- **Output encoding:** Streamlit escapes HTML by default; CSV exports sanitize formula prefixes (SEC-001 fixed the metadata-comment bypass). PDF generation escapes dynamic text via `xml.sax.saxutils.escape` before reportlab Paragraph markup — verified in `core/incident_report.py`.
- **Error handling:** broad `except Exception` in UI paths; one traceback disclosure (SEC-002). Pipeline code fails closed on artifact/calibration fingerprint mismatch.
- **Information disclosure:** settings page shows paths/versions but never secrets (verified — `.env` values are not rendered; `secrets.toml` is gitignored).

## B. File security

- **Upload validation:** see A. `process_uploaded_file` decode errors are caught by the caller's try/except.
- **CSV formula injection:** all export helpers route through `_sanitize_csv_frame`; SEC-001 closed the last gap.
- **PDF generation:** reportlab Platypus with escaped Paragraph text; no external entity/file references.
- **Path traversal:** no user-controlled path joins; all writes under fixed `data/`, `reports/`, `models/` dirs with atomic replace.
- **Permissions/temp files:** `tempfile.mkstemp` in target dir + `os.replace`; DLQ/sink/checkpoint/incident stores all append-or-atomic-replace; no world-readable secret files created.
- **Overwrite behavior:** incident store `upsert` preserves analyst state; sink/DLQ append-only.

## C. AI security

- **Ollama handling:** stdlib `urllib` client, typed errors, timeouts; base URL from env/UI (user-controlled endpoint — SEC-006 info). No API keys exist (local server).
- **Prompt injection:** `sanitize_for_prompt` strips control chars, masks 5 injection pattern classes, truncates; all untrusted content wrapped in `<<UNTRUSTED_DATA>>`/`<<END_UNTRUSTED_DATA>>`; system prompt explicitly designates it as data.
- **Evidence grounding:** prompts built only from `ai_context` bundles of real detection fields; `_fallback_summary` produces deterministic notes when Ollama is down — never fabricates.
- **Output validation:** model text is displayed as markdown and returned with `evidence_used`; no parsing into commands/state.
- **Arbitrary execution:** none — threat hunting restricted to 10 predefined pandas functions (`THREAT_HUNT_QUERIES`); `match_threat_hunt` keyword-matches intents only.
- **State mutation:** AI paths are read-only by contract; tests `test_ai_grounding.py` cover grounding/injection cases.

## D. Streaming security

- **Redis:** `REDIS_URL` env-configurable; supports `rediss://`; consumer groups isolate workers; `validate_event` rejects `label` fields in production streams; bounded queues + explicit backpressure reject; DLQ entries carry reason codes and are replayable with `replay_of` lineage; dedup LRU bounded (100k).
- **Message validation:** schema validation at ingest (required columns, timestamp parse, event-type vocabulary, numeric coercion) before enqueue.
- **Checkpoint integrity:** atomic writes; pickle deserialization risk bounded to local file tampering (SEC-004).
- **Replay abuse:** operator-initiated only; deterministic `event_id` + `(event_id, replay_of)` delivery keys keep replays idempotent.

## E. Data & dependency security

- **Secrets:** `.env` gitignored and untracked; `.env.example` contains placeholders only; no credentials in repo (verified via `git ls-files` + `git check-ignore`).
- **Dependencies:** `pip-audit` (2026-11-12) → **only `pip 26.1.2` flagged** (PYSEC-2026-3721, fixed in 26.2) — a dev-environment tool, not a shipped runtime dependency. All runtime deps (streamlit, pandas, numpy, scikit-learn, joblib, plotly, matplotlib, pytest, reportlab, redis) clean.
- **Logs:** no sensitive-data logging found; DLQ/sink records contain event fields only (synthetic schema).
- **Incident store:** JSON, atomic writes, load-error surfaced rather than crash.
- **Model artifacts:** fingerprint-bound calibration (`detector_fingerprint`/`component_fingerprint`) detects component mismatch; see SEC-004 for tamper model.
- **Backups:** covered in Phase 4.4.

## Deliverables

- `docs/security/phase_4_2_security_audit.md` (this file)
- `docs/security/vulnerability_register.md`
- `docs/security/remediation_report.md`

**Phase 4.2 status: COMPLETE — audit finished, one low-severity fix applied with regression test, no high-impact vulnerabilities found.**
