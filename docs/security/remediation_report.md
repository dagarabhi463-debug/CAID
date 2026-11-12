# Remediation Report — Phase 4.2 Security Audit

**Date:** 2026-11-12 · **Baseline:** `f2a3997` + Phase 4.1 commit `3a5bc14`

## Fixes applied

### SEC-001 — CSV formula injection via export metadata (Low, confirmed)

- **Change:** `modules/export.py::export_filtered_data` now collapses all whitespace (including newlines) in `search_query` and filter values before embedding them in `#` metadata comment lines.
- **Regression test:** `tests/test_milestone_17.py::TestCsvInjectionSafety::test_export_filtered_data_metadata_newline_injection` (new).
- **Verification:** `pytest tests/test_milestone_17.py -q` → 24 passed.
- **Scope check:** no detection, scoring, or UI behavior change; export content identical for all normal inputs.

## Findings accepted without code change (with justification)

| ID | Severity | Justification for deferral |
|----|----------|----------------------------|
| SEC-002 | Low | Traceback in UI is a UX/ops-hygiene issue on a single-operator localhost tool; changing it alters user-facing error behavior — logged for hardening backlog. |
| SEC-003 | Low | Adding a pinned MITRE bundle hash requires a hash-maintenance decision; JSON-only parsing bounds impact to mapping correctness. Backlog. |
| SEC-004 | Low | Standard ML artifact trust model; requires local write access to exploit. Documented guidance: protect `models/` with filesystem permissions. |
| SEC-005 | Info | By-design local tool; mitigated via deployment guidance (bind 127.0.0.1, no tunnel exposure) in Phase 4.4 docs. |
| SEC-006 | Info | Defaults are loopback; remote endpoints already supported via URL env vars. Documented. |
| SEC-007 | Info | Read-only display of operator-chosen paths; no write primitive. |
| SEC-008 | Info | Streamlit escapes HTML in `st.markdown`; impact limited to formatting. |
| SEC-009 | Info | `pip-audit` clean for all runtime deps; only `pip` (dev tool) flagged — upgrade pip in dev environments, not a product issue. |

## Regression status

- Targeted: `tests/test_milestone_17.py` — 24 passed (incl. new test).
- Full suite re-run deferred to Phase 4.5 final regression (single authoritative run); the change is confined to CSV export metadata formatting.

## Backlog (post-v1.0 hardening, not release blockers)

1. Replace `st.error(traceback.format_exc())` with generic message + server-side log.
2. Pin/verify MITRE bundle SHA-256.
3. Optional artifact signing for model/checkpoint files if CAID is ever deployed beyond a trusted host.
4. Add `--server.address 127.0.0.1` guidance enforcement or Streamlit config check at startup.
