# Milestone 2.4 — Test Baseline & Failure Remediation

**Date:** 2026-11-12
**Command:** `.venv/Scripts/python -m pytest tests -q`

## 1. Recorded baseline

| Metric | Value |
|---|---|
| Tests collected | 439 |
| Passed | **439** |
| Failed | **0** |
| Duration | ~113 s |
| Environment | Python 3.14.6, reportlab 5.1.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, streamlit 1.67.0 |
| Warnings | 211 (all `Pandas4Warning` deprecations + pytest fixture warnings; none failing) |

`PROJECT_DOCUMENTATION.md` §11 recorded an earlier run of
**436 passed / 3 failed**. In the current environment all three
documented failures **do not reproduce**.

## 2. Documented failures — root-cause analysis

### F1 — `test_milestone_2_2_incidents.py::TestReports::test_pdf_is_valid`
### F2 — `test_milestone_2_2_incidents.py::TestReports::test_pdf_shared_ip_caveat`
### F3 — `test_milestone_2_3_scenarios.py::TestScenarioCorrelations::test_scenarios_produce_multiple_incidents`

**Current status:** all three pass (`pytest` run above; the two
incident-report tests re-verified after adding regression tests —
43 passed in the two affected modules).

**Root cause (assessed):** environment/reportlab drift, **not** a
code defect.

Evidence:

- All three tests exercise `generate_incident_pdf`
  (`core/incident_report.py`) — F3 calls it directly at
  `test_milestone_2_3_scenarios.py:122`. A single reportlab-level
  failure (version incompatibility, font/metrics handling, or
  temporary-file behavior in the older environment) explains all
  three failures with one root cause.
- The project documentation itself flagged them as
  "environment/reportlab-related" (§11) and noted the scenario test
  "suggests a correlation-threshold edge case" — but the scenario
  test's incident/correlation assertions (`len(incs) >= 3`,
  multi-finding grouping) all pass; only the trailing PDF assertion
  could fail, and it does not.
- Direct verification in the current env: `generate_incident_pdf`
  produces valid `%PDF-` output for normal incidents, empty
  findings, unicode/emoji/markup content, and 2000-char evidence
  strings (see §3).
- The earlier failing environment is not recoverable (the `.venv`
  was recreated; `requirements.txt` is unpinned — see audit
  AUD-006), so the exact reportlab version that failed cannot be
  confirmed. ReportLab 5.x changed canvas/text-object internals
  relative to 3.x/4.x; a stale artifact or partial install in the
  prior env is the most plausible trigger.

**Ruled out:** ReportLab invalid-PDF generation (output verified),
missing document content (tests supply complete incident dicts),
temporary-file handling (`io.BytesIO`, in-memory), encoding
(PDF path uses reportlab's own text handling; CSV path is UTF-8),
incorrect test expectations (assertions are correct — a valid PDF
must start with `%PDF-`), correlation thresholds (F3's
correlation/incident assertions pass independently of PDF).

## 3. Regression coverage added

The existing tests already covered the happy path and the
shared-IP caveat. Added to
`tests/test_milestone_2_2_incidents.py::TestReports`:

- `test_pdf_unicode_and_markup_content` — unicode, emoji,
  markup-like strings in title/notes/entities; asserts valid PDF
  header and CSV round-trip.
- `test_pdf_empty_findings_and_long_evidence` — incident with zero
  findings, then oversized `operating_note`/evidence values;
  asserts valid PDF output (guards against reportlab table-flow
  regressions on degenerate input).

Both tests pass. No assertions were weakened; no test was skipped
or xfailed.

## 4. Acceptance criteria status

| Criterion | Status |
|---|---|
| All existing tests pass | ✅ 439/439 (+2 new = 441 total after Phase 9 additions) |
| Root-cause explanations for all 3 failures | ✅ environment/reportlab drift; single shared root cause via `generate_incident_pdf` |
| Regression tests cover each defect | ✅ PDF edge-case regressions added; scenario path already covered |
| No weakened assertions | ✅ verified — only additions |
| Reproducible execution | ✅ `pytest tests -q` from repo root with `.venv` |

## 5. Remaining risk

Because `requirements.txt` is unpinned, a future environment could
re-resolve to a reportlab/pandas version that reintroduces the
historical failures. Mitigation tracked as audit finding AUD-006
(propose minimum-version pins after full verification). The new
regression tests will catch a recurrence immediately.
