# CAID Milestone 2.2 — Threat Investigation & Incident Reports

## Scope

Investigation layer on top of Milestone 2.1 cross-session correlation.
Correlation findings are grouped into persisted incidents that analysts
can triage, annotate, merge, split, and export as PDF/CSV reports.
Synthetic/defensive research data only — no real Canvas connectivity.

## Modules

- `core/incident_manager.py` — `IncidentManager`, `IncidentConfig`,
  `IncidentStore` (JSON persistence).
- `core/incident_report.py` — `generate_incident_pdf` (reportlab) and
  `generate_incident_csv` (sanitized via `modules/export.py`).
- `modules/investigation_dashboard.py` — Streamlit page module.
- `app.py` — new "Threat Investigation" navigation page.
- `tests/test_milestone_2_2_incidents.py` — incident tests.

## Design contract

- **Retrospective only.** Incidents consume correlation findings and
  scored events; they never feed back into per-event causal scores,
  DetectionEngine channel weights, risk levels, or correlation
  thresholds. The v1.9 / v2.0 / v2.1 contracts are unchanged.
- **Evidence-first.** Incidents embed the full finding dicts (including
  evidence lists); reports render that chain verbatim.
- **Deterministic.** Incident IDs are `inc_<sha1[:12]>` over sorted
  finding IDs; auto-grouping is stable for identical input.
- **JSON persistence.** Incidents persist to
  `data/incidents/incidents.json` (atomic temp-file + replace write).
  Missing/corrupt files recover to an empty store with a warning flag.
- **Shared-IP policy preserved.** `shared_ip_only` findings keep their
  capped score and operating note; incidents composed solely of
  shared-IP findings are flagged in the UI and carry an explicit caveat
  in PDF reports.

## Auto-grouping

`build_incidents` runs union-find over findings: two findings merge into
one candidate incident when they share any configured entity kind —
users (`entities`), sessions (`sessions`), or IPs extracted from finding
evidence (`shared_source_ip` signals). Incidents with fewer than
`min_findings_per_incident` findings are skipped. Re-running analysis
upserts by deterministic ID, preserving analyst status and notes.

## Lifecycle

- `create_from_findings(finding_ids, notes)` — manual incident.
- `merge(incident_ids)` — unions findings/notes, records a system note,
  removes sources.
- `split(incident_id, finding_ids)` — moves a subset into a new incident;
  rejected if it would empty either side.
- `add_note`, `set_status` — validated against `IncidentConfig.statuses`
  (`Open`, `Investigating`, `Contained`, `Closed`, `FalsePositive`).
- `evidence_timeline(incident_id, scored_events)` — merged, time-ordered
  event rows for all incident users/sessions, tagged with the finding
  IDs that cover each row.

## Severity

Derived from the max `correlation_score` across an incident's findings
via `IncidentConfig.severity_thresholds` (default: <0.4 Low, <0.6
Medium, <0.8 High, else Critical). Severity is recomputed on every
finding change; it is a triage aid, not a calibrated probability.

## Reports

- **CSV** — tidy long format (section/item/detail) covering incident
  metadata, entities, findings, per-signal evidence JSON, timeline rows,
  notes, and disclaimer; formula-injection sanitized.
- **PDF** — reportlab `SimpleDocTemplate`: metadata table, findings
  table, per-finding evidence, shared-IP caveat block when applicable,
  evidence timeline table, analyst notes.

## Dashboard

The "Threat Investigation" page reuses `analyze_correlation` (the 2.1
pipeline) once per session, then shows: incident metrics, status/
severity/shared-IP filters, an incident table, manual grouping and merge
controls, per-incident detail (status, notes, findings + evidence JSON,
split, timeline chart), and PDF/CSV downloads.

## Limitations

- **Grouping is heuristic.** Shared sessions/IPs can over-merge
  unrelated findings; analysts must review auto-groups (merge/split
  exist for this reason).
- **No incident ground truth.** Like correlations, incidents cannot be
  scored for precision/recall on the synthetic datasets.
- **Single-user store.** JSON file persistence assumes one analyst at a
  time; no locking or multi-user sync.
- **Retrospective only.** Incidents describe the analyzed window; they
  cannot trigger real-time response.
- **Severity ≠ risk.** Incident severity summarizes correlation scores;
  it does not re-score underlying events.

## Verification

- `pytest tests/test_milestone_2_2_incidents.py -q` — 20 tests.
- Full suite: `pytest tests -q` — 396 passed.
- Manual: Streamlit "Threat Investigation" page — load sample data,
  verify auto-grouped incidents, notes/status persistence across
  reruns, PDF/CSV downloads.
