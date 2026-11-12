# Changelog

## v1.0 — 2026-11-12

First stable release of CAID (Cybersecurity AI Detection & Investigation Platform).

### Highlights
- Five-channel detection fusion (ML anomaly, rules, behavioral, session, sequence) with fingerprint-checked v2_0 calibration.
- Cross-session correlation → incident management → MITRE ATT&CK mapping.
- Local AI Analyst (Ollama) with prompt-injection sanitization and deterministic fallback.
- Streaming ingestion pipeline (asyncio + Redis Streams) with DLQ, replay, checkpoints, and verified 60-minute soak (360k events, 0 loss).
- 13-page Streamlit SOC dashboard with CSV/PDF exports.

### Validation evidence
- 544 tests passing (see `docs/release/` phase reports).
- Independent holdout evaluation: `docs/research/phase_4_3_independent_validation.md` — recall 0.94 / precision 0.50 on synthetic OOD data (honest bounds in `CAID_KNOWN_LIMITATIONS.md`).
- Security audit: `docs/security/` — no critical/high findings; one low-severity CSV injection fix.

### Known limitations
See `docs/release/CAID_KNOWN_LIMITATIONS.md`.

## v1.0-rc1 — checkpoint `f2a3997`
Release candidate preserved; tag `v1.0-rc1` unchanged.

## Pre-1.0 milestones (summary)
- M1.x: log ingestion, validation, risk engine, Isolation-Forest detector.
- M2.0–2.4: v2_0 detection stack, entity-disjoint benchmarks, calibration artifacts.
- M2.5/2.5.1: streaming pipeline + reliability hardening.
- M2.7–2.9: detection quality, AI threat analysis, final validation.
- M3.x: SOC dashboard, scenarios, incidents, exports.
