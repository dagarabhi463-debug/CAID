# CAID v1.0 — Release Notes

**Release:** v1.0 · **Date:** 2026-11-12 · **Base:** v1.0-rc1 (`f2a3997`) + Phase 4.1–4.5 validation commits.

## What CAID is

A defensive research prototype for cyber-threat detection and investigation over **synthetic** security logs. It is not connected to real Canvas/enterprise systems and makes no real-world accuracy claims.

## What's in the box

- **Detection:** five channels (ML Isolation Forest, detection rules, behavioral baseline, session baseline, sequence model) fused by `DetectionEngine` v2_0 with calibrated, fingerprint-verified thresholds.
- **Investigation:** cross-session correlation findings → incidents (JSON-persisted, analyst workflow) → MITRE ATT&CK technique mappings.
- **AI Analyst (Beta):** local Ollama assistant — grounded explanations, incident summaries, attack-chain reconstruction, predefined threat-hunting queries. Deterministic fallback when Ollama is offline.
- **Streaming:** optional Redis Streams pipeline with consumer groups, XAUTOCLAIM recovery, DLQ + replay, atomic checkpoints, batch/stream score parity (atol 1e-9).
- **Dashboard:** 13 Streamlit pages — Overview, Events, Anomaly Detection, Streaming health, Detection Quality, Session Intelligence, Correlation, Investigation, MITRE ATT&CK, AI Analyst, Log Analysis, Settings, Reports & Exports.

## Verification snapshot

| Check | Result |
|---|---|
| pytest | 544 passed, 0 skipped, 4 warnings (see Phase 4.5 report) |
| compileall | clean |
| Streaming soak (prior) | 360,000 events, 0 lost/DLQ/rejected/dup, 98.7 eps |
| Redis | verified live on 7.4.8 and 5.0.14.1 |
| Security audit | no critical/high; SEC-001 CSV injection fixed |
| Independent detection eval | recall 0.94, precision 0.50, FPR 0.39 (synthetic OOD) |
| Backup/restore | smoke-tested (136 incidents + artifacts restored) |

## Upgrade notes

- v1.0-rc1 tag is preserved; v1.0 is a new tag on the release commit.
- Parallel SOC-UI work lives on branch `soc-ui-overhaul` (`9b54f03`) and is **not** part of v1.0.
- `.env` is never committed; copy `.env.example`.

## Documentation

- Install/config/backup: `docs/deployment/`
- Security: `docs/security/`
- Detection eval: `docs/research/phase_4_3_independent_validation.md`
- Streaming: `docs/release/phase_4_1_streaming_validation.md`, `docs/architecture/real_time_ids_architecture.md`
- Demo: `docs/release/CAID_DEMONSTRATION_GUIDE.md`
- Limitations: `docs/release/CAID_KNOWN_LIMITATIONS.md`
