# CAID v1.0 — Known Limitations

Honest bounds. Nothing here is a defect; all are documented design or evidence limits.

## Detection performance

1. **Independent holdout (synthetic OOD):** combined v2_0 recall 0.94 but **precision 0.50 / FPR 0.39** (`docs/research/phase_4_3_independent_validation.md`). Internal splits look near-perfect — do not extrapolate them.
2. **Synthetic data only.** No real Canvas/enterprise logs were evaluated; an authorized real-world validation set does not exist for this project.
3. **Partial entity overlap:** the independent_v2 holdout shares 48/64 user IDs with train/validation by cohort design; only `user_h*` users are fully disjoint (same FP pattern observed there).
4. **Rules channel is near-silent** on independent data (recall 0.018); session/sequence channels carry recall unevenly. Fusion favors sensitivity.
5. **Scores are calibrated anomaly scores, not probabilities.** Thresholds are fixed post-calibration; no per-deployment tuning is supported.

## Streaming

6. **Burst ceiling ~289 events/sec** single-consumer (preload-drain, batch 1000). The 500 eps burst target is unmet; sustained ~100 eps with p95 < 1s is verified (60-min soak: 360k events, 0 loss).
7. **asyncio queue is dev-grade** without `persist_path`; use Redis Streams for durability.
8. **In-memory `committed_ids` dedup** doesn't survive restart — redelivery re-scores deterministically; consumers must dedup by `event_id`.
9. **Pickle checkpoints** are atomic but version-sensitive; no schema migration.
10. **Late events are DLQ'd, not retroactively scored** (causal-consistency trade-off). Duplicates separated by more than the reorder window flag only later occurrences.
11. **Metrics are in-process snapshots**; no Prometheus export.

## Security / deployment

12. **No authentication** — bind to localhost only; never expose the port.
13. **Plaintext local services** (Ollama HTTP, Redis RESP); use TLS endpoints if remote.
14. **Model/checkpoint artifacts are unsigned** — protect `models/` via filesystem permissions (SEC-004).
15. **MITRE bundle has no integrity pin** — JSON-only parsing bounds impact (SEC-003).
16. **Traceback shown on upload errors** (SEC-002); arbitrary path display on Streaming health page (SEC-007).

## Platform

17. **Windows dev-verified** (Python 3.14.6); Linux/macOS expected to work but not CI-verified.
18. **Retrospective analysis model** — correlation/incidents describe the analyzed window; streaming detection exists but the dashboard correlation layer is batch-oriented.
19. **Shared-IP ambiguity** — IP is a weak corroborating signal by policy (NAT/VPN).
20. **PROJECT_DOCUMENTATION.md predates v2_0** — section 11 is stale; this file + `RELEASE_NOTES.md` are authoritative for v1.0.
