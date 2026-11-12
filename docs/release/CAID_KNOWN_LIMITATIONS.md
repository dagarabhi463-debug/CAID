# CAID v1.0 — Known Limitations

Honest bounds. Nothing here is a defect; all are documented design or evidence limits.

## Detection performance

1. **Independent holdout (synthetic OOD):** after the Phase 4.6 fusion recalibration, combined v2_0 is **precision 0.88 / recall 0.81 / FPR 0.04** (`docs/research/phase_4_6_fusion_recalibration.md`, `reports/independent_detection_evaluation_v2.json`). The v1.0 operating point (P 0.50 / R 0.94 / FPR 0.39) is preserved in `phase_4_3_independent_validation.md`. Internal splits look near-perfect — do not extrapolate them. Recall was traded for precision; the holdout has now been evaluated twice (post-hoc caveat in Phase 4.6 §5).
2. **Synthetic data only.** No real Canvas/enterprise logs were evaluated; an authorized real-world validation set does not exist for this project.
3. **Partial entity overlap:** the independent_v2 holdout shares 48/64 user IDs with train/validation by cohort design; only `user_h*` users are fully disjoint (same FP pattern observed there).
4. **Rules channel is near-silent** on independent data (recall 0.018); session/behavioral channels carry most of the precision since the Phase 4.6 reweight. Fusion still favors the weighted-mean operating point, not corroboration gating.
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
15. ~~**MITRE bundle has no integrity pin**~~ — **resolved.** `core/mitre_attack.py` now pins a SHA-256 hash in `data/mitre_attack/mitre_attack_meta.json`; cached bundles failing the hash check are rejected (SEC-003).
16. ~~**Traceback shown on upload errors / arbitrary path display**~~ — **resolved.** `modules/log_analysis.py` logs tracebacks server-side via `logger.exception()` and shows a generic UI message (SEC-002); `app_pages/streaming_health.py` restricts displayed checkpoint/DLQ paths to within the project directory (SEC-007).

## Platform

17. **Windows dev-verified** (Python 3.14.6); Linux/macOS expected to work but not CI-verified.
18. **Retrospective analysis model** — correlation/incidents describe the analyzed window; streaming detection exists but the dashboard correlation layer is batch-oriented.
19. **Shared-IP ambiguity** — IP is a weak corroborating signal by policy (NAT/VPN).
20. **PROJECT_DOCUMENTATION.md predates v2_0** — section 11 is stale; this file + `RELEASE_NOTES.md` are authoritative for v1.0.

## Resolved since v1.0 (audit hardening pass)

- **ReDoS in Events search** — `app_pages/events.py` now passes `regex=False` to `str.contains()` so user filter input is treated as a literal substring.
- **Events page hard cap removed** — the Events list now supports page-size selection (100/250/500/1000/2500) plus page navigation; the detail selectbox cap was raised 200 → 500.
- **Incident store file locking** — `core/incident_manager.py` uses `msvcrt`/`fcntl` advisory locks (shared on read, exclusive on write) so concurrent dashboard sessions can't corrupt `incidents.json`.
- **`replay_timerange` sync wrapper** — `core/streaming/event_replay.py` no longer raises `NotImplementedError`; it runs `replay_timerange_async` via `asyncio.run` in a worker thread.
- **AI chat persistence** — `app_pages/ai_analyst.py` saves per-scenario chat history to `data/ai_chat/*.json` and reloads it on first visit; a "Clear chat history" button is provided.

## Remaining design bounds (not code defects)

The following are intentional scope decisions, not bugs — removing them would change the product's purpose and are out of scope for a hardening pass:

- **No authentication (SEC-005)** — localhost-only by design.
- **Plaintext Ollama/Redis (SEC-006)** — supports TLS URLs; no default cert management.
- **Unsigned model artifacts (SEC-004)** — `joblib.load`/`pickle.loads` on local files; mitigate via filesystem permissions.
- **Streaming not wired into dashboard (F-1)** — Streaming health page is observational only (reads benchmark JSON).
- **Log Analysis uses separate session state (F-2)** — doesn't feed Correlation/Investigation/MITRE/AI.
- **Synthetic data only (D-1)** — no real-world validation set exists.
- **Hardcoded detection thresholds/weights (D-7)** — no UI tuning surface.
- **Burst ceiling ~289 eps (P-1)** — below the 500 eps target; requires performance rework.
