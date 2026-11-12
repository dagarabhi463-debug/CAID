# Milestone 2.9 — Final Validation Report

Date: 2026-11-12  
Project scope: standalone AI-powered cybersecurity research and threat-detection prototype. Synthetic/defensive data only; no enterprise deployment, multi-user management, SaaS, or organizational workflow features.

## Final regression

```powershell
.venv/Scripts/python.exe -m pytest tests -q
```

Result: **530 passed, 8 skipped** in 117.33s. Warnings were pytest deprecation warnings for class-scoped fixtures; no test failures.

## Milestone 2.5.1 — Streaming reliability

Key fixes:

- Redis Streams pending recovery now supports Redis >= 5.0, including an `XAUTOCLAIM` fallback path for Redis 5.x/6.0 via pending-consumer inspection plus `XCLAIM`.
- Asyncio pending reclaim now respects idle time; persistence ordering was tightened for enqueue/reject/restart behavior.
- Worker dedup now uses `(event_id, replay_of)` delivery keys so DLQ replay is not incorrectly suppressed while redelivery remains idempotent.
- Late events are bounded, cleaned from pending acks, and routed to the DLQ with traceable ordering context.
- Retry parameters are honored; tracker failures go to DLQ; checkpoint save/restore covers tracker state, committed delivery keys, and pending ack mappings.
- Result-sink batch serialization was hardened.

Measured validation:

| Run | Produced | Processed | DLQ | Lost | Effective eps | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sustained 60s | 6000 | 6000 | 0 | 0 | 98.8–100.2 | ~0.05–0.12 | ~0.88–0.90 | ~0.94–1.13 |
| Burst 10s | 5000 | 5000 | 0 | 0 | 195.7 | 0.050 | 1.278 | 1.675 |
| Scaled soak 120s | 12000 | 12000 | 0 | 0 | 99.8 | 0.041 | 0.775 | 0.913 |

Reports: `reports/stream_benchmark_20261112T032934Z.json`, `reports/stream_benchmark_20261112T033837Z.json`, `reports/stream_benchmark_20261112T034245Z.json`.

Status: sustained ~100 eps and scaled-soak reliability are demonstrated; the 500 eps burst target and p95 < 1s under burst are **not met**. The true 60-minute soak remains explicitly pending.

## Milestone 2.6 — Monitoring and observability

- Added streaming health aggregation in `core/streaming/health.py`.
- Extended `core/streaming/stream_monitor.py` with stage latency, tracker pending gauges, checkpoint save tracking, artifact status, and drift status helpers.
- Added `app_pages/streaming_health.py` and registered it in `app.py` under Monitor.
- Tests: `tests/test_streaming_health.py` passes; Streamlit app compiled and started successfully on port 8502 during verification.

## Milestone 2.7 — Detection quality

- Added `scripts/evaluate_detection_quality.py` for unseen-user holdout metrics, validation-only threshold selection, calibration status, score separation, and synthetic scenario FP checks.
- Recalibrated v2 detection thresholds from calibration-normal `combined_score` percentiles `(30, 65, 85)` to higher-tail `(90, 99, 99.9)` to reduce false positives.
- New calibrated detection thresholds: medium `0.1332`, high `0.2007`, critical `0.2321`.

Results in `reports/detection_quality.json`:

- Holdout: ROC-AUC `1.0`, PR-AUC `0.9714`; default calibrated precision `0.9211`, recall `1.0`, F1 `0.9589`, FPR `0.0306`.
- Validation-selected threshold applied to holdout: precision `1.0`, recall `0.9714`, F1 `0.9855`, FPR `0.0`.
- Synthetic scenario High/Critical FPR dropped from roughly `0.77–0.85` to `0.047–0.167` after recalibration; baseline FPR is `0.0467`.

Report: `docs/audit/milestone_2_7_detection_quality_report.md`.

## Milestone 2.8 — Advanced AI threat analysis

- Added grounded attack-chain context and investigation in `core/ai_context.py` / `core/ai_analyst.py`.
- Added automated read-only investigation summaries.
- Added four safe predefined threat-hunting queries; no generated code/SQL/shell execution.
- Added an Attack chain tab to `app_pages/ai_analyst.py`.
- Safety preserved: AI output is advisory only and cannot modify scores, thresholds, operating modes, or incident state.

Tests: `tests/test_ai_grounding.py` — **16 passed**.

Report: `docs/audit/milestone_2_8_ai_threat_analysis_report.md`.

## Security and safety notes

- All data is synthetic/defensive research data.
- Prompt-injection sanitization and untrusted-data delimiters remain in place for AI context.
- Threat hunting remains restricted to predefined safe queries.
- ATT&CK mappings remain potential/defensive research mappings, not confirmed attacker behavior.
- Drift monitoring remains observational and does not retrain models or alter thresholds.

## Remaining limitations / explicitly pending work

- **60-minute soak:** not run; only a 120-second scaled soak passed. Run `scripts/stream_benchmark.py --mode soak --rate 100 --duration 3600` to close this item.
- **Burst target:** 500 eps was not achieved; measured burst throughput was ~196 eps with zero loss.
- **Independent validation:** not performed; all accuracy claims are internal reproducibility on the current labeled corpora.
- **Scenario FP:** nonzero scenario FPR remains (`baseline` 4.7%; attack scenarios ~10–17%), largely from shared-IP/after-hours normal-pattern overlap.
- **Live Ollama:** deterministic fallback is tested; live model generation was not exercised in this environment.
- **Redis coverage:** pending recovery verified on local Redis `5.0.14.1`; newer `XAUTOCLAIM` tuple handling is implemented but not verified against a live Redis >= 6.2 server.
- **Checkpoint format:** tracker state is pickled; atomic but version-sensitive.

## Demonstration entry points

- Streamlit dashboard: `.venv/Scripts/streamlit.exe run app.py`
- Streaming health page: Monitor → Streaming health
- AI analyst page: AI → AI Analyst → Attack chain
- Detection quality report: `reports/detection_quality.json`
- Streaming benchmark reports: `reports/stream_benchmark_*.json`
