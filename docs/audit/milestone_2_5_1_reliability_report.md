# Milestone 2.5.1 — Streaming Reliability Report

Date: 2026-11-12  
Scope: standalone defensive research prototype; synthetic data only.

## Implemented fixes

- **Redis Streams compatibility (Redis >= 5.0)**
  - `RedisStreamsQueue.claim_pending` now handles `XAUTOCLAIM` 2-tuple and 3-tuple replies for newer servers.
  - Added a Redis 5.x/6.0 fallback using pending-consumer inspection plus `XCLAIM` when `XAUTOCLAIM` is unavailable.
  - Verified against local Redis `5.0.14.1`; `tests/test_streaming_queue.py` passes including pending recovery.

- **Queue redelivery semantics**
  - `AsyncioEventQueue.claim_pending` now respects `min_idle_ms` instead of returning every in-flight entry, preventing eager reclaim of events still buffered for ordering.
  - Asyncio persistence now writes the payload before exposing the queue id and writes an ack tombstone if the bounded queue rejects the event, avoiding stranded payloads after restart.

- **Late-event / DLQ / replay correctness**
  - Worker dedup now uses a delivery key of `(event_id, replay_of)` so operator DLQ replays are not incorrectly suppressed as duplicates of the original delivery, while redelivery of the same replay remains idempotent.
  - Late events now remove their `pending_acks` entries before DLQ acknowledgement.
  - Tracker late-event metadata is bounded (`late_events` deque + bounded `late_details`).

- **Worker reliability**
  - `run_worker` now honors `max_attempts`, `base_backoff_s`, and `max_backoff_s` instead of hardcoded retry parameters.
  - Tracker ingest failures are routed to the DLQ as `processing_error` instead of crashing the worker.
  - Checkpointing is wired into `run_worker` with atomic saves after commit boundaries and restore of tracker state, committed delivery keys, and pending ack mappings.

- **Result durability**
  - `JsonlResultSink.write_batch` serializes the complete batch before writing, preventing partial batches on serialization failure.

- **Benchmark reporting**
  - `scripts/stream_benchmark.py` now reports `events_dead_lettered`, `dlq_reasons`, `events_rejected`, `events_duplicate`, `events_inflight`, and only counts true unaccounted events as `events_lost`.

- **Performance**
  - Removed per-row `pd.to_datetime` work for the shared `scored_at` timestamp and added a fast ISO-8601 parse path.
  - Removed pandas `Series` construction from streaming duplicate/hash helpers and allowed the session baseline scorer to consume plain mappings.

## Verification

Commands run:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_streaming_queue.py -q
.venv/Scripts/python.exe -m pytest tests/test_streaming_queue.py tests/test_streaming_ordering.py tests/test_streaming_duplicates.py tests/test_streaming_pipeline.py -q
.venv/Scripts/python.exe scripts/stream_benchmark.py --mode sustained --rate 100 --duration 60 --batch-size 100
.venv/Scripts/python.exe scripts/stream_benchmark.py --mode burst --rate 500 --duration 10 --batch-size 100
.venv/Scripts/python.exe scripts/stream_benchmark.py --mode soak --rate 100 --duration 120 --batch-size 100
```

Results:

| Run | Produced | Processed | DLQ | Rejected | Duplicate | Inflight/Lost | Effective eps | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sustained 60s, 100 eps target | 6000 | 6000 | 0 | 0 | 0 | 0 | 100.2 / 98.8* | 0.117 / 0.047* | 0.895 / 0.878* | 0.943 / 1.128* |
| Burst 10s, 500 eps target | 5000 | 5000 | 0 | 0 | 0 | 0 | 195.7 | 0.050 | 1.278 | 1.675 |
| Scaled soak 120s, 100 eps | 12000 | 12000 | 0 | 0 | 0 | 0 | 99.8 | 0.041 | 0.775 | 0.913 |

*Sustained was run twice after optimization; both runs processed all events with zero loss. The earlier run measured 100.2 eps and the latest run 98.8 eps, so sustained throughput is approximately at the 100 eps target with run-to-run variance. p95 remained below 1 second in both sustained runs.

Benchmark reports:

- `reports/stream_benchmark_20261112T032934Z.json` — sustained, 100.2 eps
- `reports/stream_benchmark_20261112T033837Z.json` — burst, 195.7 eps
- `reports/stream_benchmark_20261112T034245Z.json` — scaled soak 120s, 99.8 eps

## Target status

- **100 events/sec sustained:** approximately met (98.8–100.2 eps measured; p95 < 1s).
- **500 events/sec burst:** not met as sustained throughput (195.7 eps measured). Burst completed with zero loss/DLQ, but the 500 eps target remains unmet.
- **p95 detection latency < 1s:** met for sustained and scaled soak; not met for the 500 eps burst (p95 1.278s).
- **Zero acknowledged-event loss:** met in all reported runs (produced = processed, DLQ = 0, inflight/lost = 0).
- **60-minute soak:** not run in this cycle per project decision; a 120-second scaled soak passed. Full `--duration 3600` remains explicitly pending.

## Known limitations

- Burst throughput remains below the 500 eps target; further optimization would require broader vectorization or partitioned workers, which is out of scope for this reliability pass.
- Checkpoint state uses pickle for tracker internals; it is atomic but remains version-sensitive. No schema migration mechanism has been added.
- Redis fallback pending recovery is verified on Redis 5.0.14.1; newer `XAUTOCLAIM` behavior is implemented but only unit-covered by code paths, not a live Redis >= 6.2 server.
- Metrics remain in-process snapshots; Prometheus export is still deferred.

## Addendum — live Redis >= 6.2 verification & 60-minute soak (Devin session)

**Redis Streams live verification (Redis 7.4.8, redis-windows msys2 build,
port 6380, AOF everysec):**
- `tests/test_redis_streams.py` — 8 tests passed against the live server:
  pending-not-redelivered, acked-never-redelivered, consumer restart
  recovery via XAUTOCLAIM, idempotent group creation, backpressure on
  stream depth, payload fidelity, factory selection, and a full
  `run_worker` end-to-end over Redis Streams (25 events, 0 DLQ,
  pending_count 0 after ack).
- Manual restart test: 10 events enqueued, 6 delivered, 2 acked, server
  killed (`shutdown nosave`) and restarted with AOF. Post-restart:
  XAUTOCLAIM recovered exactly the 4 unacked pending entries; the 2
  acked entries were never redelivered; remaining 4 new entries read
  normally; pending_count 0 after final ack.
- This closes the prior gap: XAUTOCLAIM behavior is now verified live on
  Redis >= 6.2, not only on the 5.0.14.1 fallback path.

**Ordering root cause (Phase 1 evidence):**
- The original 29 DLQ events were traced to `scripts/stream_benchmark.py`
  `_events()` assigning its +60s timestamp offset by GLOBAL row index,
  producing non-monotonic per-user timestamps (verified: exactly 29
  non-monotonic adjacencies = 29 `late_event` DLQ entries). The tracker
  late-detection was behaving correctly; the generator was fixed to
  stable-sort by timestamp (per-user monotonic) with an opt-in bounded
  `--reorder-window` shuffle for reorder coverage. Post-fix sustained
  run: 1500/1500 processed, 0 DLQ.

**Duplicate semantics (Phase 2):**
- Stream `is_duplicate_event` aligned to batch within causal bounds:
  all occurrences of duplicated resolved-content are flagged when the
  hash is observed >= 2 times causally (co-pending drain members +
  emitted history). Documented residual: duplicates separated by more
  than the reorder window flag only later occurrences. Regression tests
  in `tests/test_streaming_duplicates.py` (8 cases incl. checkpoint
  restore and replay lineage).

**Worker idempotency:**
- `run_worker` dedups redelivered queue entries by delivery key
  `(event_id, replay_of)` via bounded LRU (`committed_ids`), acked
  without re-scoring; persisted through `StreamCheckpoint` worker_state.
  Covered by `TestWorkerIdempotency` in `tests/test_streaming_pipeline.py`.

**Performance work (Phase 4, measured):**
- cProfile of the worker hot path: `pd.to_datetime`/`_parse_iso`/
  `from_dict` ≈ 48% of wall time; `_hash_for` `pd.Series` construction
  ≈ 11%. Fixed via `datetime.fromisoformat` fast path (`_to_timestamp`,
  `_parse_iso`) and a Series-free `_hash_for` (hash output verified
  byte-identical). Sweep-emitted rows now accumulate into a micro-batch
  buffer flushed at `batch_size` or `max_batch_latency_s` (default 0.1s)
  instead of per-sweep commits; commit/retry/DLQ tail shared via
  `_commit_rows`.
- `scripts/profile_streaming.py` added as a committed profiling harness.

**Benchmark configuration notes:**
- Sustained/burst runs used `--max-hold-s 0.25` (reorder hold is the
  dominant latency term; 0.5s default gave p95 ≈ 1.05s, 0.25s gives
  p95 ≈ 0.74–0.94s). Preload burst (`--preload --batch-size 1000`)
  measured pure consumer drain at ~289 ev/s — the single-consumer
  ceiling on this hardware; the 500 ev/s burst target remains unmet
  (needs partitioned workers or vectorization, out of scope).

**Full suite:** `pytest tests -q` → 528 passed, 8 skipped (Redis tests
skip without `REDIS_URL`; they pass when `REDIS_URL=redis://127.0.0.1:6380/0`).

**60-minute soak (COMPLETE):** `--mode soak --rate 100 --duration 3600
--max-hold-s 0.25 --batch-size 500` — 360,000/360,000 events processed
over 3,648.8s wall, **0 lost / 0 DLQ / 0 rejected / 0 duplicate /
0 inflight**, effective 98.7 eps, detection latency p50 0.30s,
p95 0.88s, p99 1.86s (single max outlier 21.9s). Hardware: Windows 11,
Python 3.14.6, pandas 3.1.1, asyncio queue, reorder_window_s 0.05,
max_hold_s 0.25. Report: `reports/stream_benchmark_20261112T045510Z.json`.
