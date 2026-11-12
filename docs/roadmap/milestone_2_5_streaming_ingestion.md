# Milestone 2.5 — Streaming Ingestion (Roadmap)

**Status:** IMPLEMENTED — see `docs/audit/milestone_2_5_implementation_report.md`.
Asyncio path fully tested incl. batch-parity; Redis Streams adapter
implemented (integration tests auto-skip: local Redis 3.0.504 < 5.0).
Real-time readiness NOT claimed — scaled benchmark misses 100 ev/s /
p95<1s targets; documented in the report.
**Prerequisite docs:** `docs/architecture/real_time_ids_architecture.md`

## Scope

Implement the streaming event pipeline designed in 2.4 — additive to
the batch pipeline, which remains operational and unchanged.

## Deliverables (from the 2.4 architecture)

- `core/streaming/event_schema.py` — canonical event schema +
  envelope validation.
- `core/streaming/event_ingestion.py` — producer/file-tail adapter.
- `core/streaming/event_queue.py` — queue abstraction; Redis Streams
  implementation + asyncio implementation for tests/dev.
- `core/streaming/event_processor.py` — per-event detection dispatch
  reusing existing engines with frozen baselines.
- `core/streaming/event_checkpoint.py` — offset/state snapshots.
- `core/streaming/event_replay.py` — DLQ + time-range replay.
- `core/streaming/stream_monitor.py` — metrics hooks per the 2.4
  observability catalog.

## Implementation plan (approved architecture, additive)

### `core/streaming/event_schema.py`
- `StreamingEvent` dataclass: the 14-column log schema + envelope
  (`event_id`, `ingested_at_utc`, `source`); `label` rejected if
  present in production stream.
- `event_id = sha1(source | event_ts | user_id | session_id |
  event_type | payload-hash)[:16]`; ordering key `(event_ts, event_id)`.
- `validate_event(dict) -> (event|None, reason)` — required fields,
  timestamp parse, vocabulary check; rejects go to DLQ path, never
  silently dropped.
- Bounded dedup LRU (default 100k event_ids); `max_late_seconds=300`.

### `core/streaming/event_queue.py`
- `EventQueue` ABC: `enqueue`, `dequeue(batch, timeout)`, `ack`,
  `pending_count`, `depth`.
- `AsyncioEventQueue`: in-process, bounded (50k), `asyncio.Queue` +
  `QueueFull` backpressure signal — **test/dev implementation only**.
- `RedisStreamsQueue`: Redis Streams + consumer groups
  (`XADD`/`XREADGROUP`/`XACK`), pending-entry retry via
  `XAUTOCLAIM`, `MAXLEN ~` retention, `REDIS_URL` env config.
  Requires `redis` Python package — added to requirements only when
  the Redis path is exercised (or vendored minimal RESP client —
  decide at kickoff; prefer `redis>=5,<7` pinned floor+cap).
- Factory `make_queue(config)` selects impl by config; batch pipeline
  never imports this module.

### `core/streaming/event_processor.py`
- Loads the SAME persisted artifacts as the batch pipeline via the
  AUD-002 `ensure_*` helpers (detector, behavioral, session,
  sequence) — frozen baselines, `artifact_status` surfaced.
- Per-event (or micro-batch) dispatch reusing engine `process_event`/
  channel producers; emits the same score contract fields +
  `detection_latency_ms` metric (never a score input).
- Known bottleneck: engines are per-row (`iterrows`/`apply(axis=1)`);
  M2.5 must either micro-batch through `process_dataframe` or
  vectorize hot paths — measured, not assumed.
- Idempotent: identical event → identical scores (deterministic
  engines + frozen baselines already guaranteed).

### `core/streaming/event_checkpoint.py`
- Periodic snapshots: consumer-group last-delivered IDs +
  session-state version + artifact fingerprints; atomic write
  (tempfile+`os.replace`, same pattern as incident store).
- Resume: on restart, seek to checkpoint; pending entries
  re-delivered by the queue.

### `core/streaming/event_replay.py`
- DLQ: validation failures + exhausted retries, with reason codes;
  inspectable, replayable, never auto-deleted.
- Time-range replay: re-feed events through the identical pipeline;
  `replay_of=<event_id>` lineage recorded; scores unchanged.

### `core/streaming/stream_monitor.py`
- In-process metrics registry implementing the §8 catalog
  (`caid_events_received_total`, `caid_queue_depth`,
  `caid_detection_latency_seconds`, `caid_dlq_depth`, etc.).
- Prometheus `/metrics` export deferred — registry exposes a
  `snapshot()` for tests/UI; exporter is a later additive step.

### `core/streaming/event_ingestion.py`
- File-tail adapter (JSONL/CSV watch) + callable producer API;
  assigns `ingested_at_utc`, computes `event_id`, enqueues or
  applies backpressure (bounded wait → explicit reject signal).

## Tests (all asyncio-impl; no Redis required)

- `tests/test_streaming_schema.py` — envelope validation, event_id
  determinism, dedup LRU, late-event policy, `label` rejection.
- `tests/test_streaming_queue.py` — asyncio queue bounds,
  backpressure, ack/pending semantics; Redis impl tests skipped
  unless `REDIS_URL` reachable.
- `tests/test_streaming_processor.py` — per-event scores equal
  batch-pipeline scores on identical input (parity test); missing
  channels flagged; artifact statuses surfaced.
- `tests/test_streaming_recovery.py` — checkpoint/resume,
  pending re-delivery, DLQ routing, replay idempotency.
- `tests/test_streaming_backpressure.py` — producer block/reject
  under saturated queue; zero silent drops.

## Load & latency harness

- `scripts/stream_benchmark.py`: sustained/burst/soak runner over the
  asyncio impl (and Redis impl when available) recording hardware,
  env, throughput, p50/p95/p99 `detection_latency_ms`, queue depth,
  loss count → `reports/stream_benchmark_<ts>.json`.
- **Real-time is NOT claimed** until: continuous ingestion, crash
  recovery, replay, backpressure, and measured latency are all
  implemented AND the harness results meet 100 ev/s sustained /
  500 ev/s burst / 60-min soak / p95 < 1 s on documented hardware.

## Entry criteria

- ✅ Milestone 2.4 docs/tests landed; v2 calibration artifact
  regenerated via `scripts/calibrate_v2_detection.py`
  (`cal-20261112T015417Z-2962264f`, verified `calibrated`).
- ⚠️ Redis availability in target environment — confirm at kickoff;
  asyncio impl covers tests/dev regardless. SQLite-queue fallback
  only if Redis is unavailable (documented decision required).

## Engineering targets (targets, not assumptions)

- 100 events/s sustained; 500 events/s burst; 60-min soak.
- Zero loss of acknowledged events under declared test conditions.
- p95 detection latency < 1 s on documented hardware/workload.
- Controlled load tests with recorded hardware/env/results —
  never fabricated metrics.

## Explicit non-goals

- No Kafka/Kubernetes without a documented need.
- No retraining-on-drift automation; drift stays observational
  (`core/drift_monitor.py` may be sampled as a metric only).
- No change to batch pipeline semantics, score contracts, or the
  persisted artifact set (streaming consumes the same artifacts).
