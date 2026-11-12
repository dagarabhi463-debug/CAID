# Milestone 2.5 — Streaming Ingestion: Implementation Report

**Status:** IMPLEMENTED (asyncio path fully tested; Redis Streams adapter
implemented, integration tests auto-skip — local Redis is 3.0.504, Streams
require >= 5.0). **Real-time readiness NOT claimed** — see benchmark results.

## What was built

`core/streaming/` (additive; batch pipeline untouched):

- `event_schema.py` — canonical envelope, `validate_event`, deterministic
  `event_id` (sha1[:16]), bounded dedup LRU, `label` rejection in
  production mode.
- `event_queue.py` — `EventQueue` ABC; `AsyncioEventQueue` (bounded 50k,
  optional JSONL durable backing for crash-recovery tests);
  `RedisStreamsQueue` (XADD/XREADGROUP/XACK/XAUTOCLAIM, consumer groups,
  MAXLEN ~ retention); `make_queue` factory.
- `event_ingestion.py` — producer API + JSONL tail adapter; bounded
  backpressure -> explicit reject, never silent drops.
- `session_state.py` — `IncrementalSessionTracker`: per-user/per-session
  reorder buffers, timeout/explicit/missing session assignment, causal
  prefix features, per-session sequence windows. Replicates batch
  semantics including the `itertuples` `_ts` quirk (duration/rate/has_ts
  are always NaN/0 in batch — replicated for parity).
- `event_processor.py` — loads the SAME persisted artifacts via AUD-002
  `ensure_*` helpers + fingerprint-checked v2_0 calibration; per
  micro-batch runs ML/rules/behavioral/risk channels then fusion via
  `DetectionEngine('v2_0')`; `run_worker` loop: dequeue -> order ->
  process -> sink commit -> ack (ack ONLY after durable commit),
  exponential backoff retry, DLQ on exhaustion/poison/late events,
  periodic pending-claim + idle sweep.
- `event_checkpoint.py` — atomic JSON+pkl snapshots (queue meta, tracker
  state, artifact fingerprints, monitor).
- `event_replay.py` — append-only DLQ + `replay_dlq_async` /
  `replay_timerange_async` with `replay_of` lineage.
- `stream_monitor.py` — §8 metrics catalog registry + `snapshot()`.
- `result_sink.py` — append-only JSONL durable commit (flush+fsync).

`scripts/stream_benchmark.py` — sustained/burst/soak runner ->
`reports/stream_benchmark_<ts>.json`.

Deps: `redis>=5,<7` added to requirements; `REDIS_URL` in `.env.example`.

## Parity result

`tests/test_streaming_pipeline.py::TestParity` feeds all 650 sample
events (per-user ts order) through the streaming tracker+processor and
the batch `analyze_session_intelligence`; ALL score-contract columns
(`combined_score`, channel deviations, `risk_score`, `rule_score`,
`operating_mode`, `risk_level`) match exactly (atol 1e-9).

## Benchmark (scaled, honest)

`stream_benchmark.py --mode sustained --rate 100 --duration 15` on this
Windows dev machine (Python 3.14.6, pandas 3.1.1):
- 1471/1500 processed; **29 DLQ'd as late/out-of-order** (synthetic
  timestamp collisions — see limitations).
- effective ~80.5 ev/s; p50 0.79s, **p95 1.66s, p99 1.89s** —
  **misses the 100 ev/s and p95<1s targets**. No real-time claim.

## Known limitations / remaining work

- **Redis Streams untested live**: local Redis 3.0.504 lacks Streams
  (needs >=5.0). Adapter implemented per spec; tests auto-skip. Deploy
  target needs Redis >= 5 or the SQLite-queue fallback (documented ADR).
- **Late events**: events arriving before the per-stream emitted frontier
  go to DLQ (`late_event`) — replayable but scores reflect replay-time
  state. The 29 DLQ events in the benchmark came from synthetic data
  with non-monotonic per-user timestamps.
- **`is_duplicate_event` divergence**: batch flags ALL duplicate-content
  occurrences; stream flags 2nd+ causally. Affects sequence window
  membership only when exact duplicates exist.
- **Throughput/latency**: pandas per-event overhead; micro-batching
  already applied to batch-invariant channels. Needs profiling/possible
  vectorization or smaller reorder windows to hit targets.
- **60-min soak**: harness supports `--duration 3600`; only scaled runs
  executed so far.
- Prometheus export deferred (registry `snapshot()` only).
- Session-state checkpoint pickle is version-fragile across code changes
  (version field present; migrations not implemented).

## Test results

- New streaming tests: schema (13), queue asyncio+redis-skip (6),
  pipeline parity/recovery/backpressure/dedup/DLQ (6) — all passing.
- Full suite: see final report (486 baseline + new tests).
