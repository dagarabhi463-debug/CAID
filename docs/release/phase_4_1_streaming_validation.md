# Phase 4.1 — Streaming Reliability Validation

**Date:** 2026-11-12
**Commit:** `f2a3997` (v1.0-rc1, branch `master`)
**Scope:** verification only — no architecture, threshold, or algorithm changes.
**Machine:** Windows 11, Python 3.14.6, pandas 3.1.1 (same as reported baseline).

## 1. Independent verification performed

| Check | Command / method | Result |
|---|---|---|
| Full regression suite | `pytest tests -q` | **543 passed, 0 skipped, 4 warnings** — matches reported baseline exactly |
| Streaming unit tests | `pytest tests/test_streaming_{queue,ordering,duplicates,schema,health}.py -q` | 35 passed |
| Batch/stream parity + pipeline | `pytest tests/test_streaming_pipeline.py tests/test_score_consistency.py -q` | 28 passed |
| Redis Streams on **7.4.8** | `REDIS_URL=redis://127.0.0.1:6380/0 pytest tests/test_redis_streams.py -q` | 8 passed (live XAUTOCLAIM path) |
| Redis Streams on **5.0.14.1** | `REDIS_URL=redis://127.0.0.1:6379/0 pytest tests/test_redis_streams.py -q` | 8 passed (XPENDING/XCLAIM fallback path) |
| Fresh sustained benchmark | `stream_benchmark.py --mode sustained --rate 100 --duration 30 --batch-size 100 --max-hold-s 0.25` | 3000/3000 processed, **0 lost / 0 DLQ / 0 rejected / 0 duplicate**, 105.6 eps, p50 0.033s, p95 0.639s, p99 0.717s |

Report: `reports/stream_benchmark_20261112T052045Z.json`.

The fresh run corroborates the reported ~100 eps sustained envelope and sub-second p95.

## 2. Architecture verification (code review)

All items in `core/streaming/` were inspected line-by-line:

- **Asyncio queue** (`event_queue.py::AsyncioEventQueue`): bounded `asyncio.Queue` (default 50k), optional JSONL durable backing. Payload is persisted *before* the id is exposed; a rejected enqueue writes an ack tombstone so restarts cannot resurrect rejected events. `BackpressureError` on saturation — explicit, never silent.
- **Redis Streams** (`event_queue.py::RedisStreamsQueue`): `XADD` with `MAXLEN ~` retention, `XREADGROUP` (`>` new deliveries), `XACK` only after downstream commit, `XAUTOCLAIM` on Redis ≥ 6.2 with a verified `XPENDING`+`XCLAIM` fallback for 5.x. Consumer-group creation is idempotent (`BUSYGROUP` tolerated). Backpressure via `XLEN` vs `max_depth`.
- **Pending recovery**: `claim_pending` is invoked periodically by `run_worker` (`claim_pending_interval_s=30s`); asyncio implementation respects `min_idle_ms` so in-flight ordering-buffered events are not eagerly reclaimed.
- **Checkpoint restoration** (`event_checkpoint.py`): atomic `tempfile`+`os.replace` writes for JSON metadata and pickled tracker state; restores tracker state, `committed_ids`, and `pending_acks` via `worker_state`. Restore failures are counted (`caid_checkpoint_restore_errors`), never fatal.
- **Replay / delivery-key idempotency**: worker dedup key is `(event_id, replay_of)` through a bounded LRU (`committed_ids`); duplicates are acked without re-scoring. Operator DLQ replays bypass ingest dedup but remain downstream-idempotent.
- **DLQ accounting** (`event_replay.py::DeadLetterQueue`): append-only JSONL, `flush`+`fsync` commit, reason codes (`payload_decode`, `processing_error`, `late_event`, `retry_exhausted`). Benchmark `events_lost` counts only produced events with no scored row, DLQ record, duplicate status, or rejection.
- **Late / out-of-order handling** (`session_state.py::IncrementalSessionTracker`): per-stream reorder buffers — timed events per `user_id`, explicit-session events per `(user_id, session_id)`. Emission when `event_ts ≤ max_ts − reorder_window` or `max_hold_seconds` elapsed. Events earlier than the emitted frontier go to the DLQ as `late_event` — never silently dropped or retroactively inserted.
- **Batch/stream parity**: `TestParity.test_stream_scores_match_batch` feeds all 650 sample events through both paths; every score-contract column matches at `atol=1e-9`. The tracker deliberately replicates a batch `itertuples` quirk (prefix duration/rate always NaN/0) — documented in code.

## 3. Reported results — verdict

### 60-minute soak — CONFIRMED

`reports/stream_benchmark_20261112T045510Z.json`:

- 360,000 produced / 360,000 processed over 3,648.8s wall
- **0 lost, 0 DLQ, 0 rejected, 0 duplicates, 0 inflight**
- 98.7 effective eps; latency p50 0.30s, p95 0.88s, p99 1.86s (single 21.9s max outlier)
- Config: asyncio queue, batch_size 500, reorder_window_s 0.05, max_hold_s 0.25

Consistent with the fresh 30s run (§1). Preserved as reported.

### Burst target — CONFIRMED NOT MET (honest bound)

- 500 eps burst target **not achieved** as sustained throughput.
- Honest single-consumer ceiling: **~289 eps** in the preload-drain test (`stream_benchmark_20261112T033832Z.json`, batch_size=1000, `preload=True`), which measures pure consumer drain after the queue is filled.
- **Caveat documented (S41-01):** non-preload burst reports (e.g. `...T040147Z.json`, 410.1 eps) compute `effective_eps` over the *production window only* — post-production drain time is excluded. That figure must not be quoted as consumer throughput. No code change made; the harness behavior is preserved so reported evidence stays reproducible.

### Sustained vs burst distinction — VERIFIED

The benchmark distinguishes modes (`sustained` rate-limited producer, `burst` as-fast-as-possible producer, `soak` long sustained) and `preload` drain measurement. Sustained ≈ 98–106 eps across runs; burst ceiling ≈ 289 eps single-consumer.

### Redis 7.4.8 — VERIFIED LIVE

Both Redis versions available on this machine were exercised: 7.4.8 (port 6380, XAUTOCLAIM path) and 5.0.14.1 (port 6379, XPENDING/XCLAIM fallback). All 8 integration tests pass on each, including consumer-restart pending recovery and a full `run_worker` end-to-end over Redis Streams.

## 4. Findings

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| S41-01 | info | Non-preload `effective_eps` excludes drain time | Documented; benchmark harness unchanged |
| S41-02 | info | `committed_ids` dedup LRU is in-memory (cross-restart relies on deterministic re-score + consumer dedup) | Documented limitation |
| S41-03 | info | Checkpoint tracker state is pickle (atomic but version-sensitive, no migration) | Documented limitation |
| S41-04 | info | 500 eps burst target unmet; ~289 eps single-consumer ceiling | Accepted limitation → known-limitations doc in Phase 4.5 |

No confirmed defects. No code changes were required; every correction-worthy item is already covered by existing regression tests.

## 5. Updated streaming limitations

Carried forward into `docs/release/CAID_KNOWN_LIMITATIONS.md` (Phase 4.5):

1. **Throughput ceiling:** single-consumer streaming sustains ~100 eps comfortably (p95 < 1s) but peaks ~289 eps on drain; the 500 eps burst target is unmet. Partitioned workers/vectorization are out of scope.
2. **asyncio queue is dev/test grade** without `persist_path`; memory-only mode loses events on crash. Redis Streams is the durable production path.
3. **In-memory dedup:** `committed_ids` does not survive restart; redelivery after restart re-scores deterministically (identical rows) — consumers must dedup by `event_id`.
4. **Pickle checkpoints:** atomic but version-sensitive; no schema migration mechanism.
5. **Late-event policy:** events earlier than the emitted frontier are DLQ'd (`late_event`), not retroactively scored — a deliberate causal-consistency trade-off.
6. **Duplicate semantics:** bounded causal window — duplicates separated by more than the reorder window flag only later occurrences (documented divergence from batch).
7. **Metrics are in-process snapshots**; Prometheus export deferred.
8. **`max_hold_seconds` trades latency for ordering completeness** — it is the dominant p95 latency term.

## 6. Deliverables

- This document: `docs/release/phase_4_1_streaming_validation.md`
- Machine-readable results: `reports/final_streaming_validation.json`
- Fresh benchmark evidence: `reports/stream_benchmark_20261112T052045Z.json`
- Prior evidence (unchanged): `docs/audit/milestone_2_5_*.md`, `reports/stream_benchmark_*.json`

**Phase 4.1 status: COMPLETE — evidence verified, no contradictions found.**
