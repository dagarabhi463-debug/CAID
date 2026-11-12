# Real-Time IDS Architecture — Design (Milestone 2.4, Phase 6)

**Status:** DESIGN ONLY — no streaming code implemented. The batch
pipeline (`app.py` → `modules/analysis_state.py` →
`modules/session_intelligence.py::analyze_session_intelligence`)
remains the operational path and is unchanged.

## 1. Goals for the streaming evolution

Continuous ingestion → real-time anomaly/rule/behavioral/session/
sequence detection → fused scoring → alerting — with reliability
properties the batch pipeline never needed: bounded latency,
durability, replay, and recovery.

## 2. Canonical event schema

Every ingested record normalizes to the existing 14-column security
log schema plus streaming metadata:

```
timestamp            : ISO-8601 UTC (required; parsed once at edge)
user_id              : str (required)
role                 : str
source_ip            : str
event_type           : str (validated vocabulary)
courses_accessed     : int
accounts_accessed    : int
api_requests         : int
data_mb              : float
token_age_minutes    : int
after_hours          : 0|1
session_id           : str (optional; explicit wins per
                       session_reconstruction rules)
auth_result          : 'success'|'failure' (optional)
label                : NEVER present in production stream
# --- streaming envelope (not fed to detectors) ---
event_id             : str  (ingestion-assigned, see §3)
ingested_at_utc      : str
source               : str  (producer identifier)
```

Detectors consume only the log-schema fields; envelope fields are
carried for identity/dedup/lineage and must never influence scores.

## 3. Event identity, ordering, dedup

- `event_id = sha1(source | event_ts | user_id | session_id |
  event_type | payload-hash)[:16]` — deterministic, content-derived.
- Ordering key: `(event_ts, event_id)` — total order, replay-safe.
- Dedup: keep a bounded LRU of recent `event_id`s (default 100k);
  duplicates are ACKed and counted (`events_duplicate`), not
  reprocessed. The existing `is_duplicate_event` content-hash flag
  in `session_reconstruction` remains the semantic duplicate
  detector for session logic.
- Late events: accepted up to `max_late_seconds` (default 300) into
  the session window; older events go to replay path, not dropped.

## 4. Pipeline stages

```
producers ──► ingest API ──► durable queue ──► detector workers
                                          │         │
                                          │         ▼
                                          │   score+channels (existing engines)
                                          │         │
                                          │         ▼
                                          │   alert router ──► incident/correlation
                                          │         │            (retrospective, batch)
                                          ▼         ▼
                                     checkpoint    dead-letter
                                     (offsets)     (poison events)
```

Each stage is idempotent: reprocessing an event reproduces identical
scores (all engines are deterministic given identical inputs and
frozen baselines — a prerequisite already satisfied).

## 5. Queue, backpressure, retry, DLQ

- **Queue:** see ADR §7 — recommended: Redis Streams (durable,
  consumer groups, pending-entry retries) behind a thin
  `event_queue.py` abstraction so asyncio queues can back tests.
- **Backpressure:** bounded queue (e.g., 50k events). Producers get
  `429/try-again` or block with timeout when full — never silent
  drops. Queue depth is a first-class metric.
- **Retry:** exponential backoff (100ms → 30s cap, jitter), max 5
  attempts for transient failures (DB/queue errors).
- **Dead-letter:** events failing validation or exhausting retries
  → DLQ with reason code; DLQ is inspectable and replayable, never
  deleted silently.
- **Checkpointing:** consumer-group pending offsets + periodic
  snapshot of session-state version → `event_checkpoint.py`.
- **Replay:** `event_replay.py` re-feeds DLQ or a time range through
  the same idempotent pipeline; replayed events get
  `replay_of=<event_id>` lineage, scores unchanged.
- **Failure recovery:** on restart, resume from last checkpoint;
  in-flight pending entries re-delivered by the queue; session
  baselines are immutable artifacts (no state to corrupt).
- **Latency measurement:** `ingested_at_utc` → `scored_at_utc`
  delta recorded per event (`detection_latency_ms`) — a metric,
  never a score input.

## 6. Proposed module layout (Milestone 2.5 — NOT implemented)

```
core/streaming/
    event_schema.py      # canonical schema + validation
    event_ingestion.py   # producer API / file-tail adapter
    event_queue.py       # queue abstraction (Redis Streams impl +
                         # asyncio impl for tests)
    event_processor.py   # per-event detection dispatch
    event_checkpoint.py  # offset/state snapshots
    event_replay.py      # DLQ + time-range replay
    stream_monitor.py    # metrics hooks (see §8)
```

Batch pipeline remains operational throughout; streaming is
additive.

## 7. ADR — queue technology

| Option | Durability | Ops cost | Fit |
|---|---|---|---|
| **asyncio in-process queue** | none (memory) | zero | tests/dev only; loses events on crash — unacceptable for IDS |
| **Redis Streams** | AOF/RDB persistence, consumer groups, pending retries, consumer lag | one Redis instance | **recommended** — smallest durable option meeting retry/DLQ/checkpoint needs |
| Kafka | strongest durability/scale | brokers, JVM ops, partitioning model | overkill for single-node research prototype; revisit only with documented multi-node need |
| SQLite-backed queue | durable, zero deps | polling latency | viable fallback if Redis unavailable; weaker consumer semantics |

**Recommendation:** Redis Streams behind the `event_queue.py`
abstraction. Rationale: durability + consumer-group retries +
lag metrics with one well-understood dependency; no Kafka/K8s
justified at this scale. asyncio queue retained as the test/dev
implementation of the same interface.

## 8. Observability metrics catalog (design — PLANNED)

Metrics are operational telemetry, **kept separate from detection
evidence** (never fed into scores). Prometheus/Grafana export is
incremental; nothing below is claimed as implemented.

| Metric | Type | Description |
|---|---|---|
| `caid_events_received_total` | counter | events accepted at ingest |
| `caid_events_rejected_total` | counter | validation rejects (by reason) |
| `caid_events_duplicate_total` | counter | dedup hits |
| `caid_queue_depth` | gauge | pending events |
| `caid_queue_wait_seconds` | histogram | enqueue→dequeue latency |
| `caid_processing_throughput_eps` | gauge | events/sec processed |
| `caid_detection_latency_seconds` | histogram | ingest→score latency (p50/p95/p99) |
| `caid_alert_latency_seconds` | histogram | score→alert/incident latency |
| `caid_processing_failures_total` | counter | worker exceptions (by stage) |
| `caid_retries_total` | counter | retry attempts (by reason) |
| `caid_dlq_depth` | gauge | dead-letter backlog |
| `caid_memory_bytes` / `caid_cpu_ratio` | gauge | process resources |
| `caid_model_inference_seconds` | histogram | per-channel inference duration |
| `caid_channel_available{channel}` | gauge | 0/1 per detection channel |
| `caid_drift_status{feature}` | gauge | DriftMonitor status ordinal |

Targets (engineering goals, NOT measured results): 100 ev/s
sustained, 500 ev/s burst, 60-min soak, zero loss of acknowledged
events, p95 detection latency < 1 s — all on documented hardware.

## 9. AuthN/AuthZ/audit-logging design (future deployment)

- **AuthN:** local deployment → Streamlit-native or reverse-proxy
  SSO; service accounts (API keys hashed at rest) for ingest
  producers. No credentials in code; `.env` pattern extended.
- **AuthZ:** roles `viewer` (dashboards), `analyst` (incidents,
  notes, reports), `operator` (config, recalibration, replay),
  `admin`. Enforced at page/API boundary, not in detection code.
- **Audit log:** append-only JSONL (`data/audit/audit.jsonl`) —
  who/what/when for incident mutations, config changes, replay,
  calibration loads, model artifact swaps. Detection scores are
  never audit-mutated; audit is metadata only.
- **Secrets:** extend `.env.example`; never log secrets; joblib
  artifacts accepted only from trusted local paths (documented
  trust boundary — see audit AUD-008 for the MITRE-bundle analog).
