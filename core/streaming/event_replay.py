"""
Dead-letter queue + replay — Milestone 2.5.

DLQ is an append-only JSONL store (inspectable, replayable, never
auto-deleted). Replay re-feeds events through the IDENTICAL ingestion ->
ordering -> detection pipeline with ``replay_of=<event_id>`` lineage;
scores are unchanged because all engines are deterministic.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from core.streaming.event_ingestion import read_jsonl_events


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class DeadLetterQueue:
    """Append-only JSONL DLQ with reason codes."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._fh = None
        self.depth_count = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Count existing entries so depth() is accurate across restarts.
        self.depth_count = len(read_jsonl_events(path))
        self._fh = open(path, 'a', encoding='utf-8')

    def write(self, record: Dict) -> None:
        entry = dict(record)
        entry.setdefault('dead_lettered_at_utc', _utcnow())
        line = json.dumps(entry, default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + '\n')
            self._fh.flush()
            self.depth_count += 1

    def commit(self) -> None:
        with self._lock:
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def depth(self) -> int:
        return self.depth_count

    def entries(self) -> List[Dict]:
        self.commit()
        return read_jsonl_events(self.path)

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def replay_dlq(dlq_path: str, ingestor, *,
               include_reasons: Optional[Iterable[str]] = None) -> List:
    """
    Re-ingest DLQ entries through the identical pipeline.

    Entries are re-validated with ``replay_of=<original event_id>`` lineage.
    Dedup is bypassed for explicit replays (operator-initiated); scores are
    deterministic so reprocessing is idempotent.
    """
    entries = read_jsonl_events(dlq_path)
    results = []
    wanted = set(include_reasons) if include_reasons else None
    for entry in entries:
        reason = entry.get('reason')
        if wanted is not None and reason not in wanted:
            continue
        raw = entry.get('event') or entry.get('payload') or {}
        orig_id = (raw.get('event_id') if isinstance(raw, dict) else None) \
            or entry.get('event_id')
        import asyncio
        if asyncio.iscoroutinefunction(getattr(ingestor, 'ingest', None)):
            # Caller is expected to run inside a loop; provide sync wrapper.
            raise RuntimeError(
                'replay_dlq is async — use replay_dlq_async')
        results.append(ingestor.ingest(raw, replay_of=orig_id or ''))
    return results


async def replay_dlq_async(dlq_path: str, ingestor, *,
                           include_reasons: Optional[Iterable[str]] = None
                           ) -> List:
    entries = read_jsonl_events(dlq_path)
    results = []
    wanted = set(include_reasons) if include_reasons else None
    for entry in entries:
        reason = entry.get('reason')
        if wanted is not None and reason not in wanted:
            continue
        raw = entry.get('event') or entry.get('payload') or {}
        if not isinstance(raw, dict):
            continue
        # Strip envelope fields added by DLQ bookkeeping; keep the original
        # event payload (which already carries event_id for lineage).
        orig_id = raw.get('event_id', '')
        payload = {k: v for k, v in raw.items()
                   if k not in ('scored_at_utc', 'detection_latency_ms',
                                'combined_score', 'risk_level',
                                'operating_mode', 'detection_id')}
        results.append(await ingestor.ingest(
            payload, replay_of=orig_id, skip_dedup=True))
    return results


def replay_timerange(sink_path: str, ingestor, *,
                     start_utc: str, end_utc: str) -> List:
    """
    Re-feed scored events whose ``timestamp`` falls in [start, end) through
    the identical pipeline. Synchronous convenience wrapper — callers in
    async contexts should iterate ``ingestor.ingest`` directly.
    """
    raise NotImplementedError(
        'Use replay_timerange_async in async pipeline contexts')


async def replay_timerange_async(sink_path: str, ingestor, *,
                                 start_utc: str, end_utc: str) -> List:
    """Async time-range replay from a result-sink JSONL."""
    records = read_jsonl_events(sink_path)
    results = []
    for rec in records:
        ts = rec.get('timestamp')
        if ts is None:
            continue
        try:
            ts_norm = str(ts)
        except Exception:
            continue
        if not (start_utc <= ts_norm < end_utc):
            continue
        payload = {k: v for k, v in rec.items()
                   if k in _REPLAY_LOG_FIELDS or k in _REPLAY_OPTIONAL}
        results.append(await ingestor.ingest(payload))
    return results


_REPLAY_LOG_FIELDS = (
    'timestamp', 'user_id', 'role', 'source_ip', 'event_type',
    'courses_accessed', 'accounts_accessed', 'api_requests', 'data_mb',
    'token_age_minutes', 'after_hours',
)
_REPLAY_OPTIONAL = ('session_id', 'auth_result')
