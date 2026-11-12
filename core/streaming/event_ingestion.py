"""
Event ingestion — Milestone 2.5.

Producer-facing API: validate -> dedup -> enqueue with bounded backpressure.
Includes a JSONL file-tail adapter for development and benchmarks.

Production stream events must not carry synthetic ``label``/ground-truth
fields — rejected by ``validate_event`` (``allow_labels=False`` default).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from core.streaming.event_queue import BackpressureError, EventQueue
from core.streaming.event_schema import (
    BoundedDedupLRU, StreamingEvent, validate_event)
from core.streaming.stream_monitor import MetricsRegistry


@dataclass
class IngestResult:
    status: str                      # 'accepted' | 'duplicate' | 'rejected'
    reason: Optional[str] = None
    event_id: Optional[str] = None
    queue_id: Optional[str] = None


class EventIngestor:
    """
    Validates, deduplicates, and enqueues raw event dicts.

    Backpressure: ``queue.enqueue`` is bounded; when the queue is saturated
    the producer gets an explicit ``rejected``/``backpressure`` result after
    ``backpressure_timeout_s`` — never a silent drop.
    """

    def __init__(self, queue: EventQueue,
                 source: str = 'api',
                 allow_labels: bool = False,
                 dedup: Optional[BoundedDedupLRU] = None,
                 monitor: Optional[MetricsRegistry] = None,
                 backpressure_timeout_s: float = 5.0):
        self.queue = queue
        self.source = source
        self.allow_labels = allow_labels
        self.dedup = dedup if dedup is not None else BoundedDedupLRU()
        self.monitor = monitor or MetricsRegistry()
        self.backpressure_timeout_s = float(backpressure_timeout_s)

    async def ingest(self, raw: Dict, *,
                     source: Optional[str] = None,
                     replay_of: str = '',
                     skip_dedup: bool = False) -> IngestResult:
        """Validate + dedup + enqueue one raw event dict."""
        event, reason = validate_event(
            raw, source=source or self.source,
            allow_labels=self.allow_labels, replay_of=replay_of)
        if event is None:
            self.monitor.event_rejected(reason or 'unknown')
            return IngestResult(status='rejected', reason=reason)

        # Explicit operator replays bypass dedup: the event_id is
        # deterministic so lineage is preserved, and downstream processing
        # is idempotent.
        if not skip_dedup and self.dedup.check_and_add(event.event_id):
            self.monitor.event_duplicate()
            return IngestResult(status='duplicate', reason='duplicate_event_id',
                                event_id=event.event_id)

        # Bounded backpressure: try enqueue; on saturation wait up to the
        # configured timeout, then reject explicitly.
        deadline = time.monotonic() + self.backpressure_timeout_s
        while True:
            try:
                queue_id = await self.queue.enqueue(event.to_dict())
                break
            except BackpressureError:
                if time.monotonic() >= deadline:
                    self.monitor.event_rejected('backpressure')
                    return IngestResult(
                        status='rejected', reason='backpressure',
                        event_id=event.event_id)
                self.monitor.queue_depth(await self._safe_depth())
                await asyncio.sleep(0.01)

        self.monitor.event_received()
        return IngestResult(status='accepted', event_id=event.event_id,
                            queue_id=queue_id)

    async def ingest_batch(self, raws: Iterable[Dict],
                           **kwargs) -> List[IngestResult]:
        return [await self.ingest(r, **kwargs) for r in raws]

    async def _safe_depth(self) -> int:
        try:
            return await self.queue.depth()
        except Exception:
            return -1


async def tail_jsonl(path: str, ingestor: EventIngestor,
                     poll_interval_s: float = 0.05,
                     start_offset: int = 0,
                     stop_after_eof: bool = False,
                     max_events: Optional[int] = None):
    """
    Tail a JSONL file and ingest appended lines.

    Yields IngestResult objects. With ``stop_after_eof`` the loop exits at
    end-of-file (used by benchmarks/tests); otherwise it keeps polling.
    """
    offset = start_offset
    produced = 0
    while True:
        if not os.path.exists(path):
            if stop_after_eof:
                return
            await asyncio.sleep(poll_interval_s)
            continue
        with open(path, 'r', encoding='utf-8') as f:
            f.seek(offset)
            lines = f.readlines()
            offset = f.tell()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                yield IngestResult(status='rejected', reason='invalid_json')
                continue
            result = await ingestor.ingest(raw)
            yield result
            produced += 1
            if max_events is not None and produced >= max_events:
                return
        if stop_after_eof and not lines:
            return
        if not stop_after_eof:
            await asyncio.sleep(poll_interval_s)


def read_jsonl_events(path: str) -> List[Dict]:
    """Read all JSON objects from a JSONL file (benchmark/replay helper)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out
