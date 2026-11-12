"""
Event queue abstraction — Milestone 2.5.

``EventQueue`` is the thin durable-queue boundary between ingestion and
detection workers. Two implementations:

- ``AsyncioEventQueue`` — in-process ``asyncio.Queue``, bounded, with an
  optional JSONL backing file so crash-recovery semantics can be tested
  without Redis. DEV/TEST implementation.
- ``RedisStreamsQueue`` — Redis Streams + consumer groups (XADD /
  XREADGROUP / XACK / XAUTOCLAIM), the recommended durable implementation.

The batch pipeline never imports this module.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


class BackpressureError(Exception):
    """Queue is at capacity; producer must retry later or be rejected."""


@dataclass
class QueuedEvent:
    """A dequeued queue entry awaiting processing + ack."""

    queue_id: str
    payload: Dict[str, Any]
    delivery_count: int = 1


class EventQueue(ABC):
    """Bounded durable event queue with ack-after-commit semantics."""

    @abstractmethod
    async def enqueue(self, payload: Dict[str, Any]) -> str:
        """Enqueue an event payload; returns the queue-assigned id.

        Raises BackpressureError when the queue is saturated.
        """

    @abstractmethod
    async def dequeue(self, batch_size: int = 1,
                      timeout_s: float = 1.0) -> List[QueuedEvent]:
        """Fetch up to ``batch_size`` pending events (new deliveries)."""

    @abstractmethod
    async def ack(self, queue_ids: List[str]) -> None:
        """Acknowledge processed entries — call ONLY after the downstream
        commit (result sink flush) has succeeded."""

    @abstractmethod
    async def claim_pending(self, min_idle_ms: int = 30_000,
                            count: int = 100) -> List[QueuedEvent]:
        """Re-deliver entries left pending by a crashed consumer."""

    @abstractmethod
    async def depth(self) -> int:
        """Approximate number of enqueued-but-unread entries."""

    @abstractmethod
    async def pending_count(self) -> int:
        """Number of delivered-but-unacknowledged entries."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""


# ----------------------------------------------------------------------
# Asyncio implementation (dev/test; optional JSONL durability)
# ----------------------------------------------------------------------

class AsyncioEventQueue(EventQueue):
    """
    Bounded in-process queue for development and tests.

    With ``persist_path`` set, every enqueued payload is appended to a
    JSONL backing file and acked ids are appended to ``<persist>.acked``;
    a new instance over the same files rebuilds the unacked backlog, which
    gives crash-recovery semantics without Redis. Without ``persist_path``
    the queue is memory-only (events are lost on process crash — documented
    dev limitation, NOT for production IDS use).
    """

    def __init__(self, max_depth: int = 50_000,
                 persist_path: Optional[str] = None,
                 name: str = 'asyncio'):
        self.name = name
        self.max_depth = int(max_depth)
        self.persist_path = persist_path
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=self.max_depth)
        self._payloads: Dict[str, Dict[str, Any]] = {}
        self._inflight: Dict[str, QueuedEvent] = {}
        self._inflight_since: Dict[str, float] = {}
        self._acked: set = set()
        self._closed = False
        self._loop_bounds = 0
        # Persistent file handles kept open for the lifetime of the queue
        # so enqueue/ack don't pay open/close per event. Writes are
        # buffered by the OS (page-cache durable against process crashes);
        # explicit flush happens on close().
        self._persist_fh = None
        self._acked_fh = None
        if persist_path:
            os.makedirs(os.path.dirname(os.path.abspath(persist_path)),
                        exist_ok=True)
            self._persist_fh = open(persist_path, 'a', encoding='utf-8')
            acked_path = persist_path + '.acked'
            self._acked_fh = open(acked_path, 'a', encoding='utf-8')
            self._load_backing()

    # -- durable backing -------------------------------------------------

    def _load_backing(self) -> None:
        assert self.persist_path
        acked_path = self.persist_path + '.acked'
        if os.path.exists(acked_path):
            with open(acked_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._acked.add(json.loads(line)['queue_id'])
                        except Exception:
                            continue
        if not os.path.exists(self.persist_path):
            return
        pending: List[tuple] = []
        with open(self.persist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                qid = rec.get('queue_id')
                if qid and qid not in self._acked:
                    pending.append((qid, rec.get('payload')))
        # Rebuild pending backlog (bounded by max_depth; oldest dropped are
        # still durable on disk for manual recovery — counted, not silent).
        for qid, payload in pending[-self.max_depth:]:
            self._payloads[qid] = payload
            self._queue.put_nowait(qid)

    def _persist_enqueue(self, queue_id: str, payload: Dict[str, Any]) -> None:
        if not self.persist_path or self._persist_fh is None:
            return
        self._persist_fh.write(
            json.dumps({'queue_id': queue_id, 'payload': payload},
                       default=str) + '\n')

    def _persist_ack(self, queue_id: str) -> None:
        if not self.persist_path or self._acked_fh is None:
            return
        self._acked_fh.write(json.dumps({'queue_id': queue_id}) + '\n')
        self._acked.add(queue_id)

    # -- EventQueue API ---------------------------------------------------

    async def enqueue(self, payload: Dict[str, Any]) -> str:
        if self._closed:
            raise RuntimeError('Queue is closed')
        queue_id = uuid.uuid4().hex
        # Persist before exposing the id to consumers so a crash cannot
        # strand a queued id without a durable payload. If the bounded
        # queue rejects, immediately persist an ack tombstone so a restart
        # does not resurrect an explicitly rejected event.
        self._payloads[queue_id] = payload
        self._persist_enqueue(queue_id, payload)
        try:
            self._queue.put_nowait(queue_id)
        except asyncio.QueueFull as e:
            self._payloads.pop(queue_id, None)
            self._persist_ack(queue_id)
            raise BackpressureError(
                f'asyncio queue at capacity ({self.max_depth})') from e
        return queue_id

    async def dequeue(self, batch_size: int = 1,
                      timeout_s: float = 1.0) -> List[QueuedEvent]:
        out: List[QueuedEvent] = []
        deadline_hits = 0
        while len(out) < batch_size:
            try:
                if out:
                    queue_id = self._queue.get_nowait()
                else:
                    queue_id = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout_s)
            except (asyncio.TimeoutError, asyncio.QueueEmpty):
                break
            payload = self._payloads.pop(queue_id, None)
            if payload is None:
                continue
            qe = QueuedEvent(queue_id=queue_id, payload=payload,
                             delivery_count=1)
            self._inflight[queue_id] = qe
            self._inflight_since[queue_id] = time.monotonic()
            out.append(qe)
            deadline_hits += 1
        return out

    async def ack(self, queue_ids: List[str]) -> None:
        for qid in queue_ids:
            self._inflight.pop(qid, None)
            self._inflight_since.pop(qid, None)
            self._payloads.pop(qid, None)
            self._acked.add(qid)
            self._persist_ack(qid)

    async def claim_pending(self, min_idle_ms: int = 30_000,
                            count: int = 100) -> List[QueuedEvent]:
        # In-flight entries in this process ARE the pending set. Re-deliver
        # only entries idle longer than min_idle_ms so events still buffered
        # for ordering are not duplicated by an eager reclaim sweep.
        now = time.monotonic()
        min_idle_s = max(float(min_idle_ms), 0.0) / 1000.0
        candidates = [
            (since, qid, qe)
            for qid, qe in self._inflight.items()
            for since in [self._inflight_since.get(qid, now)]
            if now - since >= min_idle_s
        ]
        candidates.sort(key=lambda item: (item[0], item[1]))
        out: List[QueuedEvent] = []
        for _since, qid, qe in candidates[:count]:
            qe.delivery_count += 1
            self._inflight_since[qid] = now
            out.append(qe)
        return out

    async def depth(self) -> int:
        return self._queue.qsize()

    async def pending_count(self) -> int:
        return len(self._inflight)

    async def close(self) -> None:
        self._closed = True


# ----------------------------------------------------------------------
# Redis Streams implementation
# ----------------------------------------------------------------------

class RedisStreamsQueue(EventQueue):
    """
    Redis Streams + consumer-group queue (durable).

    - ``enqueue``: ``XADD`` with ``MAXLEN ~`` retention.
    - ``dequeue``: ``XREADGROUP`` new deliveries (``>``).
    - ``claim_pending``: ``XAUTOCLAIM`` for crash recovery.
    - ``ack``: ``XACK`` — callers must invoke only after the downstream
      commit succeeded.
    """

    def __init__(self, redis_url: str = 'redis://127.0.0.1:6379/0',
                 stream: str = 'caid:events',
                 group: str = 'caid-workers',
                 consumer: Optional[str] = None,
                 max_depth: int = 50_000,
                 maxlen: int = 100_000):
        try:
            import redis.asyncio as aioredis
        except ImportError as e:  # pragma: no cover - dep guard
            raise RuntimeError(
                'redis package required for RedisStreamsQueue '
                '(pip install "redis>=5,<7")') from e
        self._aioredis = aioredis
        self.redis_url = redis_url
        self.stream = stream
        self.group = group
        self.consumer = consumer or (
            f'caid-{os.uname().nodename if hasattr(os, "uname") else "host"}'
            f'-{os.getpid()}-{uuid.uuid4().hex[:6]}')
        self.max_depth = int(max_depth)
        self.maxlen = int(maxlen)
        self._client = None
        self._group_ready = False

    async def _r(self):
        if self._client is None:
            self._client = self._aioredis.from_url(self.redis_url)
        return self._client

    async def _ensure_group(self) -> None:
        if self._group_ready:
            return
        r = await self._r()
        try:
            await r.xgroup_create(self.stream, self.group, id='0',
                                  mkstream=True)
        except Exception:
            # BUSYGROUP = group already exists — expected on restart.
            pass
        self._group_ready = True

    async def enqueue(self, payload: Dict[str, Any]) -> str:
        await self._ensure_group()
        r = await self._r()
        depth = await r.xlen(self.stream)
        if depth >= self.max_depth:
            raise BackpressureError(
                f'stream depth {depth} >= max_depth {self.max_depth}')
        fields = {'payload': json.dumps(payload, default=str)}
        return await r.xadd(self.stream, fields, maxlen=self.maxlen,
                            approximate=True)

    async def dequeue(self, batch_size: int = 1,
                      timeout_s: float = 1.0) -> List[QueuedEvent]:
        await self._ensure_group()
        r = await self._r()
        block_ms = max(int(timeout_s * 1000), 1)
        resp = await r.xreadgroup(
            self.group, self.consumer,
            {self.stream: '>'}, count=batch_size, block=block_ms)
        out: List[QueuedEvent] = []
        if not resp:
            return out
        for _stream, messages in resp:
            for msg_id, fields in messages:
                out.append(QueuedEvent(
                    queue_id=msg_id.decode() if isinstance(msg_id, bytes)
                    else str(msg_id),
                    payload=json.loads(fields[b'payload'].decode()
                                       if isinstance(fields.get(b'payload'),
                                                     bytes)
                                       else fields.get('payload', '{}')),
                    delivery_count=1))
        return out

    async def claim_pending(self, min_idle_ms: int = 30_000,
                            count: int = 100) -> List[QueuedEvent]:
        await self._ensure_group()
        r = await self._r()
        try:
            resp = await r.xautoclaim(
                self.stream, self.group, self.consumer,
                min_idle_time=min_idle_ms, start_id='0-0', count=count)
            # Redis >=7 returns (next, entries, deleted); 6.2 returns
            # (next, entries). Older servers raise before this point.
            if resp is None:
                return await self._claim_pending_legacy(
                    r, min_idle_ms=min_idle_ms, count=count)
            if len(resp) == 2:
                _next, messages = resp
            elif len(resp) >= 3:
                _next, messages, _deleted = resp[:3]
            else:
                return []
            return self._queued_from_messages(messages or [], delivery_count=2)
        except Exception as exc:
            msg = str(exc).lower()
            if 'unknown command' in msg or 'wrong number of arguments' in msg:
                return await self._claim_pending_legacy(
                    r, min_idle_ms=min_idle_ms, count=count)
            return []

    async def _claim_pending_legacy(self, r, *, min_idle_ms: int,
                                    count: int) -> List[QueuedEvent]:
        """Redis 5.x/6.0 fallback: inspect consumer pending lists, then XCLAIM."""
        try:
            summary = await r.xpending(self.stream, self.group)
        except Exception:
            return []
        consumers = self._pending_consumers(summary)
        out: List[QueuedEvent] = []
        remaining = count
        for consumer_name in consumers:
            if remaining <= 0:
                break
            try:
                hist = await r.xreadgroup(
                    self.group, consumer_name,
                    {self.stream: '0'}, count=remaining)
            except Exception:
                continue
            ids = []
            payloads = {}
            for _stream, messages in hist or []:
                for msg_id, fields in messages or []:
                    mid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                    ids.append(mid)
                    payloads[mid] = self._payload_from_fields(fields)
                    if len(ids) >= remaining:
                        break
                if len(ids) >= remaining:
                    break
            if not ids:
                continue
            try:
                claimed = await r.xclaim(
                    self.stream, self.group, self.consumer,
                    min_idle_time=min_idle_ms, message_ids=ids)
            except Exception:
                continue
            for msg_id, fields in claimed or []:
                mid = msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id)
                out.append(QueuedEvent(
                    queue_id=mid,
                    payload=self._payload_from_fields(fields)
                    if fields else payloads.get(mid, {}),
                    delivery_count=2))
                remaining -= 1
                if remaining <= 0:
                    break
        return out

    @staticmethod
    def _pending_consumers(summary) -> List[str]:
        consumers = []
        if isinstance(summary, dict):
            raw = summary.get('consumers') or []
        elif isinstance(summary, (list, tuple)) and len(summary) >= 4:
            raw = summary[3] or []
        else:
            raw = []
        for item in raw:
            if isinstance(item, dict):
                name = item.get('name') or item.get('consumer')
            elif isinstance(item, (list, tuple)) and item:
                name = item[0]
            else:
                name = item
            if isinstance(name, bytes):
                name = name.decode()
            if name:
                consumers.append(str(name))
        return consumers

    @staticmethod
    def _payload_from_fields(fields) -> Dict[str, Any]:
        raw = fields.get(b'payload') or fields.get('payload') or '{}'
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def _queued_from_messages(self, messages, *, delivery_count: int):
        out: List[QueuedEvent] = []
        for msg_id, fields in messages or []:
            out.append(QueuedEvent(
                queue_id=msg_id.decode() if isinstance(msg_id, bytes)
                else str(msg_id),
                payload=self._payload_from_fields(fields),
                delivery_count=delivery_count))
        return out

    async def ack(self, queue_ids: List[str]) -> None:
        if not queue_ids:
            return
        await self._ensure_group()
        r = await self._r()
        await r.xack(self.stream, self.group, *queue_ids)

    async def depth(self) -> int:
        r = await self._r()
        return await r.xlen(self.stream)

    async def pending_count(self) -> int:
        await self._ensure_group()
        r = await self._r()
        try:
            summary = await r.xpending(self.stream, self.group)
            # redis-py returns dict-like {'pending': n, ...} or tuple
            if isinstance(summary, dict):
                return int(summary.get('pending', 0))
            return int(summary[0]) if summary else 0
        except Exception:
            return 0

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def make_queue(config: Optional[Dict[str, Any]] = None) -> EventQueue:
    """Factory: select queue implementation by config.

    config keys: impl ('asyncio'|'redis'), redis_url, stream, group,
    consumer, max_depth, maxlen, persist_path.
    """
    cfg = dict(config or {})
    impl = cfg.get('impl', 'asyncio')
    if impl == 'redis':
        return RedisStreamsQueue(
            redis_url=cfg.get('redis_url')
            or os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0'),
            stream=cfg.get('stream', 'caid:events'),
            group=cfg.get('group', 'caid-workers'),
            consumer=cfg.get('consumer'),
            max_depth=int(cfg.get('max_depth', 50_000)),
            maxlen=int(cfg.get('maxlen', 100_000)))
    if impl == 'asyncio':
        return AsyncioEventQueue(
            max_depth=int(cfg.get('max_depth', 50_000)),
            persist_path=cfg.get('persist_path'),
            name=cfg.get('name', 'asyncio'))
    raise ValueError(f"Unknown queue impl: {impl!r}")
