"""Redis Streams integration tests — Milestone 2.5.1.

Run only when REDIS_URL (default redis://127.0.0.1:6379/0) points at a
Redis >= 5.0 server (Streams). XAUTOCLAIM requires 6.2; on older
servers ``claim_pending`` falls back to XPENDING + XCLAIM.
Verified live against Redis 7.4.8 (redis-windows msys2 build) and
Redis 5.0.14.1.
"""

import asyncio
import os

import pytest

from core.streaming.event_queue import (
    AsyncioEventQueue, BackpressureError, RedisStreamsQueue, make_queue)


def _payload(i, **kw):
    p = {'event_id': f'e{i}', 'user_id': 'u', 'payload': i}
    p.update(kw)
    return p


@pytest.fixture
def redis_url():
    import socket
    from urllib.parse import urlparse
    url = os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0')
    host = urlparse(url).hostname or '127.0.0.1'
    port = urlparse(url).port or 6379
    try:
        s = socket.create_connection((host, port), timeout=1)
        s.close()
    except OSError:
        pytest.skip(f'Redis not reachable at {host}:{port}')
    try:
        import redis
        info = redis.Redis.from_url(url).info('server')
        version = str(info.get('redis_version', '0'))
        major = int(version.split('.')[0])
        if major < 5:
            pytest.skip(f'Redis {version} < 5.0 (no Streams support)')
    except Exception as e:
        pytest.skip(f'Redis capability check failed: {e}')
    return url


def _stream_name(tag):
    return f'caid:test:{tag}:{os.getpid()}'


def _cleanup(url, stream):
    import redis
    redis.Redis.from_url(url).delete(stream)


class TestRedisStreamsSemantics:
    def test_pending_not_redelivered_to_new_reads(self, redis_url):
        async def run():
            stream = _stream_name('pending')
            q = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                  group='g', consumer='c1')
            for i in range(3):
                await q.enqueue(_payload(i))
            msgs = await q.dequeue(10, timeout_s=0.5)
            assert len(msgs) == 3
            # Second '>' read must NOT redeliver pending entries.
            again = await q.dequeue(10, timeout_s=0.3)
            assert again == []
            assert await q.pending_count() == 3
            await q.ack([m.queue_id for m in msgs])
            assert await q.pending_count() == 0
            await q.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_acked_entries_never_redelivered(self, redis_url):
        async def run():
            stream = _stream_name('acked')
            q1 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g', consumer='c1')
            await q1.enqueue(_payload(1))
            msgs = await q1.dequeue(1, timeout_s=0.5)
            await q1.ack([msgs[0].queue_id])
            await q1.close()

            q2 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g', consumer='c2')
            # Even aggressive pending claims find nothing.
            claimed = await q2.claim_pending(min_idle_ms=0, count=10)
            assert claimed == []
            more = await q2.dequeue(10, timeout_s=0.2)
            assert more == []
            await q2.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_consumer_restart_recovers_pending(self, redis_url):
        async def run():
            stream = _stream_name('restart')
            q1 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g', consumer='dead-consumer')
            for i in range(5):
                await q1.enqueue(_payload(i))
            msgs = await q1.dequeue(5, timeout_s=0.5)
            assert len(msgs) == 5
            await q1.close()  # simulated crash: entries stay pending

            q2 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g', consumer='new-consumer')
            claimed = await q2.claim_pending(min_idle_ms=0, count=10)
            got = sorted(m.payload['event_id'] for m in claimed)
            assert got == [f'e{i}' for i in range(5)]
            await q2.ack([m.queue_id for m in claimed])
            assert await q2.pending_count() == 0
            await q2.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_group_recreation_is_idempotent(self, redis_url):
        async def run():
            stream = _stream_name('group')
            for _ in range(2):
                q = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                      group='same-group', consumer='c')
                await q.enqueue(_payload(1))
                await q.dequeue(1, timeout_s=0.3)
                await q.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_backpressure_on_stream_depth(self, redis_url):
        async def run():
            stream = _stream_name('backpressure')
            q = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                  group='g', consumer='c', max_depth=3)
            for i in range(3):
                await q.enqueue(_payload(i))
            # Depth counts XLEN (all entries incl. acked); fill beyond.
            with pytest.raises(BackpressureError):
                await q.enqueue(_payload(99))
            await q.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_payload_roundtrip_fidelity(self, redis_url):
        async def run():
            stream = _stream_name('fidelity')
            q = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                  group='g', consumer='c')
            payload = {'event_id': 'x', 'nested': {'a': [1, 2.5, 's']},
                       'ts': '2024-01-01T00:00:00'}
            await q.enqueue(payload)
            msgs = await q.dequeue(1, timeout_s=0.5)
            assert msgs[0].payload == payload
            await q.ack([msgs[0].queue_id])
            await q.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())

    def test_factory_redis_impl(self, redis_url):
        q = make_queue({'impl': 'redis', 'redis_url': redis_url,
                        'stream': _stream_name('factory'),
                        'max_depth': 10})
        assert isinstance(q, RedisStreamsQueue)


class TestRedisWorkerEndToEnd:
    """Full worker loop over Redis Streams (uses real detection artifacts)."""

    @pytest.fixture(scope='class')
    def processor(self):
        from core.streaming.event_processor import (
            StreamingDetectionProcessor)
        art = StreamingDetectionProcessor.load_artifacts(
            tracker_kwargs={'reorder_window_seconds': 0.0,
                            'max_hold_seconds': 0.0})
        return StreamingDetectionProcessor(art)

    def test_worker_processes_and_acks(self, redis_url, processor, tmp_path):
        from core.streaming.event_schema import validate_event
        from core.streaming.event_ingestion import EventIngestor
        from core.streaming.result_sink import JsonlResultSink
        from core.streaming.event_replay import DeadLetterQueue
        from core.streaming.stream_monitor import MetricsRegistry
        import pandas as pd

        async def run():
            stream = _stream_name('e2e')
            monitor = MetricsRegistry()
            queue = make_queue({'impl': 'redis', 'redis_url': redis_url,
                                'stream': stream, 'group': 'g',
                                'consumer': 'e2e-consumer',
                                'max_depth': 10_000})
            ing = EventIngestor(queue, source='redis-e2e', monitor=monitor)
            sink = JsonlResultSink(str(tmp_path / 'sink.jsonl'))
            dlq = DeadLetterQueue(str(tmp_path / 'dlq.jsonl'))

            n = 25
            base = pd.Timestamp('2024-02-01T10:00:00')
            raws = [{'timestamp': (base + pd.Timedelta(seconds=i))
                     .strftime('%Y-%m-%dT%H:%M:%S'),
                     'user_id': f'u{i % 4}', 'role': 'student',
                     'source_ip': '192.0.2.1', 'event_type': 'login',
                     'courses_accessed': 1, 'accounts_accessed': 1,
                     'api_requests': 1, 'data_mb': 1.0,
                     'token_age_minutes': 1, 'after_hours': 0}
                    for i in range(n)]
            for raw in raws:
                ev, reason = validate_event(raw, source='redis-e2e')
                assert ev is not None, reason
                res = await ing.ingest(ev.to_dict())
                assert res.status == 'accepted', res

            stop = asyncio.Event()
            worker = asyncio.create_task(processor.run_worker(
                queue, sink, dlq, batch_size=10, dequeue_timeout_s=0.2,
                sweep_interval_s=0.05, stop_event=stop))
            deadline = asyncio.get_event_loop().time() + 30
            while sink.records_written < n and \
                    asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.1)
            stop.set()
            await asyncio.wait_for(worker, timeout=20)

            assert sink.records_written == n
            assert dlq.depth() == 0
            assert await queue.pending_count() == 0
            sink.close()
            dlq.close()
            await queue.close()
            _cleanup(redis_url, stream)
        asyncio.run(run())
