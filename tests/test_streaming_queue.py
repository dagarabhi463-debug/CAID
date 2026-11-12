"""Tests for core/streaming/event_queue.py — Milestone 2.5."""

import asyncio
import os

import pytest

from core.streaming.event_queue import (
    AsyncioEventQueue, BackpressureError, RedisStreamsQueue, make_queue)


def _payload(i):
    return {'event_id': f'e{i}', 'user_id': 'u', 'payload': i}


class TestAsyncioQueue:
    def test_enqueue_dequeue_ack(self):
        async def run():
            q = AsyncioEventQueue(max_depth=10)
            qid = await q.enqueue(_payload(1))
            msgs = await q.dequeue(1, timeout_s=0.1)
            assert len(msgs) == 1 and msgs[0].queue_id == qid
            assert msgs[0].payload['event_id'] == 'e1'
            assert await q.pending_count() == 1
            await q.ack([qid])
            assert await q.pending_count() == 0
            await q.close()
        asyncio.run(run())

    def test_backpressure_bounded(self):
        async def run():
            q = AsyncioEventQueue(max_depth=2)
            await q.enqueue(_payload(1))
            await q.enqueue(_payload(2))
            with pytest.raises(BackpressureError):
                await q.enqueue(_payload(3))
            await q.close()
        asyncio.run(run())

    def test_persistent_backing_recovers_unacked(self, tmp_path):
        async def run():
            path = str(tmp_path / 'q.jsonl')
            q1 = AsyncioEventQueue(max_depth=100, persist_path=path)
            await q1.enqueue(_payload(1))
            await q1.enqueue(_payload(2))
            msgs = await q1.dequeue(2, timeout_s=0.1)
            await q1.ack([msgs[0].queue_id])  # ack only first
            await q1.close()

            # Simulate crash: new instance over same backing files.
            q2 = AsyncioEventQueue(max_depth=100, persist_path=path)
            recovered = await q2.dequeue(10, timeout_s=0.1)
            payloads = sorted(m.payload['event_id'] for m in recovered)
            assert payloads == ['e2']  # e1 acked, e2 redelivered
            await q2.close()
        asyncio.run(run())

    def test_factory(self):
        async def run():
            q = make_queue({'impl': 'asyncio', 'max_depth': 5})
            assert isinstance(q, AsyncioEventQueue)
            await q.close()
        asyncio.run(run())


class TestRedisQueue:
    """Redis Streams tests — run only when a Redis server is reachable."""

    @pytest.fixture
    def redis_url(self):
        import os
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
        # Streams require Redis >= 5.0.
        try:
            import redis
            info = redis.Redis.from_url(url).info('server')
            major = int(str(info.get('redis_version', '0')).split('.')[0])
            if major < 5:
                pytest.skip(f'Redis {info.get("redis_version")} < 5.0 '
                            '(no Streams support)')
        except Exception as e:
            pytest.skip(f'Redis capability check failed: {e}')
        return url

    def test_stream_roundtrip_and_pending(self, redis_url):
        async def run():
            stream = f'caid:test:{os.getpid()}'
            q = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                  group='g1', consumer='c1', max_depth=1000)
            await q.enqueue(_payload(1))
            msgs = await q.dequeue(1, timeout_s=1.0)
            assert len(msgs) == 1
            assert msgs[0].payload['event_id'] == 'e1'
            assert await q.pending_count() >= 1
            await q.ack([msgs[0].queue_id])
            await q.close()
            import redis
            redis.Redis.from_url(redis_url).delete(stream)
        asyncio.run(run())

    def test_pending_recovery_via_autoclaim(self, redis_url):
        async def run():
            stream = f'caid:test2:{os.getpid()}'
            q1 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g1', consumer='dead-consumer',
                                   max_depth=1000)
            await q1.enqueue(_payload(9))
            msgs = await q1.dequeue(1, timeout_s=1.0)
            assert len(msgs) == 1
            await q1.close()  # 'crash' — entry stays pending

            q2 = RedisStreamsQueue(redis_url=redis_url, stream=stream,
                                   group='g1', consumer='new-consumer',
                                   max_depth=1000)
            claimed = await q2.claim_pending(min_idle_ms=0, count=10)
            assert any(m.payload['event_id'] == 'e9' for m in claimed)
            await q2.ack([m.queue_id for m in claimed])
            await q2.close()
            import redis
            redis.Redis.from_url(redis_url).delete(stream)
        asyncio.run(run())
