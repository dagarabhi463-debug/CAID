"""
Streaming benchmark harness — Milestone 2.5.

Measures sustained/burst/soak throughput and detection latency over the
asyncio queue + incremental pipeline (Redis impl when REDIS_URL points at
a Redis >= 5 server). Synthetic events only — labels stripped.

Usage:
    .venv/Scripts/python.exe scripts/stream_benchmark.py \
        --mode sustained --rate 100 --duration 60
    .venv/Scripts/python.exe scripts/stream_benchmark.py \
        --mode soak --rate 100 --duration 600
    .venv/Scripts/python.exe scripts/stream_benchmark.py --mode burst --rate 500 --duration 10

Results -> reports/stream_benchmark_<UTC ts>.json
"""

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from core.log_processor import load_sample_dataset
from core.streaming.event_schema import validate_event
from core.streaming.event_queue import make_queue
from core.streaming.event_ingestion import EventIngestor
from core.streaming.event_processor import StreamingDetectionProcessor
from core.streaming.result_sink import JsonlResultSink
from core.streaming.event_replay import DeadLetterQueue
from core.streaming.stream_monitor import MetricsRegistry


def _hardware():
    return {
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'processor': platform.processor() or 'unknown',
        'machine': platform.machine(),
    }


def _events(n, seed=42, reorder_window=0):
    """Synthetic stream events derived from sample data (labels stripped).

    Events are stable-sorted by timestamp before returning so every
    per-user subsequence is monotonically non-decreasing — matching a
    realistic producer feed. ``reorder_window`` > 0 additionally applies
    a bounded local shuffle (each event may be displaced at most that
    many positions), deliberately injecting out-of-order arrivals for
    reorder-buffer/DLQ coverage.
    """
    import random
    rng = random.Random(seed)
    df = load_sample_dataset('all')
    df = df.drop(columns=['label'], errors='ignore')
    df = df.sort_values(['user_id', 'timestamp'], kind='mergesort')
    raws = df.to_dict('records')
    out = []
    i = 0
    while len(out) < n:
        r = dict(raws[i % len(raws)])
        # Vary timestamp slightly to avoid dedup collisions.
        ts = pd.to_datetime(r['timestamp']) + pd.Timedelta(
            seconds=(i // len(raws)) * 60)
        r['timestamp'] = ts.strftime('%Y-%m-%dT%H:%M:%S')
        ev, reason = validate_event(r, source='benchmark')
        if ev is not None:
            out.append(ev.to_dict())
        i += 1
        if i > n * 3:
            break
    out = out[:n]
    # Guarantee monotonic per-user arrival order: a global timestamp sort
    # makes every per-user subsequence non-decreasing (users stay
    # interleaved across the shared timeline).
    out.sort(key=lambda e: e['timestamp'])
    if reorder_window > 0:
        # Bounded local shuffle: displace each event by at most
        # ``reorder_window`` positions from its sorted slot.
        for i in range(len(out)):
            j = min(len(out) - 1, i + rng.randint(0, reorder_window))
            out[i], out[j] = out[j], out[i]
    return out


async def run_mode(mode, rate, duration, queue_impl, out_prefix,
                   batch_size=100, reorder_window=0,
                   reorder_window_s=0.05, max_hold_s=0.5,
                   preload=False):
    monitor = MetricsRegistry()
    queue = make_queue({'impl': queue_impl, 'max_depth': 50_000})
    ingestor = EventIngestor(queue, source='benchmark', monitor=monitor,
                             backpressure_timeout_s=2.0)
    sink_path = os.path.join('data', 'stream_results',
                             f'bench_{mode}_{int(time.time())}.jsonl')
    dlq_path = os.path.join('data', 'stream_dlq',
                            f'bench_{mode}_{int(time.time())}.jsonl')
    sink = JsonlResultSink(sink_path)
    dlq = DeadLetterQueue(dlq_path)

    art = StreamingDetectionProcessor.load_artifacts(
        tracker_kwargs={'reorder_window_seconds': reorder_window_s,
                        'max_hold_seconds': max_hold_s})
    processor = StreamingDetectionProcessor(art, monitor=monitor)

    total = int(rate * duration) if mode != 'burst' else int(rate * duration)
    events = _events(total, reorder_window=reorder_window)
    warmup = min(200, total // 10)

    stop = asyncio.Event()
    processed = {'n': 0}
    produced = 0

    async def _produce(upto):
        nonlocal produced
        while produced < upto:
            chunk = events[produced:produced + batch_size]
            for ev in chunk:
                res = await ingestor.ingest(ev)
                if res.status == 'rejected':
                    monitor.event_rejected(res.reason or 'unknown')
            produced += len(chunk)
            if mode == 'burst' and produced >= upto:
                break
            await asyncio.sleep(
                0 if mode == 'burst' else 1.0 / rate * batch_size)

    if preload:
        # Burst-drain test: fill the queue completely BEFORE the consumer
        # starts; effective_eps then measures pure consumer throughput.
        await _produce(total)
        t0 = time.monotonic()
        worker_task = asyncio.create_task(processor.run_worker(
            queue, sink, dlq, batch_size=batch_size,
            dequeue_timeout_s=0.2, sweep_interval_s=0.02,
            stop_event=stop))
        deadline = time.monotonic() + 600
        while sink.records_written < produced and \
                time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        elapsed = time.monotonic() - t0
        stop.set()
        try:
            await asyncio.wait_for(worker_task, timeout=60)
        except asyncio.TimeoutError:
            worker_task.cancel()
    else:
        # Warm-up
        for ev in events[:warmup]:
            await ingestor.ingest(ev)
        produced = warmup
        t0 = time.monotonic()
        worker_task = asyncio.create_task(processor.run_worker(
            queue, sink, dlq, batch_size=batch_size,
            dequeue_timeout_s=0.2, sweep_interval_s=0.02,
            stop_event=stop))
        await _produce(total)
        elapsed = time.monotonic() - t0

        # Drain: let worker flush tracker + commit remaining.
        deadline = time.monotonic() + 30
        while sink.records_written < produced and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        stop.set()
        try:
            await asyncio.wait_for(worker_task, timeout=30)
        except asyncio.TimeoutError:
            worker_task.cancel()

    snap = monitor.snapshot()
    dlq_entries = dlq.entries()
    dlq_reasons = {}
    for entry in dlq_entries:
        reason = str(entry.get('reason') or 'unknown')
        dlq_reasons[reason] = dlq_reasons.get(reason, 0) + 1
    rejected = sum(
        float(v) for k, v in snap['counters'].items()
        if k.startswith('caid_events_rejected_total'))
    duplicates = float(snap['counters'].get('caid_events_duplicate_total', 0.0))
    dead_lettered = dlq.depth()
    inflight = max(0, produced - sink.records_written - dead_lettered
                   - int(rejected) - int(duplicates))
    result = {
        'benchmark': 'milestone_2_5_streaming',
        'mode': mode, 'target_rate_eps': rate, 'duration_s': duration,
        'events_produced': produced,
        'events_processed': sink.records_written,
        'events_dead_lettered': dead_lettered,
        'dlq_reasons': dlq_reasons,
        'events_rejected': int(rejected),
        'events_duplicate': int(duplicates),
        'events_inflight': inflight,
        'events_lost': inflight,
        'dlq_depth': dlq.depth(),
        'queue_impl': queue_impl,
        'batch_size': batch_size,
        'reorder_window': reorder_window,
        'reorder_window_s': reorder_window_s,
        'max_hold_s': max_hold_s,
        'preload': preload,
        'warmup_events': warmup if not preload else 0,
        'wall_seconds': round(elapsed, 3),
        'effective_eps': round(sink.records_written / max(elapsed, 1e-9), 1),
        'latency': snap['histograms'].get(
            'caid_detection_latency_seconds'),
        'counters': snap['counters'],
        'gauges': snap['gauges'],
        'artifact_statuses': art.artifact_statuses,
        'calibration_statuses': art.calibration_statuses,
        'hardware': _hardware(),
        'redis_url': os.environ.get('REDIS_URL', 'redis://127.0.0.1:6379/0'),
        'notes': [
            'Synthetic events only; labels stripped.',
            'Scores are calibrated anomaly scores, not probabilities.',
            'Redis impl supports Redis >= 5.0; XAUTOCLAIM is used when '
            'available and XPENDING/XCLAIM fallback covers Redis 5.x.',
            'events_lost counts only produced events with no scored result, '
            'DLQ record, duplicate status, or explicit rejection.',
        ],
    }
    sink.close()
    dlq.close()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sustained', 'burst', 'soak'],
                    default='sustained')
    ap.add_argument('--rate', type=float, default=100.0)
    ap.add_argument('--duration', type=float, default=60.0,
                    help='seconds (soak default scaled; use 3600 for full)')
    ap.add_argument('--queue', choices=['asyncio', 'redis'],
                    default='asyncio')
    ap.add_argument('--batch-size', type=int, default=100)
    ap.add_argument('--reorder-window', type=int, default=0,
                    help='bounded positional shuffle for reorder testing')
    ap.add_argument('--reorder-window-s', type=float, default=0.05,
                    help='event-time reorder horizon (seconds)')
    ap.add_argument('--max-hold-s', type=float, default=0.5,
                    help='max wall-clock buffering before forced emit')
    ap.add_argument('--preload', action='store_true',
                    help='burst-drain: enqueue all events before consumer '
                         'starts (measures pure consumer throughput)')
    args = ap.parse_args()

    result = asyncio.run(run_mode(args.mode, args.rate, args.duration,
                                  args.queue, 'bench', args.batch_size,
                                  reorder_window=args.reorder_window,
                                  reorder_window_s=args.reorder_window_s,
                                  max_hold_s=args.max_hold_s,
                                  preload=args.preload))
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = os.path.join('reports', f'stream_benchmark_{ts}.json')
    os.makedirs('reports', exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: result[k] for k in
                      ('mode', 'events_produced', 'events_processed',
                       'events_dead_lettered', 'events_rejected',
                       'events_duplicate', 'events_inflight',
                       'events_lost', 'dlq_depth', 'effective_eps',
                       'latency')}, indent=2, default=str))
    print(f'Report: {out}')


if __name__ == '__main__':
    main()
