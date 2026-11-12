"""Streaming health aggregation for CAID monitoring (observational only)."""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from typing import Dict, Optional

from core.streaming.event_replay import read_jsonl_events
from core.streaming.event_checkpoint import StreamCheckpoint
from core.streaming.stream_monitor import MetricsRegistry


def latest_benchmark_report(reports_dir: str = 'reports') -> Optional[Dict]:
    paths = glob.glob(os.path.join(reports_dir, 'stream_benchmark_*.json'))
    if not paths:
        return None
    path = max(paths, key=os.path.getmtime)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        data['_path'] = path
        return data
    except Exception:
        return None


def checkpoint_meta(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {'present': False}
    try:
        meta = StreamCheckpoint(path).load() or {}
        return {'present': True, 'meta': meta}
    except Exception as exc:
        return {'present': False, 'error': str(exc)}


def dlq_summary(path: Optional[str]) -> Dict:
    if not path or not os.path.exists(path):
        return {'present': False, 'depth': 0, 'reasons': {}}
    entries = read_jsonl_events(path)
    reasons = Counter(str(e.get('reason') or 'unknown') for e in entries)
    return {'present': True, 'depth': len(entries),
            'reasons': dict(reasons)}


def health_snapshot(*, monitor: Optional[MetricsRegistry] = None,
                    benchmark: Optional[Dict] = None,
                    checkpoint: Optional[Dict] = None,
                    dlq: Optional[Dict] = None) -> Dict:
    """Build a stable health snapshot; never affects detection scores."""
    monitor_snap = monitor.snapshot() if monitor is not None else {}
    bench = benchmark or {}
    ckpt = checkpoint or {'present': False}
    dlq_sum = dlq or {'present': False, 'depth': 0, 'reasons': {}}

    lost = int(bench.get('events_lost') or 0)
    dead = int(bench.get('events_dead_lettered') or dlq_sum.get('depth') or 0)
    rejected = int(bench.get('events_rejected') or 0)
    eps = float(bench.get('effective_eps') or 0.0)
    p95 = ((bench.get('latency') or {}).get('p95'))
    status = 'ok'
    if lost > 0 or dead > 0:
        status = 'warning' if lost == 0 else 'critical'
    elif rejected > 0 or (p95 is not None and p95 > 1.0):
        status = 'warning'
    elif ckpt.get('present') and not ckpt.get('meta', {}).get('saved_utc'):
        status = 'warning'

    return {
        'status': status,
        'benchmark': {
            'path': bench.get('_path'),
            'mode': bench.get('mode'),
            'effective_eps': bench.get('effective_eps'),
            'events_produced': bench.get('events_produced'),
            'events_processed': bench.get('events_processed'),
            'events_dead_lettered': bench.get('events_dead_lettered'),
            'events_rejected': bench.get('events_rejected'),
            'events_duplicate': bench.get('events_duplicate'),
            'events_lost': lost,
            'latency_p95': p95,
            'artifact_statuses': bench.get('artifact_statuses') or {},
            'calibration_statuses': bench.get('calibration_statuses') or {},
        },
        'queue': {
            'impl': bench.get('queue_impl'),
            'batch_size': bench.get('batch_size'),
            'depth': (monitor_snap.get('gauges') or {}).get('caid_queue_depth'),
            'pending': (monitor_snap.get('gauges') or {}).get('caid_dlq_depth'),
        },
        'checkpoint': ckpt,
        'dlq': dlq_sum,
        'monitor': monitor_snap,
        'notes': [
            'Observational monitoring only; does not affect detection scores.',
            'DLQ entries are durable terminal handling, not silent loss.',
        ],
    }
