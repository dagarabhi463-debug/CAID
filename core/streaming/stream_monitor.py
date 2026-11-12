"""
Streaming observability metrics — Milestone 2.5.

In-process metrics registry implementing the §8 observability catalog from
``docs/architecture/real_time_ids_architecture.md``. Metrics are operational
telemetry ONLY — they are never fed into detection scores.

Prometheus export is deferred; ``snapshot()`` exposes the registry for
tests, benchmarks, and UI surfaces.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional


def _labels_str(labels: Optional[Dict[str, str]]) -> str:
    if not labels:
        return ''
    inner = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return '{' + inner + '}'


class MetricsRegistry:
    """Counters, gauges, and histograms with labeled series."""

    def __init__(self):
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, list] = defaultdict(list)
        self._started = time.monotonic()

    # -- recording ---------------------------------------------------------

    def incr(self, name: str, value: float = 1.0,
             labels: Optional[Dict[str, str]] = None) -> None:
        self._counters[f'{name}{_labels_str(labels)}'] += value

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        self._gauges[f'{name}{_labels_str(labels)}'] = float(value)

    def gauge(self, name: str, value: float,
              labels: Optional[Dict[str, str]] = None) -> None:
        """Alias for set_gauge (checkpoint/worker call sites)."""
        self.set_gauge(name, value, labels=labels)

    def observe(self, name: str, value: float,
                labels: Optional[Dict[str, str]] = None) -> None:
        self._histograms[f'{name}{_labels_str(labels)}'].append(float(value))

    # -- catalog helpers ---------------------------------------------------

    def event_received(self) -> None:
        self.incr('caid_events_received_total')

    def event_rejected(self, reason: str) -> None:
        self.incr('caid_events_rejected_total', labels={'reason': reason})

    def event_duplicate(self) -> None:
        self.incr('caid_events_duplicate_total')

    def queue_depth(self, depth: int) -> None:
        self.set_gauge('caid_queue_depth', depth)

    def queue_wait(self, seconds: float) -> None:
        self.observe('caid_queue_wait_seconds', seconds)

    def throughput(self, eps: float) -> None:
        self.set_gauge('caid_processing_throughput_eps', eps)

    def detection_latency(self, seconds: float) -> None:
        self.observe('caid_detection_latency_seconds', seconds)

    def processing_failure(self, stage: str) -> None:
        self.incr('caid_processing_failures_total', labels={'stage': stage})

    def retry(self, reason: str) -> None:
        self.incr('caid_retries_total', labels={'reason': reason})

    def dlq_depth(self, depth: int) -> None:
        self.set_gauge('caid_dlq_depth', depth)

    def dead_letter(self, reason: str) -> None:
        self.incr('caid_events_dead_lettered_total', labels={'reason': reason})

    def stage_latency(self, stage: str, seconds: float) -> None:
        self.observe('caid_stage_latency_seconds', seconds,
                     labels={'stage': stage})

    def tracker_pending(self, counts: Dict[str, int]) -> None:
        for name, value in (counts or {}).items():
            self.set_gauge('caid_tracker_pending', value,
                           labels={'kind': str(name)})

    def checkpoint_saved(self) -> None:
        self.incr('caid_checkpoint_saves_total')
        self.set_gauge('caid_checkpoint_last_save_timestamp',
                       time.time())

    def artifact_status(self, component: str, status: Optional[str]) -> None:
        self.set_gauge('caid_artifact_status',
                       1.0 if status == 'loaded' else 0.0,
                       labels={'component': component, 'status': status or 'unknown'})

    def drift_status(self, feature: str, status: str) -> None:
        rank = {'ok': 0, 'warning': 1, 'critical': 2,
                'insufficient_reference': -1}.get(status, -1)
        self.set_gauge('caid_drift_status', rank,
                       labels={'feature': feature})

    def channel_available(self, channel: str, available: bool) -> None:
        self.set_gauge('caid_channel_available',
                       1.0 if available else 0.0, labels={'channel': channel})

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def _percentiles(samples: list) -> Dict[str, float]:
        if not samples:
            return {'count': 0, 'p50': None, 'p95': None, 'p99': None,
                    'min': None, 'max': None, 'mean': None}
        s = sorted(samples)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
            return s[idx]

        return {
            'count': n,
            'p50': pct(50), 'p95': pct(95), 'p99': pct(99),
            'min': s[0], 'max': s[-1],
            'mean': sum(s) / n,
        }

    def snapshot(self) -> Dict:
        """Point-in-time registry dump for tests/benchmarks/reports."""
        return {
            'uptime_seconds': time.monotonic() - self._started,
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
            'histograms': {
                name: self._percentiles(samples)
                for name, samples in self._histograms.items()
            },
        }

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._started = time.monotonic()
