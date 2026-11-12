"""
Streaming observability metrics — Milestone 2.5.

In-process metrics registry implementing the §8 observability catalog from
``docs/architecture/real_time_ids_architecture.md``. Metrics are operational
telemetry ONLY — they are never fed into detection scores.

Prometheus export is available via ``prometheus_exposition()`` (text
exposition format, no client dependency) and the stdlib HTTP server in
``core.streaming.metrics_server``. ``snapshot()`` exposes the registry
for tests, benchmarks, and UI surfaces.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, Optional, Tuple


def _labels_str(labels: Optional[Dict[str, str]]) -> str:
    if not labels:
        return ''
    inner = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return '{' + inner + '}'


def _escape_label_value(value: str) -> str:
    return (str(value).replace('\\', '\\\\')
            .replace('"', '\\"')
            .replace('\n', '\\n'))


import re

_LABEL_PAIR_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _split_key(key: str) -> Tuple[str, str]:
    """Split ``name{k="v",...}`` into (name, canonical labels string).

    Label values may themselves contain commas/quotes (escaped), so a
    naive comma split is unsafe — pairs are extracted with a regex.
    """
    if '{' not in key:
        return key, '{}'
    name, rest = key.split('{', 1)
    inner = rest.rsplit('}', 1)[0]
    pairs = []
    for m in _LABEL_PAIR_RE.finditer(inner):
        k, v = m.group(1), m.group(2)
        pairs.append(f'{k}="{_escape_label_value(v)}"')
    pairs.sort()
    return name, '{' + ','.join(pairs) + '}'


def _histogram_sum(samples: list) -> float:
    return float(sum(samples))


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

    # -- Prometheus exposition ----------------------------------------------

    def prometheus_exposition(self) -> str:
        """Render the registry in Prometheus text exposition format.

        Counters keep their ``_total`` names; gauges export verbatim;
        histograms export ``_count``/``_sum`` plus ``quantile`` gauges
        (p50/p95/p99) since raw buckets are not retained. No external
        dependency — scrapeable by any Prometheus server.
        """
        lines: list = []
        for key, value in sorted(self._counters.items()):
            name, labels = _split_key(key)
            lines.append(f'# TYPE {name} counter')
            lines.append(f'{name}{labels} {value}')
        for key, value in sorted(self._gauges.items()):
            name, labels = _split_key(key)
            lines.append(f'# TYPE {name} gauge')
            lines.append(f'{name}{labels} {value}')
        for key, samples in sorted(self._histograms.items()):
            name, labels = _split_key(key)
            pct = self._percentiles(samples)
            count = int(pct.get('count') or 0)
            lines.append(f'# TYPE {name}_count gauge')
            lines.append(f'{name}_count{labels} {count}')
            lines.append(f'# TYPE {name}_sum gauge')
            lines.append(f'{name}_sum{labels} {_histogram_sum(samples)}')
            for q, field in (('0.5', 'p50'), ('0.95', 'p95'),
                             ('0.99', 'p99')):
                val = pct.get(field)
                if val is None:
                    continue
                inner = labels[1:-1]
                qlabels = ('{' + (inner + ',' if inner else '')
                           + f'quantile="{q}"' + '}')
                lines.append(f'# TYPE {name} gauge')
                lines.append(f'{name}{qlabels} {val}')
        lines.append(f'caid_metrics_uptime_seconds '
                     f'{time.monotonic() - self._started}')
        return '\n'.join(lines) + '\n'

    def reset(self) -> None:
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()
        self._started = time.monotonic()
