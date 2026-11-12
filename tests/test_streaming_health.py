import json

from core.streaming.health import health_snapshot, latest_benchmark_report
from core.streaming.stream_monitor import MetricsRegistry


def test_health_snapshot_ok_with_benchmark(tmp_path):
    report = tmp_path / 'stream_benchmark_20260101T000000Z.json'
    report.write_text(json.dumps({
        'mode': 'soak', 'effective_eps': 100.0,
        'events_produced': 100, 'events_processed': 100,
        'events_dead_lettered': 0, 'events_rejected': 0,
        'events_duplicate': 0, 'events_lost': 0,
        'latency': {'p95': 0.5},
        'artifact_statuses': {'detector': 'loaded'},
        'calibration_statuses': {'detection': 'calibrated'},
    }))
    bench = latest_benchmark_report(str(tmp_path))
    snap = health_snapshot(benchmark=bench)
    assert snap['status'] == 'ok'
    assert snap['benchmark']['events_lost'] == 0


def test_health_snapshot_flags_loss_and_dlq():
    monitor = MetricsRegistry()
    monitor.dlq_depth(2)
    snap = health_snapshot(
        monitor=monitor,
        benchmark={'events_lost': 1, 'events_dead_lettered': 2,
                   'events_rejected': 0, 'effective_eps': 50.0,
                   'latency': {'p95': 1.2}},
        dlq={'present': True, 'depth': 2, 'reasons': {'late_event': 2}})
    assert snap['status'] == 'critical'
    assert snap['dlq']['reasons']['late_event'] == 2
