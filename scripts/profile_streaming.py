"""Streaming hot-path profiler — Milestone 2.5.1.

Times each stage of the streaming detection path over the parity corpus
(and optionally synthetic sustained input):

- tracker.ingest (ordering/session/sequence causal scoring)
- process_rows sub-channels: detector.predict, risk engine, rules,
  behavioral, detection_engine fusion, output serialization prep
- queue wait / sink fsync are measured separately by the benchmark.

Usage:
    .venv/Scripts/python.exe scripts/profile_streaming.py [--events 650]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from core.log_processor import load_sample_dataset
from core.streaming.event_schema import validate_event
from core.streaming.event_processor import StreamingDetectionProcessor


def _corpus_events(n):
    df = load_sample_dataset('all')
    df = df.drop(columns=['label'], errors='ignore')
    df = df.sort_values(['user_id', 'timestamp'], kind='mergesort')
    raws = df.to_dict('records')
    out = []
    i = 0
    while len(out) < n and i < len(raws) * 2:
        r = dict(raws[i % len(raws)])
        ts = pd.to_datetime(r['timestamp']) + pd.Timedelta(
            seconds=(i // len(raws)) * 60)
        r['timestamp'] = ts.strftime('%Y-%m-%dT%H:%M:%S')
        ev, reason = validate_event(r, source='profile')
        if ev is not None:
            out.append(ev)
        i += 1
    # Per-user monotonic arrival order (global ts sort).
    out.sort(key=lambda e: e.timestamp)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', type=int, default=650)
    ap.add_argument('--repeat', type=int, default=3,
                    help='timed repetitions (first is warmup)')
    args = ap.parse_args()

    art = StreamingDetectionProcessor.load_artifacts(
        tracker_kwargs={'reorder_window_seconds': 0.0,
                        'max_hold_seconds': 0.0})
    processor = StreamingDetectionProcessor(art)
    events = _corpus_events(args.events)
    print(f'{len(events)} events; artifacts: '
          f'{art.calibration_statuses}')

    # Warmup
    tr = art.tracker
    rows = []
    for ev in events:
        r, _ = tr.ingest(ev)
        rows.extend(r)
    r2, _ = tr.flush_all()
    rows.extend(r2)
    processor.process_rows(rows)

    stage = defaultdict(float)
    counts = defaultdict(int)

    for rep in range(args.repeat):
        # Fresh tracker state per rep for identical work.
        from core.streaming.session_state import IncrementalSessionTracker
        art.tracker = IncrementalSessionTracker(
            art.session_analyzer.baseline_manager, art.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        tr = art.tracker

        t0 = time.perf_counter()
        rows = []
        for ev in events:
            r, _ = tr.ingest(ev)
            rows.extend(r)
        r2, _ = tr.flush_all()
        rows.extend(r2)
        stage['tracker.ingest'] += time.perf_counter() - t0
        counts['events'] += len(events)

        # Sub-channel timing inside process_rows: replicate its stages.
        t0 = time.perf_counter()
        df = pd.DataFrame(rows)
        stage['process_rows:DataFrame'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        df2 = art.detector.predict(df)
        stage['process_rows:detector.predict'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        risk_scores = []
        for _, row in df2.iterrows():
            score, factors = art.risk_engine.calculate_risk_score(row)
            risk_scores.append(score)
        stage['process_rows:risk_engine'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        df3 = art.rules_engine.process_events(df2)
        stage['process_rows:rules_engine'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        df4 = art.behavioral.analyze_user_behavior(df3)
        stage['process_rows:behavioral'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        df5 = art.detection_engine.process_dataframe(df4)
        stage['process_rows:detection_engine'] += time.perf_counter() - t0

        t0 = time.perf_counter()
        out = processor.process_rows(rows)
        stage['process_rows:TOTAL'] += time.perf_counter() - t0

    n = max(counts['events'], 1)
    print(f'\nPer-event stage costs (avg over {args.repeat} reps, {n} events):')
    for name in sorted(stage, key=lambda k: -stage[k]):
        total = stage[name]
        print(f'  {name:38s} {total / n * 1000:8.3f} ms/event '
              f'({total:7.2f}s total)')


if __name__ == '__main__':
    main()
