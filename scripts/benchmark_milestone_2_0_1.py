"""
Milestone 2.0.1 performance benchmark for CAID v2.0 pipeline components.

Uses data/independent_v2 train+calibration+validation (no holdout) tiled to
reach 1k/5k/10k/50k events. Tiling preserves contiguous session blocks but
repeats them; results are research-scale latency measurements, not SLAs.
"""

import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from core.behavioural_analyzer import BehavioralAnalyzer
from core.session_analyzer import SessionAnalyzer
from core.sequence_detection import SequenceAnalyzer
from core.session_reconstruction import SessionReconstructor
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'independent_v2')
ART = os.path.join(ROOT, 'models', 'saved_models', 'v2_0')
OUT = os.path.join(DATA, 'audit_2_0_1', 'performance_benchmark_2_0_1.json')
SIZES = [1000, 5000, 10000, 50000]
REPS = {1000: 3, 5000: 3, 10000: 2, 50000: 1}
WARMUP = 1


def _load(split):
    return pd.read_csv(os.path.join(DATA, f'{split}.csv'))


def make_df(n: int) -> pd.DataFrame:
    base = pd.concat([_load('train'), _load('calibration'), _load('validation')],
                     ignore_index=True)
    if n <= len(base):
        return base.head(n).copy()
    reps = int(np.ceil(n / len(base)))
    return pd.concat([base] * reps, ignore_index=True).head(n).copy()


def _timed(fn, reps: int):
    times = []
    peaks = []
    for _ in range(reps):
        tracemalloc.start()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(dt)
        peaks.append(peak / 1e6)
    return {
        'median_s': round(float(statistics.median(times)), 4),
        'p95_s': round(float(np.percentile(times, 95)), 4),
        'mean_s': round(float(np.mean(times)), 4),
        'peak_mb': round(float(max(peaks)), 2),
        'reps': reps,
    }


def run_full(df, detector, behavioral, session, sequence, rules):
    out = detector.predict(df.copy())
    out = rules.process_events(out)
    out = behavioral.analyze_user_behavior(out)
    out = session.analyze(out)
    out = sequence.analyze(out)
    return DetectionEngine(channel_profile='v2_0').process_dataframe(out)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    detector = AnomalyDetector().load(os.path.join(ART, 'anomaly_detector.joblib'))
    behavioral = BehavioralAnalyzer().load(os.path.join(ART, 'behavioral_baselines.joblib'))
    session = SessionAnalyzer().load(os.path.join(ART, 'session_baselines.joblib'))
    sequence = SequenceAnalyzer().load(os.path.join(ART, 'sequence_model.joblib'))
    recon = SessionReconstructor(session.config)
    rules = DetectionRulesEngine()

    report = {
        'milestone': '2.0.1',
        'machine': {'python': platform.python_version(), 'platform': platform.platform(),
                    'cpu_count': os.cpu_count()},
        'dataset': 'data/independent_v2 train+calibration+validation tiled to size; holdout excluded',
        'warmup_runs': WARMUP,
        'sizes': {},
    }

    for n in SIZES:
        print(f'benchmark n={n}', flush=True)
        df = make_df(n)
        # Warmup end-to-end once per size.
        _ = run_full(df, detector, behavioral, session, sequence, rules)

        # Precompute reconstruction once for transform/sequence isolation.
        recon_df = recon.reconstruct(df)
        scored_df = run_full(df, detector, behavioral, session, sequence, rules)

        reps = REPS[n]
        entry = {
            'rows': int(len(df)), 'users': int(df['user_id'].nunique()),
            'sessions_explicit': int((df['session_id'].fillna('') != '').sum()),
            'components': {
                'session_reconstruction': _timed(lambda: recon.reconstruct(df), reps),
                'session_baseline_transform': _timed(
                    lambda: session.baseline_manager.transform_events(recon_df), reps),
                'sequence_detection': _timed(lambda: sequence.analyze(recon_df), reps),
                'ml_inference': _timed(lambda: detector.predict(df.copy()), reps),
                'detection_fusion': _timed(
                    lambda: DetectionEngine(channel_profile='v2_0').process_dataframe(scored_df), reps),
                'total_pipeline': _timed(
                    lambda: run_full(df, detector, behavioral, session, sequence, rules), reps),
            },
        }
        report['sizes'][str(n)] = entry
        print(json.dumps({k: v['median_s'] for k, v in entry['components'].items()}, indent=2), flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'benchmark -> {OUT}')


if __name__ == '__main__':
    main()
