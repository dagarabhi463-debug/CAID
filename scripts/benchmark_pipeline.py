"""
Milestone 1.7 performance benchmark.

Measures pipeline stage timings on progressively larger synthetic datasets
(100-10,000 events, tiled deterministically from the 650-event sample).
Reports median and p95 across repeated runs.

    python scripts/benchmark_pipeline.py
"""

import os
import sys
import time
import platform
import statistics
import psutil
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.log_processor import load_sample_dataset
from models.anomaly_detector import AnomalyDetector, get_model_path
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine
from core.risk_engine import RiskEngine
from core.behavioural_analyzer import ensure_fitted_analyzer

RUNS = 3
SIZES = [100, 500, 1000, 5000, 10000]


def tile_to(df, n):
    reps = (n // len(df)) + 1
    return pd.concat([df] * reps, ignore_index=True).head(n)


def bench(fn, *args, **kwargs):
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    return time.perf_counter() - t0


def main():
    print("=" * 78)
    print("CAID Milestone 1.7 Pipeline Benchmark (median / p95 over", RUNS, "runs)")
    print("=" * 78)
    print(f"Python: {sys.version.split()[0]} | pandas: {pd.__version__}")
    print(f"Platform: {platform.platform()} | CPU: {platform.processor()}")
    print(f"RAM: {psutil.virtual_memory().total/1e9:.1f} GB")
    print()

    base = load_sample_dataset('all')
    detector = AnomalyDetector()
    t0 = time.perf_counter()
    detector.load(get_model_path())
    print(f"Model load: {(time.perf_counter()-t0)*1000:.1f} ms\n")

    behavioral = ensure_fitted_analyzer()
    rules = DetectionRulesEngine()

    stages = ['ml_predict', 'risk', 'rules', 'behavioral', 'detection', 'TOTAL']
    results = {n: {s: [] for s in stages} for n in SIZES}

    for n in SIZES:
        df = tile_to(base, n)
        for r in range(RUNS):
            t = {}
            t_ml0 = time.perf_counter()
            df1 = detector.predict(df)
            t['ml_predict'] = time.perf_counter() - t_ml0

            t0_ = time.perf_counter(); df2 = RiskEngine().process_events(df1)
            t['risk'] = time.perf_counter() - t0_

            t0_ = time.perf_counter(); df3 = rules.process_events(df2)
            t['rules'] = time.perf_counter() - t0_

            t0_ = time.perf_counter(); df4 = behavioral.analyze_user_behavior(df3)
            t['behavioral'] = time.perf_counter() - t0_

            t0_ = time.perf_counter(); DetectionEngine().process_dataframe(df4)
            t['detection'] = time.perf_counter() - t0_

            t['TOTAL'] = t_ml0 and (time.perf_counter() - t_ml0)
            for s in stages:
                results[n][s].append(t[s])

    def med_p95(xs):
        # With RUNS=3, p95 ~= max (nearest-rank)
        return statistics.median(xs), max(xs)

    print(f"{'rows':>6} | " + " | ".join(f"{s:>16}" for s in stages))
    print("-" * (7 + 19 * len(stages)))
    for n in SIZES:
        cells = []
        for s in stages:
            xs = results[n][s]
            m, p = med_p95(xs)
            cells.append(f"{m*1000:7.0f}/{p*1000:7.0f}m")
        print(f"{n:>6} | " + " | ".join(f"{c:>16}" for c in cells))


if __name__ == '__main__':
    main()
