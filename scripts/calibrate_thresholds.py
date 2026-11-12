"""
Milestone 1.9 — Calibration workflow entry point.

    Training -> Calibration -> Threshold persistence -> Application loading
    -> Evaluation

Calibrates RiskEngine fixed thresholds on the CALIBRATION split ONLY and
persists them via core.calibration.save_calibration, fingerprint-bound to
the detector that produced the scores.

GUARD: refuses to run unless the dataset manifest declares the input
file's role as 'calibration'. Holdout/validation/train data cannot
silently set thresholds.

Usage:
    python scripts/calibrate_thresholds.py \
        --detector models/saved_models/v1_9/anomaly_detector.joblib \
        --data data/independent/calibration.csv \
        --manifest data/independent/dataset_manifest.json \
        [--behavioral-baseline models/saved_models/v1_9/behavioral_baselines.joblib] \
        [--output models/saved_models/calibration/thresholds_<fp>.joblib]
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from core.risk_engine import RiskEngine
from core.calibration import (save_calibration, dataset_manifest_hash,
                              detector_fingerprint, default_calibration_path)
from core.detection_rules import DetectionRulesEngine
from core.behavioural_analyzer import BehavioralAnalyzer
from core.behavioral_baseline import BehavioralBaselineManager


def _manifest_role(manifest_path, data_path):
    """Return the manifest-declared role for a data file, or None."""
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    target = os.path.basename(os.path.abspath(data_path))
    for name, info in manifest.get('splits', {}).items():
        if info.get('file') == target:
            return name, manifest
    return None, manifest


def run_pipeline_scores(df, detector, analyzer):
    """Produce risk_score rows using the same channel order as the app."""
    out = detector.predict(df)
    out = RiskEngine().process_events(out)
    out = DetectionRulesEngine().process_events(out)
    out = analyzer.analyze_user_behavior(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', required=True,
                    help='Path to fitted AnomalyDetector artifact')
    ap.add_argument('--data', required=True,
                    help='Calibration CSV (manifest role must be '
                         "'calibration')")
    ap.add_argument('--manifest', required=True,
                    help='dataset_manifest.json for role verification')
    ap.add_argument('--behavioral-baseline', default=None,
                    help='Optional BehavioralBaselineManager artifact; '
                         'defaults to the production baseline')
    ap.add_argument('--output', default=None)
    ap.add_argument('--percentiles', default='30,65,85',
                    help='medium,high,critical percentiles of the '
                         'calibration risk-score distribution')
    args = ap.parse_args()

    # --- Role guard: calibration data ONLY -------------------------------
    role, manifest = _manifest_role(args.manifest, args.data)
    if role != 'calibration':
        sys.exit(
            f"REFUSED: {args.data} has manifest role {role!r}, not "
            "'calibration'. Thresholds may only be calibrated on the "
            "calibration split — never on train/validation/holdout data.")

    percentiles = tuple(float(x) for x in args.percentiles.split(','))

    detector = AnomalyDetector().load(args.detector)
    df = pd.read_csv(args.data)

    if args.behavioral_baseline:
        mgr = BehavioralBaselineManager().load(args.behavioral_baseline)
        analyzer = BehavioralAnalyzer(mgr)
    else:
        from core.behavioural_analyzer import ensure_fitted_analyzer
        analyzer = ensure_fitted_analyzer()

    scored = run_pipeline_scores(df, detector, analyzer)

    engine = RiskEngine()
    thresholds = engine.calibrate_thresholds(scored, percentiles=percentiles)

    artifact = save_calibration(
        thresholds=thresholds,
        calibration_info=engine.calibration_info,
        detector=detector,
        dataset_role='calibration',
        manifest_hash=dataset_manifest_hash(manifest),
        path=args.output,
        percentiles=percentiles,
    )

    print(f"Detector fingerprint : {detector_fingerprint(detector)}")
    print(f"Threshold version    : {artifact['threshold_version']}")
    print(f"Thresholds           : {thresholds}")
    print(f"Calibration rows     : {engine.calibration_info['calibration_rows']}")
    print(f"Artifact             : "
          f"{args.output or default_calibration_path(detector)}")
    print("NOTE: thresholds are absolute risk_score cutoffs calibrated on "
          "the calibration split only. anomaly scores are calibrated "
          "anomaly scores, NOT probabilities.")


if __name__ == '__main__':
    main()
