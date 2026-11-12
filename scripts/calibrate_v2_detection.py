"""
Milestone 2.4 closure — controlled v2_0 detection-calibration workflow.

Regenerates the fingerprint-bound v2 calibration artifact
(``v2_0_thresholds_<detector_fingerprint>.joblib``) for the CURRENT
production artifacts, replacing the stale artifact built for an older
fitted detector state.

DATA & SPLITS (reproducible, leakage-safe):
    Reuses scripts/benchmark_models.py deterministic entity-disjoint
    splits (SEED=42) over the current labeled corpora:
        data/raw/security_logs.csv
        data/processed/evaluation_data.csv
    The ``calibration`` bucket contains ONLY normal-labeled rows whose
    users are disjoint from every other split bucket. It is used for
    NOTHING except threshold derivation — never for fitting.

COMPONENTS calibrated (must match modules/session_intelligence.py):
    detector            : models/saved_models/anomaly_detector.joblib
    feature_baseline    : detector.baseline_manager
    behavioral_baseline : models/saved_models/behavioral_baselines.joblib
    session_baseline    : models/saved_models/session_baselines.joblib
    sequence_model      : models/saved_models/sequence_model.joblib

All components are loaded via the ensure_* helpers (AUD-002 lifecycle:
persisted artifacts load; missing/corrupt -> quarantine + refit on
training-normal data only).

THRESHOLDS:
    - risk (v1 fields): RiskEngine.calibrate_thresholds percentiles
      (30, 65, 85) of calibration-normal risk_score
    - detection (v2):   higher-tail percentiles (90, 99, 99.9) of
      calibration-normal combined_score under channel_profile='v2_0'
      to reduce false-positive High/Critical classifications.
      Alternatively, --detection-thresholds medium,high,critical sets
      absolute cutoffs explicitly. This is REQUIRED after the v2_0
      channel-weight recalibration (2026-11): under the new weights
      calibration-normal combined scores sit near zero, so percentile
      cutoffs cannot express the validation-derived operating point
      (medium=0.20 recall-band, high=0.30 F1-max, critical=0.45
      precision-band; see docs/research/phase_4_6_fusion_recalibration.md).

GUARANTEES:
    - dataset_role='calibration' enforced by save_calibration
    - detector + component fingerprints bound into the artifact
    - holdout/validation/sup_train data NEVER touched
    - writes reports/calibration_v2_manifest.json with provenance,
      including the count of calibration users that also appear in
      normal_activity.csv (detector training data) — disclosed, not a
      leakage violation (detector never trains on these rows)

Usage:
    .venv/Scripts/python scripts/calibrate_v2_detection.py [--check]

    --check : verify only — load artifacts, confirm fingerprints would
              match, print status; do NOT write the artifact.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from benchmark_models import (  # noqa: E402
    DATA_FILES, SEED, load_labeled_data, make_splits)

from core.artifact_lifecycle import (  # noqa: E402
    STATUS_LOADED, STATUS_REFIT_MISSING, STATUS_REFIT_CORRUPT)
from core.behavioural_analyzer import ensure_fitted_analyzer  # noqa: E402
from core.calibration import (  # noqa: E402
    V2_ARTIFACT_VERSION, apply_detection_calibration,
    apply_to_engine as apply_risk_calibration,
    component_fingerprint, dataset_manifest_hash, detector_fingerprint,
    save_calibration)
from core.detection_engine import DetectionEngine  # noqa: E402
from core.detection_rules import DetectionRulesEngine  # noqa: E402
from core.log_processor import load_sample_dataset  # noqa: E402
from core.risk_engine import RiskEngine  # noqa: E402
from core.sequence_detection import (  # noqa: E402
    ensure_fitted_sequence_analyzer)
from core.session_analyzer import ensure_fitted_session_analyzer  # noqa: E402
from models.anomaly_detector import ensure_model_exists  # noqa: E402

CAL_DIR = os.path.join(ROOT, 'models', 'saved_models', 'calibration')
MANIFEST_PATH = os.path.join(ROOT, 'reports',
                             'calibration_v2_manifest.json')
NORMAL_ACTIVITY = os.path.join(ROOT, 'data', 'raw',
                               'normal_activity.csv')
PERCENTILES = (30.0, 65.0, 85.0)
# Detection thresholds use a higher tail of the same calibration-normal
# combined_score distribution: the v1 risk percentiles produced excessive
# High/Critical false positives on normal scenario traffic.
DETECTION_PERCENTILES = (90.0, 99.0, 99.9)


def _load_components():
    """Load production components exactly as the runtime pipeline does."""
    detector = ensure_model_exists()
    behavioral = ensure_fitted_analyzer()
    normal = load_sample_dataset('normal')  # ignored once artifacts exist
    session = ensure_fitted_session_analyzer(train_df=normal)
    sequence = ensure_fitted_sequence_analyzer(train_df=normal)
    return detector, behavioral, session, sequence


def _fingerprints(detector, behavioral, session, sequence) -> Dict[str, str]:
    return {
        'detector': detector_fingerprint(detector),
        'feature_baseline': component_fingerprint(
            detector.baseline_manager, 'feature_baseline'),
        'behavioral_baseline': component_fingerprint(
            behavioral.baseline_manager, 'behavioral_baseline'),
        'session_baseline': component_fingerprint(
            session.baseline_manager, 'session_baseline'),
        'sequence_model': component_fingerprint(sequence, 'sequence_model'),
    }


def _run_v2_pipeline(df: pd.DataFrame, detector, behavioral, session,
                     sequence, engine: DetectionEngine) -> pd.DataFrame:
    """Identical channel order to modules/session_intelligence.py."""
    out = detector.predict(df.copy())
    risk = RiskEngine()
    apply_risk_calibration(risk, detector=detector)
    out = risk.process_events(out)
    out = DetectionRulesEngine().process_events(out)
    out = behavioral.analyze_user_behavior(out)
    out = session.analyze(out)
    out = sequence.analyze(out)
    return engine.process_dataframe(out)


def _component_versions(detector, behavioral, session, sequence) -> Dict:
    return {
        'detector': str(getattr(detector, 'ARTIFACT_VERSION', '')),
        'behavioral': str(getattr(
            behavioral.baseline_manager, 'ARTIFACT_VERSION', '')),
        'session': str(getattr(
            session.baseline_manager, 'ARTIFACT_VERSION', '')),
        'sequence': str(getattr(sequence, 'ARTIFACT_VERSION', '')),
    }


def _training_user_overlap(cal_users) -> Dict:
    """Disclose user_id overlap between calibration rows and the
    detector's training file (normal_activity.csv)."""
    try:
        train_users = set(pd.read_csv(NORMAL_ACTIVITY, usecols=['user_id'])
                          ['user_id'].astype(str))
    except Exception:
        return {'error': 'normal_activity.csv unreadable'}
    overlap = sorted(set(map(str, cal_users)) & train_users)
    return {'normal_activity_users': len(train_users),
            'overlap_users': len(overlap),
            'overlap_user_ids': overlap[:50],
            'note': 'detector trains ONLY on normal_activity.csv rows; '
                    'overlap means shared user_ids, not shared rows'}


def main() -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='verify fingerprints/artifacts without writing')
    ap.add_argument('--detection-thresholds', metavar='MEDIUM,HIGH,CRITICAL',
                    help='explicit detection cutoffs overriding the '
                         'percentile-derived values (e.g. 0.20,0.30,0.45)')
    args = ap.parse_args()

    # --- deterministic entity-disjoint splits -------------------------
    df = load_labeled_data()
    splits = make_splits(df)
    cal = splits['calibration']
    assert set(cal['label'].unique()) == {'normal'}, \
        'calibration split must contain normal rows only'
    users = {k: set(v['user_id'].astype(str)) for k, v in splits.items()}
    for a in users:
        for b in users:
            if a != b:
                assert not (users[a] & users[b]), f'user leak {a}/{b}'

    # --- production components ----------------------------------------
    detector, behavioral, session, sequence = _load_components()
    fingerprints = _fingerprints(detector, behavioral, session, sequence)
    statuses = {
        'detector': getattr(detector, 'artifact_status', None),
        'feature_baseline': getattr(
            getattr(detector, 'baseline_manager', None),
            'artifact_status', None),
        'behavioral_baseline': getattr(
            getattr(behavioral, 'baseline_manager', None),
            'artifact_status', None),
        'session_baseline': getattr(
            getattr(session, 'baseline_manager', None),
            'artifact_status', None),
        'sequence_model': getattr(sequence, 'artifact_status', None),
    }
    non_loaded = {k: v for k, v in statuses.items()
                  if v not in (STATUS_LOADED, None)}
    if non_loaded:
        print('WARNING: non-persisted component artifacts detected: '
              f'{non_loaded}')

    cal_path = os.path.join(
        CAL_DIR, f'v2_0_thresholds_{fingerprints["detector"]}.joblib')

    if args.check:
        existing = os.path.exists(cal_path)
        print(json.dumps({
            'check_only': True,
            'detector_fingerprint': fingerprints['detector'],
            'artifact_path': cal_path,
            'artifact_exists': existing,
            'component_statuses': statuses,
        }, indent=2))
        return {'check': True, 'artifact_exists': existing}

    # --- score calibration rows through the full v2 pipeline -----------
    engine = DetectionEngine(channel_profile='v2_0')
    scored = _run_v2_pipeline(cal, detector, behavioral, session,
                              sequence, engine)

    # Risk thresholds on calibration-normal risk scores.
    risk = RiskEngine()
    risk.calibrate_thresholds(scored, percentiles=PERCENTILES)

    # Detection thresholds: percentiles of calibration-normal
    # combined_score (same recipe as the removed M2.0 workflow).
    scores = pd.to_numeric(scored['combined_score'],
                           errors='coerce').dropna()
    if len(scores) == 0:
        raise RuntimeError('Calibration produced no finite combined_score')
    p = np.percentile(scores, list(DETECTION_PERCENTILES))
    detection_thresholds = {
        'medium': float(p[0]), 'high': float(p[1]), 'critical': float(p[2])}
    detection_thresholds_source = 'calibration_normal_percentiles'
    if args.detection_thresholds:
        parts = [float(x) for x in args.detection_thresholds.split(',')]
        if len(parts) != 3 or not parts[0] <= parts[1] <= parts[2]:
            raise ValueError(
                '--detection-thresholds must be MEDIUM,HIGH,CRITICAL '
                'with medium <= high <= critical')
        detection_thresholds = {
            'medium': parts[0], 'high': parts[1], 'critical': parts[2]}
        detection_thresholds_source = 'explicit_operating_point'
    engine.thresholds.update(detection_thresholds)

    # --- provenance manifest -------------------------------------------
    manifest = {
        'manifest': 'milestone_2_4_v2_detection_calibration',
        'dataset_role': 'calibration',
        'seed': SEED,
        'source_files': [os.path.relpath(p, ROOT) for p in DATA_FILES],
        'calibration_split': {
            'rows': int(len(cal)),
            'users': int(cal['user_id'].nunique()),
            'labels': cal['label'].value_counts().to_dict(),
        },
        'entity_disjoint_across_splits': True,
        'split_counts': {k: int(len(v)) for k, v in splits.items()},
        'normal_activity_user_overlap': _training_user_overlap(
            cal['user_id']),
        'risk_percentiles': list(PERCENTILES),
        'detection_percentiles': list(DETECTION_PERCENTILES),
        'detection_thresholds': detection_thresholds,
        'detection_thresholds_source': detection_thresholds_source,
        'risk_thresholds': dict(risk.fixed_thresholds),
        'fingerprints': fingerprints,
        'notes': [
            'Calibration rows are normal-labeled, entity-disjoint from '
            'all other benchmark splits; used ONLY for threshold '
            'derivation.',
            'Detector/session/sequence/behavioral components are the '
            'persisted production artifacts (fingerprint-bound).',
            'Scores are calibrated anomaly scores, NOT probabilities.',
        ],
    }
    os.makedirs(os.path.dirname(MANIFEST_PATH), exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, default=str)

    # --- persist fingerprint-bound v2 artifact -------------------------
    artifact = save_calibration(
        thresholds=risk.fixed_thresholds,
        calibration_info=risk.calibration_info,
        detector=detector,
        dataset_role='calibration',
        manifest_hash=dataset_manifest_hash(MANIFEST_PATH),
        path=cal_path,
        percentiles=PERCENTILES,
        artifact_version=V2_ARTIFACT_VERSION,
        fingerprints=fingerprints,
        channel_profile='v2_0',
        detection_thresholds=detection_thresholds,
        component_versions=_component_versions(
            detector, behavioral, session, sequence),
        session_config=session.config.to_dict(),
        sequence_config=sequence.config.to_dict(),
    )

    # --- immediate verification ----------------------------------------
    verify_engine = DetectionEngine(channel_profile='v2_0')
    loaded = apply_detection_calibration(
        verify_engine, detector=detector, path=cal_path,
        fingerprints=fingerprints)
    calibrated = loaded is not None and \
        verify_engine.calibration_status == 'calibrated'

    print(json.dumps({
        'threshold_version': artifact['threshold_version'],
        'artifact_path': cal_path,
        'detector_fingerprint': fingerprints['detector'],
        'calibration_rows': risk.calibration_info['calibration_rows'],
        'risk_thresholds': dict(risk.fixed_thresholds),
        'detection_thresholds': detection_thresholds,
        'combined_score_percentiles': {
            'p90': float(np.percentile(scores, 90)),
            'p99': float(np.percentile(scores, 99)),
            'p99_9': float(np.percentile(scores, 99.9)),
        },
        'component_statuses': statuses,
        'verification': 'calibrated' if calibrated
                        else 'FAILED',
        'manifest': os.path.relpath(MANIFEST_PATH, ROOT),
    }, indent=2, default=str))
    if not calibrated:
        raise RuntimeError('Post-save calibration verification failed')
    return artifact


if __name__ == '__main__':
    main()
