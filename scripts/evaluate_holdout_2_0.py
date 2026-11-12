"""
Milestone 2.0 independent holdout evaluation.

Fits v2 artifacts on independent_v2/train.csv normals only, calibrates on
calibration.csv normals only, sanity-checks on validation.csv, and reports
holdout metrics for v1.9 vs session/sequence/full-v2 variants.

    .venv/Scripts/python.exe scripts/evaluate_holdout_2_0.py
"""

import json
import os
import sys
import time
import tracemalloc
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector, get_model_path
from models.session_anomaly_detector import SessionAwareAnomalyDetector
from core.behavioural_analyzer import BehavioralAnalyzer
from core.session_analyzer import SessionAnalyzer
from core.sequence_detection import SequenceAnalyzer, get_sequence_model_path
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine
from core.risk_engine import RiskEngine
from core.calibration import (save_calibration, apply_to_engine,
                              apply_detection_calibration, component_fingerprint,
                              detector_fingerprint, dataset_manifest_hash,
                              V2_ARTIFACT_VERSION)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'independent_v2')
ART_DIR = os.path.join(ROOT, 'models', 'saved_models', 'v2_0')
V19_DIR = os.path.join(ROOT, 'models', 'saved_models', 'v1_9')
CAL_DIR = os.path.join(ROOT, 'models', 'saved_models', 'calibration')


def _load(split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, f'{split}.csv'))


def _normals(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['label'] == 'normal'].copy()


def _metrics(df: pd.DataFrame, score_col: str = 'combined_score') -> Dict:
    y_true = df['label'].isin(['suspicious', 'high_risk']).astype(int)
    pred = df['risk_level'].isin(['High', 'Critical']).astype(int)
    score = pd.to_numeric(df.get(score_col, pd.Series(np.nan, index=df.index)),
                          errors='coerce').fillna(0.0)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc = float(roc_auc_score(y_true, score)) if y_true.nunique() > 1 else float('nan')
        pr = float(average_precision_score(y_true, score)) if y_true.nunique() > 1 else float('nan')
    except Exception:
        roc = pr = float('nan')
    return {
        'n': int(len(df)), 'positives': int(y_true.sum()),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': round(precision, 4), 'recall': round(recall, 4),
        'f1': round(f1, 4), 'fpr': round(fpr, 4),
        'specificity': round(specificity, 4),
        'roc_auc': round(roc, 4) if np.isfinite(roc) else None,
        'pr_auc': round(pr, 4) if np.isfinite(pr) else None,
    }


def _scenario_masks(df: pd.DataFrame, manifest: Dict) -> Dict[str, pd.Series]:
    phases = manifest.get('behavior_phases', {})
    cohorts = manifest.get('entity_cohorts', {})
    holdout_only = set(cohorts.get('holdout_only_users', []))
    masks = {
        'all': pd.Series(True, index=df.index),
        'unseen_users': df['user_id'].isin(holdout_only),
        'suspicious_label': df['label'].isin(['suspicious', 'high_risk']),
        'normal_label': df['label'] == 'normal',
    }
    for phase in ['stable_normal', 'normal_to_suspicious',
                  'suspicious_to_normal', 'mixed']:
        users = {u for u, p in phases.items() if p == phase}
        masks[f'phase_{phase}'] = df['user_id'].isin(users)
    # Common-IP coverage: mixed normal/positive users plus positive-only recall.
    shared_ips = set(cohorts.get('shared_ips', []))
    masks['common_ip_all'] = df['source_ip'].isin(shared_ips)
    masks['common_ip_normal'] = (df['label'] == 'normal') & \
        df['source_ip'].isin(shared_ips)
    masks['common_ip_positives'] = df['label'].isin(['suspicious', 'high_risk']) & \
        df['source_ip'].isin(shared_ips)
    return masks


def run_pipeline(df: pd.DataFrame, detector, behavioral, rules,
                 session_analyzer=None, sequence_analyzer=None,
                 engine_profile='legacy', risk_engine=None,
                 session_aware_detector=None) -> pd.DataFrame:
    out = df.copy()
    # Optional session-aware ML experiment uses the same causal prefix features.
    ml_detector = session_aware_detector or detector
    out = ml_detector.predict(out)
    risk = risk_engine or RiskEngine()
    out = risk.process_events(out)
    out = rules.process_events(out)
    if behavioral is not None:
        out = behavioral.analyze_user_behavior(out)
    if session_analyzer is not None:
        out = session_analyzer.analyze(out)
    if sequence_analyzer is not None:
        out = sequence_analyzer.analyze(out)
    engine = DetectionEngine(channel_profile=engine_profile)
    return engine.process_dataframe(out)


def _fit_v2(train_norm: pd.DataFrame, artifact_dir: str):
    os.makedirs(artifact_dir, exist_ok=True)
    detector = AnomalyDetector().fit(train_norm)
    detector.save(os.path.join(artifact_dir, 'anomaly_detector.joblib'))
    behavioral = BehavioralAnalyzer().fit(train_norm)
    behavioral.save(os.path.join(artifact_dir, 'behavioral_baselines.joblib'))
    session = SessionAnalyzer().fit(train_norm)
    session.save(os.path.join(artifact_dir, 'session_baselines.joblib'))
    sequence = SequenceAnalyzer().fit(train_norm)
    sequence.save(os.path.join(artifact_dir, 'sequence_model.joblib'))
    return detector, behavioral, session, sequence


def _load_v2(artifact_dir: str):
    detector = AnomalyDetector().load(os.path.join(artifact_dir, 'anomaly_detector.joblib'))
    behavioral = BehavioralAnalyzer().load(os.path.join(artifact_dir, 'behavioral_baselines.joblib'))
    session = SessionAnalyzer().load(os.path.join(artifact_dir, 'session_baselines.joblib'))
    sequence = SequenceAnalyzer().load(os.path.join(artifact_dir, 'sequence_model.joblib'))
    return detector, behavioral, session, sequence


def _calibrate_v2(detector, behavioral, session, sequence, cal_norm: pd.DataFrame,
                  train_norm: pd.DataFrame):
    rules = DetectionRulesEngine()
    # Process calibration normals through the full v2 channel pipeline.
    cal = run_pipeline(cal_norm, detector, behavioral, rules,
                       session_analyzer=session, sequence_analyzer=sequence,
                       engine_profile='v2_0', risk_engine=RiskEngine())
    risk = RiskEngine()
    risk.calibrate_thresholds(cal)
    engine = DetectionEngine(channel_profile='v2_0')
    # Detection thresholds: calibration-normal combined scores -> percentiles.
    scores = pd.to_numeric(cal['combined_score'], errors='coerce').dropna()
    if len(scores):
        p = np.percentile(scores, [30, 65, 85])
        engine.thresholds.update({'medium': float(p[0]), 'high': float(p[1]),
                                  'critical': float(p[2])})
    fingerprints = {
        'detector': detector_fingerprint(detector),
        'feature_baseline': component_fingerprint(detector.baseline_manager, 'feature_baseline'),
        'behavioral_baseline': component_fingerprint(behavioral.baseline_manager, 'behavioral_baseline'),
        'session_baseline': component_fingerprint(session.baseline_manager, 'session_baseline'),
        'sequence_model': component_fingerprint(sequence, 'sequence_model'),
    }
    manifest_path = os.path.join(DATA, 'dataset_manifest.json')
    artifact = save_calibration(
        risk.fixed_thresholds or {}, risk.calibration_info, detector=detector,
        dataset_role='calibration', manifest_hash=dataset_manifest_hash(manifest_path),
        path=os.path.join(CAL_DIR, f'v2_0_thresholds_{detector_fingerprint(detector)}.joblib'),
        artifact_version=V2_ARTIFACT_VERSION,
        fingerprints=fingerprints,
        channel_profile='v2_0',
        detection_thresholds=engine.thresholds,
        component_versions={
            'detector': str(getattr(detector, 'ARTIFACT_VERSION', '')),
            'behavioral': str(getattr(behavioral.baseline_manager, 'ARTIFACT_VERSION', '')),
            'session': str(getattr(session.baseline_manager, 'ARTIFACT_VERSION', '')),
            'sequence': str(getattr(sequence, 'ARTIFACT_VERSION', '')),
        },
        session_config=session.config.to_dict(),
        sequence_config=sequence.config.to_dict(),
    )
    return risk, engine, artifact


def _variant_results(df: pd.DataFrame, manifest: Dict) -> Dict[str, Dict]:
    masks = _scenario_masks(df, manifest)
    out = {}
    for name, mask in masks.items():
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        out[name] = _metrics(sub)
    return out


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    train, cal, val, holdout = (_load(s) for s in ('train', 'calibration', 'validation', 'holdout'))
    manifest = json.load(open(os.path.join(DATA, 'dataset_manifest.json'), encoding='utf-8'))
    train_norm, cal_norm = _normals(train), _normals(cal)

    print('Fitting v2 artifacts on train normals only...')
    detector, behavioral, session, sequence = _fit_v2(train_norm, ART_DIR)
    print('Calibrating on calibration normals only...')
    risk_engine, det_engine, cal_artifact = _calibrate_v2(
        detector, behavioral, session, sequence, cal_norm, train_norm)

    # Sanity on validation (not tuned).
    val_pred = run_pipeline(val, detector, behavioral, DetectionRulesEngine(),
                            session_analyzer=session, sequence_analyzer=sequence,
                            engine_profile='v2_0', risk_engine=RiskEngine())
    val_metrics = _metrics(val_pred)

    # Holdout variants.
    rules = DetectionRulesEngine()
    variants = {}
    timings = {}
    memory = {}

    def _time_variant(name, fn):
        tracemalloc.start()
        t0 = time.perf_counter()
        res = fn()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timings[name] = {'seconds': round(elapsed, 4),
                         'rows_per_sec': round(len(holdout) / max(elapsed, 1e-9), 1)}
        memory[name] = {'peak_mb': round(peak / 1e6, 2)}
        return res

    # v1.9 baseline artifacts on v2 holdout (existing pipeline, no session/sequence).
    def _v19():
        det19 = AnomalyDetector().load(os.path.join(V19_DIR, 'anomaly_detector.joblib'))
        beh19 = BehavioralAnalyzer().load(os.path.join(V19_DIR, 'behavioral_baselines.joblib'))
        risk19 = RiskEngine()
        try:
            apply_to_engine(risk19, detector=det19)
        except Exception:
            pass
        return run_pipeline(holdout, det19, beh19, DetectionRulesEngine(),
                            engine_profile='legacy', risk_engine=risk19)
    variants['v1_9_baseline'] = _time_variant('v1_9_baseline', _v19)

    variants['v2_ml_rules_behavioral'] = _time_variant(
        'v2_ml_rules_behavioral',
        lambda: run_pipeline(holdout, detector, behavioral, rules,
                             engine_profile='legacy', risk_engine=RiskEngine()))
    variants['v2_session'] = _time_variant(
        'v2_session',
        lambda: run_pipeline(holdout, detector, behavioral, rules,
                             session_analyzer=session, engine_profile='session',
                             risk_engine=RiskEngine()))
    variants['v2_sequence'] = _time_variant(
        'v2_sequence',
        lambda: run_pipeline(holdout, detector, behavioral, rules,
                             sequence_analyzer=sequence, engine_profile='sequence',
                             risk_engine=RiskEngine()))
    variants['v2_complete'] = _time_variant(
        'v2_complete',
        lambda: run_pipeline(holdout, detector, behavioral, rules,
                             session_analyzer=session, sequence_analyzer=sequence,
                             engine_profile='v2_0', risk_engine=RiskEngine()))

    results = {
        'milestone': '2.0',
        'dataset': 'data/independent_v2',
        'calibration_artifact': cal_artifact.get('threshold_version'),
        'validation_sanity': val_metrics,
        'holdout': {},
        'timings': timings,
        'memory': memory,
        'notes': 'Synthetic-only metrics; scores are calibrated anomaly scores, not probabilities.',
    }
    for name, dfv in variants.items():
        results['holdout'][name] = {
            'overall': _metrics(dfv),
            'scenarios': _variant_results(dfv, manifest),
            'operating_modes': dfv['operating_mode'].value_counts().to_dict(),
            'missing_channel_rate': float(
                dfv['missing_channels'].apply(len).gt(0).mean()) if 'missing_channels' in dfv else None,
        }

    out_path = os.path.join(DATA, 'eval_results_v2.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    perf_path = os.path.join(DATA, 'performance_results_v2.json')
    with open(perf_path, 'w', encoding='utf-8') as f:
        json.dump({'timings': timings, 'memory': memory,
                   'platform': {'python': sys.version, 'pandas': pd.__version__}},
                  f, indent=2)
    print(f'eval -> {out_path}')
    print(f'perf -> {perf_path}')
    print(json.dumps({k: v['overall'] for k, v in results['holdout'].items()}, indent=2))


if __name__ == '__main__':
    main()
