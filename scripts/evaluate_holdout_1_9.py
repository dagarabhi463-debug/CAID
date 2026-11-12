"""
Milestone 1.9 Phases 3+6 — Independent evaluation and detection metrics.

Pipeline (same order as the application):
    detector.predict -> RiskEngine (calibrated) -> DetectionRulesEngine
    -> BehavioralAnalyzer -> DetectionEngine

Datasets (data/independent/, entity-disjoint cohorts, seeded):
    train        -> fit AnomalyDetector + BehavioralAnalyzer (normal rows)
    calibration  -> calibrate RiskEngine thresholds (normal rows only)
    validation   -> secondary sanity metrics (not tuned on)
    holdout      -> ALL reported metrics (never used for fit/calibration)

Artifacts are written under models/saved_models/v1_9/ — the production
anomaly_detector.joblib / behavioral_baselines.joblib are NOT modified.

Metrics are computed on synthetic labeled data only and are NOT
real-world detection accuracy. anomaly_score_normalized and combined_score
are calibrated anomaly scores, NOT probabilities.

Writes data/independent/eval_results.json.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import (roc_auc_score, average_precision_score,
                             confusion_matrix)

from models.anomaly_detector import AnomalyDetector
from core.behavioural_analyzer import BehavioralAnalyzer
from core.behavioral_baseline import BehavioralBaselineManager
from core.risk_engine import RiskEngine
from core.calibration import (save_calibration, apply_to_engine,
                              dataset_manifest_hash, detector_fingerprint)
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine, RiskLevel

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IND = os.path.join(BASE, 'data', 'independent')
V19 = os.path.join(BASE, 'models', 'saved_models', 'v1_9')
CAL_DIR = os.path.join(BASE, 'models', 'saved_models', 'calibration')

POS_LABELS_BINARY = {'suspicious', 'high_risk'}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(df, detector, analyzer, engine):
    out = detector.predict(df)
    out = engine.process_events(out)
    out = DetectionRulesEngine().process_events(out)
    out = analyzer.analyze_user_behavior(out)
    out = DetectionEngine().process_dataframe(out)
    return out


def fit_v1_9_artifacts():
    """Train the Milestone-1.9 evaluation artifacts on train normals only."""
    os.makedirs(V19, exist_ok=True)
    train = pd.read_csv(os.path.join(IND, 'train.csv'))
    train_normal = train[train['label'] == 'normal'].copy()

    det_path = os.path.join(V19, 'anomaly_detector.joblib')
    beh_path = os.path.join(V19, 'behavioral_baselines.joblib')

    detector = AnomalyDetector()
    detector.fit(train_normal)
    detector.save(det_path)

    analyzer = BehavioralAnalyzer()
    analyzer.fit(train_normal)
    analyzer.save(beh_path)

    return detector, analyzer, det_path, beh_path


def calibrate_v1_9(detector, analyzer):
    """Calibrate thresholds on calibration-split NORMAL rows only."""
    cal = pd.read_csv(os.path.join(IND, 'calibration.csv'))
    cal_normal = cal[cal['label'] == 'normal'].copy()

    # Score calibration rows through the same channel pipeline
    out = detector.predict(cal_normal)
    out = RiskEngine().process_events(out)
    out = DetectionRulesEngine().process_events(out)
    out = analyzer.analyze_user_behavior(out)

    engine = RiskEngine()
    thresholds = engine.calibrate_thresholds(out)

    with open(os.path.join(IND, 'dataset_manifest.json'), encoding='utf-8') as f:
        manifest = json.load(f)
    artifact = save_calibration(
        thresholds=thresholds,
        calibration_info=engine.calibration_info,
        detector=detector,
        dataset_role='calibration',
        manifest_hash=dataset_manifest_hash(manifest),
        percentiles=(30.0, 65.0, 85.0),
    )
    return engine, artifact


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp, fp, tn, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) \
        if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'precision': round(precision, 4), 'recall': round(recall, 4),
            'f1': round(f1, 4), 'fpr': round(fpr, 4),
            'specificity': round(specificity, 4)}


def metrics_block(df, y_true, y_pred, score, mode):
    """Full metric set for one (scenario, mode) evaluation."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if len(y_true) == 0:
        return {'mode': mode, 'n': 0, 'note': 'empty scenario subset'}
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    out = _prf(tp, fp, tn, fn)
    out['mode'] = mode
    out['n'] = int(len(y_true))
    out['n_positive'] = int(y_true.sum())
    out['confusion_matrix'] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    s = pd.to_numeric(pd.Series(score), errors='coerce').to_numpy(float)
    ok = np.isfinite(s)
    if ok.sum() == len(s) and len(np.unique(y_true)) == 2 \
            and len(np.unique(y_true[ok])) == 2:
        out['roc_auc'] = round(float(roc_auc_score(y_true[ok], s[ok])), 4)
        out['pr_auc'] = round(float(
            average_precision_score(y_true[ok], s[ok])), 4)
        out['score'] = 'anomaly_score_normalized'
    else:
        out['roc_auc'] = None
        out['pr_auc'] = None
        out['score'] = 'anomaly_score_normalized'
        out['note'] = 'AUC unavailable: degenerate labels or scores'
    return out


def binary_y(df, mode):
    if mode == 'binary':
        return df['label'].isin(POS_LABELS_BINARY).to_numpy()
    return (df['label'] == 'high_risk').to_numpy()


def pred_from_detection(df, mode):
    """Positive prediction from the DetectionEngine risk_level output."""
    lvl = df['risk_level'].astype(str)
    if mode == 'binary':
        return lvl.isin(['High', 'Critical']).to_numpy()
    return (lvl == 'Critical').to_numpy()


def pred_from_if(df):
    return df['is_anomaly'].astype(bool).to_numpy()


def eval_frame(df, mode, score_col='anomaly_score_normalized'):
    """Evaluate one frame in one mode; returns dict of metric blocks."""
    y = binary_y(df, mode)
    score = df[score_col] if score_col in df.columns else np.nan
    return {
        'detection_risk_level': metrics_block(
            df, y, pred_from_detection(df, mode), score, mode),
        'is_anomaly_flag': metrics_block(
            df, y, pred_from_if(df), score, mode),
    }


def aggregate_by(df, key, mode):
    """Per-entity metrics: entity score = max event anomaly score;
    entity prediction = any event flagged High/Critical (detection) or
    any is_anomaly (IF flag)."""
    results = {}
    g = df.groupby(key)
    for name, sub in g:
        if len(sub) < 1:
            continue
        ent_score = pd.to_numeric(
            sub['anomaly_score_normalized'], errors='coerce').max()
        # entity label = modal label
        ent_label = sub['label'].mode().iloc[0]
        y = np.array([ent_label in POS_LABELS_BINARY if mode == 'binary'
                      else ent_label == 'high_risk'])
        det_pred = np.array([pred_from_detection(sub, mode).any()])
        if_pred = np.array([pred_from_if(sub).any()])
        results[str(name)] = {
            'n_events': int(len(sub)),
            'label': ent_label,
            'detection_risk_level': metrics_block(
                sub, np.repeat(y[0], len(sub)),
                np.repeat(det_pred[0], len(sub)),
                np.repeat(ent_score, len(sub)), mode),
            'is_anomaly_flag': metrics_block(
                sub, np.repeat(y[0], len(sub)),
                np.repeat(if_pred[0], len(sub)),
                np.repeat(ent_score, len(sub)), mode),
        }
    return results


def score_distribution(df):
    """Score stats by label (descriptive only — not probabilities)."""
    cols = ['anomaly_score_normalized', 'anomaly_score_raw', 'risk_score',
            'rule_score', 'behavioral_deviation', 'combined_score']
    out = {}
    for lab, sub in df.groupby('label'):
        out[lab] = {}
        for col in cols:
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors='coerce').dropna()
            if len(s) == 0:
                continue
            out[lab][col] = {
                'n': int(len(s)), 'mean': round(float(s.mean()), 4),
                'median': round(float(s.median()), 4),
                'std': round(float(s.std()), 4),
                'min': round(float(s.min()), 4),
                'max': round(float(s.max()), 4),
                'p25': round(float(s.quantile(.25)), 4),
                'p75': round(float(s.quantile(.75)), 4),
            }
    return out


# ---------------------------------------------------------------------------
# Scenarios (Phase 3)
# ---------------------------------------------------------------------------

def build_scenarios(holdout, cohorts, train_event_types):
    """Return {scenario_name: boolean mask/DataFrame} over holdout rows."""
    user_cohort = cohorts['users']
    ip_cohort = cohorts['ips']
    hu = holdout['user_id'].map(user_cohort)
    hi = holdout['source_ip'].map(ip_cohort)
    hour = pd.to_datetime(holdout['timestamp']).dt.hour

    known_user = hu == 'shared'
    known_ip = hi == 'shared'
    pos = holdout['label'].isin(POS_LABELS_BINARY)

    # train-normal reference for distribution-shift scenario
    train = pd.read_csv(os.path.join(IND, 'train.csv'))
    train_normal_api_p95 = float(
        train.loc[train['label'] == 'normal', 'api_requests']
        .quantile(0.95))

    scenarios = {
        'all_holdout': holdout.index == holdout.index,
        'known_users_known_ips': known_user & known_ip,
        'unseen_users_known_ips': (~known_user) & known_ip,
        'known_users_unseen_ips': known_user & (~known_ip),
        'unseen_users_unseen_ips': (~known_user) & (~known_ip),
        'new_event_type_combinations':
            ~holdout['event_type'].isin(train_event_types),
        'normal_unusual_hours':
            (holdout['label'] == 'normal') & ((hour < 6) | (hour > 22)),
        'suspicious_common_ips':
            (holdout['label'] == 'suspicious') & known_ip,
        'high_risk_seen_users':
            (holdout['label'] == 'high_risk') & known_user,
        'normal_distribution_shift':
            (holdout['label'] == 'normal')
            & (holdout['api_requests'] > train_normal_api_p95),
    }
    return {k: holdout[v] for k, v in scenarios.items()}


def multi_anomaly_sessions(holdout_proc):
    """Scenario 9: users with >=2 anomalous predictions in holdout."""
    anom = holdout_proc[holdout_proc['is_anomaly'].astype(bool)]
    counts = anom.groupby('user_id').size()
    multi_users = set(counts[counts >= 2].index)
    return holdout_proc[holdout_proc['user_id'].isin(multi_users)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    os.makedirs(V19, exist_ok=True)
    os.makedirs(CAL_DIR, exist_ok=True)

    print('=== Phase 3a: fit v1_9 artifacts on train normals ===')
    detector, analyzer, det_path, beh_path = fit_v1_9_artifacts()
    print(f'  detector -> {det_path}')
    print(f'  behavioral -> {beh_path}')

    print('=== Phase 4: calibrate on calibration normals ===')
    engine_cal, cal_artifact = calibrate_v1_9(detector, analyzer)
    print(f"  threshold_version={cal_artifact['threshold_version']}")
    print(f"  thresholds={cal_artifact['thresholds']}")

    # Load artifacts fresh (simulates application lifecycle)
    detector = AnomalyDetector().load(det_path)
    analyzer = BehavioralAnalyzer(
        BehavioralBaselineManager().load(beh_path))

    # Engine with persisted thresholds applied exactly like the app does
    risk_engine = RiskEngine()
    applied = apply_to_engine(risk_engine, detector=detector)
    assert applied is not None, 'calibration artifact failed to load'
    print(f"  applied status={risk_engine.calibration_status} "
          f"version={risk_engine.threshold_version}")

    print('=== Phase 6: evaluate holdout ===')
    holdout_raw = pd.read_csv(os.path.join(IND, 'holdout.csv'))
    holdout_proc = run_pipeline(holdout_raw, detector, analyzer, risk_engine)

    with open(os.path.join(IND, 'entity_cohorts.json'), encoding='utf-8') as f:
        cohorts = json.load(f)
    train = pd.read_csv(os.path.join(IND, 'train.csv'))
    train_event_types = set(train['event_type'].unique())

    scenarios = build_scenarios(holdout_raw, cohorts, train_event_types)
    scenarios['multiple_anomalies_one_session'] = multi_anomaly_sessions(
        holdout_proc)
    # scenario frames must carry processed columns: rebuild masks on proc
    proc_scenarios = {}
    key_cols = ['user_id', 'source_ip', 'timestamp', 'event_type',
                'api_requests', 'label']
    raw_indexed = holdout_raw.reset_index(drop=True)
    proc_indexed = holdout_proc.reset_index(drop=True)
    assert len(raw_indexed) == len(proc_indexed)
    for name, raw_sub in scenarios.items():
        if name == 'multiple_anomalies_one_session':
            proc_scenarios[name] = raw_sub
            continue
        idx = raw_sub.index
        proc_scenarios[name] = proc_indexed.loc[idx]

    results = {
        'milestone': '1.9',
        'note': ('Synthetic labeled data only; NOT real-world detection '
                 'accuracy. Scores are calibrated anomaly scores, NOT '
                 'probabilities.'),
        'artifacts': {
            'detector': det_path,
            'behavioral_baseline': beh_path,
            'detector_fingerprint': detector_fingerprint(detector),
        },
        'calibration': {
            'threshold_version': cal_artifact['threshold_version'],
            'thresholds': cal_artifact['thresholds'],
            'calibration_info': cal_artifact['calibration_info'],
            'dataset_role': cal_artifact['dataset_role'],
            'calibration_rows_used': int(
                (pd.read_csv(os.path.join(IND, 'calibration.csv'))
                 ['label'] == 'normal').sum()),
        },
        'operating_mode_distribution':
            holdout_proc['operating_mode'].value_counts().to_dict()
            if 'operating_mode' in holdout_proc.columns else {},
        'overall': {},
        'per_scenario': {},
        'per_user': {},
        'per_ip': {},
        'score_distributions': score_distribution(holdout_proc),
        'validation_split_sanity': {},
    }

    for mode in ('binary', 'high-risk'):
        results['overall'][mode] = eval_frame(holdout_proc, mode)

    for name, sub in proc_scenarios.items():
        entry = {}
        for mode in ('binary', 'high-risk'):
            entry[mode] = eval_frame(sub, mode)
        results['per_scenario'][name] = entry

    for mode in ('binary', 'high-risk'):
        results['per_user'][mode] = aggregate_by(holdout_proc, 'user_id', mode)
        results['per_ip'][mode] = aggregate_by(holdout_proc, 'source_ip', mode)

    # Validation split sanity (not used for any fitting/tuning)
    val_raw = pd.read_csv(os.path.join(IND, 'validation.csv'))
    val_proc = run_pipeline(val_raw, detector, analyzer, risk_engine)
    for mode in ('binary', 'high-risk'):
        results['validation_split_sanity'][mode] = eval_frame(val_proc, mode)

    out_path = os.path.join(IND, 'eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Console digest
    print('\n=== OVERALL (holdout, detection risk_level) ===')
    for mode in ('binary', 'high-risk'):
        m = results['overall'][mode]['detection_risk_level']
        print(f"  {mode:10s} n={m['n']} pos={m['n_positive']} "
              f"P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f} FPR={m['fpr']:.3f} "
              f"AUC={m.get('roc_auc')} PR-AUC={m.get('pr_auc')}")
    print('\n=== PER SCENARIO (binary, detection) ===')
    for name, entry in results['per_scenario'].items():
        m = entry['binary']['detection_risk_level']
        if m.get('n'):
            print(f"  {name:38s} n={m['n']:5d} pos={m['n_positive']:4d} "
                  f"P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"FPR={m['fpr']:.3f} AUC={m.get('roc_auc')}")
    print(f'\nWrote {out_path}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
