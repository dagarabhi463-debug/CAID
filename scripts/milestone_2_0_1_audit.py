"""
Milestone 2.0.1 independent audit: apples-to-apples metrics, channel ablation,
paired/user-clustered bootstrap, fusion-weight sensitivity, and calibration
fingerprint verification.

No holdout tuning: calibration splits are used only for threshold fitting;
validation is used only for fusion-weight design comparison.
"""

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from models.session_anomaly_detector import SessionAwareAnomalyDetector
from core.behavioural_analyzer import BehavioralAnalyzer
from core.session_analyzer import SessionAnalyzer
from core.sequence_detection import SequenceAnalyzer
from core.detection_rules import DetectionRulesEngine
from core.detection_engine import DetectionEngine
from core.calibration import (CalibrationUnavailable, component_fingerprint,
                              detector_fingerprint, dataset_manifest_hash,
                              load_calibration)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_V2 = os.path.join(ROOT, 'data', 'independent_v2')
DATA_V19 = os.path.join(ROOT, 'data', 'independent')
ART_V2 = os.path.join(ROOT, 'models', 'saved_models', 'v2_0')
ART_V19 = os.path.join(ROOT, 'models', 'saved_models', 'v1_9')
OUT_DIR = os.path.join(DATA_V2, 'audit_2_0_1')
SEED = 20260220


def _load(base: str, split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(base, f'{split}.csv'))


def _manifest(base: str) -> Dict:
    path = os.path.join(base, 'dataset_manifest.json')
    return json.load(open(path, encoding='utf-8')) if os.path.exists(path) else {}


def _normals(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['label'] == 'normal'].copy()


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def load_v19():
    det = AnomalyDetector().load(os.path.join(ART_V19, 'anomaly_detector.joblib'))
    beh = BehavioralAnalyzer().load(os.path.join(ART_V19, 'behavioral_baselines.joblib'))
    return det, beh


def load_v2():
    det = AnomalyDetector().load(os.path.join(ART_V2, 'anomaly_detector.joblib'))
    beh = BehavioralAnalyzer().load(os.path.join(ART_V2, 'behavioral_baselines.joblib'))
    sess = SessionAnalyzer().load(os.path.join(ART_V2, 'session_baselines.joblib'))
    seq = SequenceAnalyzer().load(os.path.join(ART_V2, 'sequence_model.joblib'))
    return det, beh, sess, seq


def fit_session_aware(train_norm: pd.DataFrame):
    det = SessionAwareAnomalyDetector().fit(train_norm)
    os.makedirs(ART_V2, exist_ok=True)
    det.save(os.path.join(ART_V2, 'session_aware_detector.joblib'))
    return det


# ---------------------------------------------------------------------------
# Detection pipeline (DetectionEngine fusion is the final risk view)
# ---------------------------------------------------------------------------

def run_detection(df: pd.DataFrame, detector, behavioral=None, session=None,
                  sequence=None, profile='legacy', thresholds=None,
                  weights=None, rules_engine=None) -> pd.DataFrame:
    out = detector.predict(df.copy())
    out = (rules_engine or DetectionRulesEngine()).process_events(out)
    if behavioral is not None:
        out = behavioral.analyze_user_behavior(out)
    if session is not None:
        out = session.analyze(out)
    if sequence is not None:
        out = sequence.analyze(out)
    engine = DetectionEngine(channel_profile=profile, weights=weights,
                             thresholds=thresholds)
    return engine.process_dataframe(out)


def variant_configs(det19, beh19, det2, beh2, sess2, seq2, det_sa=None):
    return {
        'v1_9_baseline': dict(detector=det19, behavioral=beh19, profile='legacy'),
        'v2_base': dict(detector=det2, behavioral=beh2, profile='legacy'),
        'v2_session': dict(detector=det2, behavioral=beh2, session=sess2,
                           profile='session'),
        'v2_sequence': dict(detector=det2, behavioral=beh2, sequence=seq2,
                            profile='sequence'),
        'v2_complete': dict(detector=det2, behavioral=beh2, session=sess2,
                            sequence=seq2, profile='v2_0'),
        'v2_session_aware_complete': dict(detector=det_sa or det2,
                                          behavioral=beh2, session=sess2,
                                          sequence=seq2, profile='v2_0'),
    }


def calibrated_thresholds(df: pd.DataFrame, cfg: Dict) -> Dict[str, float]:
    scored = run_detection(df, **{k: v for k, v in cfg.items() if k != 'profile'},
                           profile=cfg['profile'], thresholds=None)
    scores = pd.to_numeric(scored['combined_score'], errors='coerce').dropna()
    if len(scores) == 0:
        return {'medium': 0.30, 'high': 0.55, 'critical': 0.75}
    p30, p65, p85 = np.percentile(scores, [30, 65, 85])
    return {'medium': float(p30), 'high': float(p65), 'critical': float(p85)}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _clf_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = int((y_true & y_pred).sum()); fp = int((~y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum()); fn = int((y_true & ~y_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4), 'fpr': round(fpr, 4),
            'specificity': round(spec, 4)}


def compute_metrics(df: pd.DataFrame, score_col: str = 'combined_score') -> Dict:
    if len(df) == 0:
        return {'n': 0}
    y_true = df['label'].isin(['suspicious', 'high_risk'])
    y_pred = df['risk_level'].isin(['High', 'Critical'])
    out = {'n': int(len(df)), 'positives': int(y_true.sum()),
           **_clf_metrics(y_true, y_pred)}
    score = pd.to_numeric(df.get(score_col, pd.Series(np.nan, index=df.index)),
                          errors='coerce')
    mask = score.notna() & y_true.notna()
    if mask.sum() and y_true[mask].nunique() > 1:
        out['roc_auc'] = round(float(roc_auc_score(y_true[mask], score[mask])), 4)
        out['pr_auc'] = round(float(average_precision_score(y_true[mask], score[mask])), 4)
        out['n_scored'] = int(mask.sum())
    else:
        out['roc_auc'] = None; out['pr_auc'] = None; out['n_scored'] = int(mask.sum())
    if 'anomaly_score_normalized' in df.columns:
        ml = pd.to_numeric(df['anomaly_score_normalized'], errors='coerce')
        mm = ml.notna() & y_true.notna()
        if mm.sum() and y_true[mm].nunique() > 1:
            out['ml_roc_auc'] = round(float(roc_auc_score(y_true[mm], ml[mm])), 4)
            out['ml_pr_auc'] = round(float(average_precision_score(y_true[mm], ml[mm])), 4)
    out['operating_modes'] = df['operating_mode'].value_counts().to_dict() \
        if 'operating_mode' in df.columns else {}
    out['missing_channel_rate'] = round(float(
        df['missing_channels'].apply(len).gt(0).mean()), 4) \
        if 'missing_channels' in df.columns else None
    return out


def scenario_masks(df: pd.DataFrame, manifest: Dict) -> Dict[str, pd.Series]:
    phases = manifest.get('behavior_phases', {})
    cohorts = manifest.get('entity_cohorts', {})
    holdout_only = set(cohorts.get('holdout_only_users', []))
    shared_ips = set(cohorts.get('shared_ips', []))
    masks = {
        'all': pd.Series(True, index=df.index),
        'unseen_users': df['user_id'].isin(holdout_only),
        'suspicious_label': df['label'].isin(['suspicious', 'high_risk']),
        'normal_label': df['label'] == 'normal',
        'common_ip_all': df['source_ip'].isin(shared_ips),
        'common_ip_normal': (df['label'] == 'normal') &
                            df['source_ip'].isin(shared_ips),
        'common_ip_positives': df['label'].isin(['suspicious', 'high_risk']) &
                               df['source_ip'].isin(shared_ips),
    }
    for phase in ['stable_normal', 'normal_to_suspicious',
                  'suspicious_to_normal', 'mixed']:
        users = {u for u, p in phases.items() if p == phase}
        masks[f'phase_{phase}'] = df['user_id'].isin(users)
    return masks


def evaluate_variants(df: pd.DataFrame, cal_df: Optional[pd.DataFrame],
                      configs: Dict[str, Dict], calibrated: bool = False,
                      weights_override: Optional[Dict[str, Dict]] = None,
                      progress: bool = True) -> Dict[str, pd.DataFrame]:
    results = {}
    for name, cfg in configs.items():
        t0 = time.perf_counter()
        thresholds = None
        weights = None
        if weights_override and name in weights_override:
            weights = weights_override[name]
        if calibrated and cal_df is not None:
            thresholds = calibrated_thresholds(_normals(cal_df), cfg)
        scored = run_detection(df, thresholds=thresholds, weights=weights, **cfg)
        results[name] = scored
        if progress:
            print(f'  {name}: {time.perf_counter()-t0:.1f}s', flush=True)
    return results


def summarize(results: Dict[str, pd.DataFrame], df: pd.DataFrame,
              manifest: Dict) -> Dict:
    masks = scenario_masks(df, manifest)
    out = {}
    for name, scored in results.items():
        out[name] = {'overall': compute_metrics(scored), 'scenarios': {}}
        for scen, mask in masks.items():
            sub = scored.loc[mask]
            if len(sub):
                out[name]['scenarios'][scen] = compute_metrics(sub)
    return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def _metric_value(y_true, y_pred, score, metric):
    y_true = np.asarray(y_true, dtype=bool); y_pred = np.asarray(y_pred, dtype=bool)
    if metric in ('precision', 'recall', 'f1', 'fpr', 'specificity'):
        m = _clf_metrics(y_true, y_pred)
        return m[metric]
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    if ok.sum() < 10 or len(np.unique(y_true[ok])) < 2:
        return np.nan
    try:
        if metric == 'roc_auc':
            return float(roc_auc_score(y_true[ok], score[ok]))
        if metric == 'pr_auc':
            return float(average_precision_score(y_true[ok], score[ok]))
    except Exception:
        return np.nan
    return np.nan


def bootstrap_compare(df_a: pd.DataFrame, df_b: pd.DataFrame,
                      metrics: List[str], n_event: int = 400,
                      n_user: int = 200, seed: int = SEED) -> Dict:
    """Paired bootstrap for detection metrics; df_a/df_b are same rows."""
    y_true = df_a['label'].isin(['suspicious', 'high_risk']).to_numpy()
    pred_a = df_a['risk_level'].isin(['High', 'Critical']).to_numpy()
    pred_b = df_b['risk_level'].isin(['High', 'Critical']).to_numpy()
    score_a = pd.to_numeric(df_a['combined_score'], errors='coerce').to_numpy(dtype=float)
    score_b = pd.to_numeric(df_b['combined_score'], errors='coerce').to_numpy(dtype=float)
    n = len(df_a)
    rng = np.random.default_rng(seed)
    users = df_a['user_id'].astype(str).to_numpy()
    user_indices = {u: np.flatnonzero(users == u) for u in np.unique(users)}
    unique_users = np.array(sorted(user_indices))

    def _eval(idx, metric):
        return (_metric_value(y_true[idx], pred_a[idx], score_a[idx], metric),
                _metric_value(y_true[idx], pred_b[idx], score_b[idx], metric))

    out = {}
    for mode, n_boot in [('event', n_event), ('user_clustered', n_user)]:
        for metric in metrics:
            diffs = []
            for _ in range(n_boot):
                if mode == 'event':
                    idx = rng.integers(0, n, n)
                else:
                    sampled = rng.choice(unique_users, size=len(unique_users),
                                         replace=True)
                    idx = np.concatenate([user_indices[u] for u in sampled]) \
                        if len(sampled) else np.array([], dtype=int)
                va, vb = _eval(idx, metric)
                if np.isfinite(va) and np.isfinite(vb):
                    diffs.append(vb - va)
            diffs = np.asarray(diffs, dtype=float)
            if len(diffs) == 0:
                out[f'{mode}:{metric}'] = {'n_boot_valid': 0}
                continue
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out[f'{mode}:{metric}'] = {
                'n_boot_valid': int(len(diffs)),
                'mean_diff': round(float(diffs.mean()), 4),
                'ci95': [round(float(lo), 4), round(float(hi), 4)],
                'excludes_zero': bool(lo > 0 or hi < 0),
            }
    return out


# ---------------------------------------------------------------------------
# Calibration/fusion audit
# ---------------------------------------------------------------------------

def calibration_audit(det2, beh2, sess2, seq2) -> Dict:
    report = {'artifact_path': None, 'checks': {}, 'passed': True}
    paths = [p for p in os.listdir(os.path.join(ROOT, 'models', 'saved_models', 'calibration'))
             if p.startswith('v2_0_thresholds_')]
    if not paths:
        report['checks']['artifact_exists'] = {'passed': False}
        report['passed'] = False
        return report
    path = os.path.join(ROOT, 'models', 'saved_models', 'calibration', sorted(paths)[-1])
    report['artifact_path'] = path
    fps = {
        'detector': detector_fingerprint(det2),
        'feature_baseline': component_fingerprint(det2.baseline_manager, 'feature_baseline'),
        'behavioral_baseline': component_fingerprint(beh2.baseline_manager, 'behavioral_baseline'),
        'session_baseline': component_fingerprint(sess2.baseline_manager, 'session_baseline'),
        'sequence_model': component_fingerprint(seq2, 'sequence_model'),
    }
    try:
        art = load_calibration(path, fingerprints=fps)
    except CalibrationUnavailable as exc:
        report['checks']['calibration_unavailable'] = str(exc)
        report['passed'] = False
        return report
    art_fps = art.get('fingerprints') or {}
    mismatches = {}
    for name, fp in fps.items():
        if fp is None:
            continue
        mismatches[name] = {'expected': fp, 'actual': art_fps.get(name),
                            'match': art_fps.get(name) == fp}
        if art_fps.get(name) != fp:
            report['passed'] = False
    report['checks'] = {
        'artifact_version': art.get('artifact_version'),
        'dataset_role': art.get('dataset_role'),
        'channel_profile': art.get('channel_profile'),
        'has_detection_thresholds': bool(art.get('detection_thresholds')),
        'detection_thresholds': art.get('detection_thresholds'),
        'fingerprint_comparisons': mismatches,
        'feature_baseline_verified': fps.get('feature_baseline') is not None,
    }
    return report


def fusion_weight_audit(validation_scored: Dict[str, pd.DataFrame],
                        holdout_best: Optional[pd.DataFrame],
                        holdout_complete: pd.DataFrame) -> Dict:
    from core.detection_engine import CHANNEL_PROFILES
    weight_sums = {k: round(sum(v.values()), 6) for k, v in CHANNEL_PROFILES.items()}
    val_summary = {name: compute_metrics(df) for name, df in validation_scored.items()}
    best = max(val_summary.items(), key=lambda kv: (kv[1].get('f1') or 0,
                                                    kv[1].get('roc_auc') or 0))
    out = {
        'channel_profile_weight_sums': weight_sums,
        'validation_candidates': val_summary,
        'validation_selected': best[0],
        'validation_selected_metrics': best[1],
        'note': 'Weight candidates were compared on validation.csv only; holdout was not used for selection.',
    }
    if holdout_best is not None:
        out['holdout_posthoc_selected'] = compute_metrics(holdout_best)
    out['holdout_v2_complete_default'] = compute_metrics(holdout_complete)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    train2 = _load(DATA_V2, 'train'); cal2 = _load(DATA_V2, 'calibration')
    val2 = _load(DATA_V2, 'validation'); hold2 = _load(DATA_V2, 'holdout')
    hold19 = _load(DATA_V19, 'holdout')
    man2 = _manifest(DATA_V2); man19 = _manifest(DATA_V19)

    print('Loading artifacts...')
    det19, beh19 = load_v19()
    det2, beh2, sess2, seq2 = load_v2()
    train_norm2 = _normals(train2)
    det_sa = fit_session_aware(train_norm2)

    configs = variant_configs(det19, beh19, det2, beh2, sess2, seq2, det_sa)

    print('Evaluating default-threshold ablation on v2 holdout...')
    res_default = evaluate_variants(hold2, cal2, configs, calibrated=False)
    summary_default = summarize(res_default, hold2, man2)

    print('Evaluating calibrated-threshold sensitivity...')
    res_cal = evaluate_variants(hold2, cal2, configs, calibrated=True)
    summary_cal = summarize(res_cal, hold2, man2)

    print('Reproducing v1.9 native holdout...')
    res19_native = evaluate_variants(hold19, _load(DATA_V19, 'calibration'),
                                     {'v1_9_baseline': configs['v1_9_baseline']},
                                     calibrated=False)
    summary19_native = summarize(res19_native, hold19, man19)

    print('Fusion-weight validation comparison...')
    candidates = {
        'v2_current': None,
        'equal_all': {'ml': 0.2, 'rules': 0.2, 'behavioral': 0.2,
                      'session': 0.2, 'sequence': 0.2},
        'ml_rules_heavy': {'ml': 0.4, 'rules': 0.3, 'behavioral': 0.1,
                           'session': 0.1, 'sequence': 0.1},
        'session_sequence_heavy': {'ml': 0.15, 'rules': 0.2, 'behavioral': 0.05,
                                   'session': 0.3, 'sequence': 0.3},
        'no_behavioral': {'ml': 0.3, 'rules': 0.3, 'behavioral': 0.0,
                          'session': 0.2, 'sequence': 0.2},
    }
    val_results = {}
    for name, weights in candidates.items():
        cfg = dict(configs['v2_complete'])
        val_results[name] = run_detection(
            val2, thresholds=None, weights=weights, **cfg)
    best_name = max({n: compute_metrics(d) for n, d in val_results.items()}.items(),
                    key=lambda kv: (kv[1].get('f1') or 0, kv[1].get('roc_auc') or 0))[0]
    best_weights = candidates[best_name]
    hold_best = run_detection(hold2, weights=best_weights, **configs['v2_complete']) \
        if best_name != 'v2_current' else res_default['v2_complete']
    fusion = fusion_weight_audit(val_results, hold_best, res_default['v2_complete'])

    print('Bootstrap comparisons...')
    comparisons = {
        'v1_9_vs_v2_complete': ('v1_9_baseline', 'v2_complete'),
        'v1_9_vs_v2_sequence': ('v1_9_baseline', 'v2_sequence'),
        'v2_base_vs_v2_complete': ('v2_base', 'v2_complete'),
        'v2_session_vs_v2_complete': ('v2_session', 'v2_complete'),
        'v2_sequence_vs_v2_complete': ('v2_sequence', 'v2_complete'),
    }
    boot = {}
    for cname, (a, b) in comparisons.items():
        boot[cname] = bootstrap_compare(res_default[a], res_default[b],
                                        ['precision', 'recall', 'f1', 'fpr',
                                         'roc_auc', 'pr_auc'])
        print(f'  bootstrap {cname} done', flush=True)

    print('Calibration audit...')
    cal_audit = calibration_audit(det2, beh2, sess2, seq2)

    report = {
        'milestone': '2.0.1',
        'seed': SEED,
        'protocol': {
            'positive_class': 'suspicious+high_risk',
            'alert_positive': 'DetectionEngine risk_level High/Critical',
            'ranking_score': 'combined_score',
            'note': 'RiskEngine risk_score is not used by DetectionEngine fusion; final risk_level is DetectionEngine output.',
        },
        'v2_holdout_default': summary_default,
        'v2_holdout_calibrated_thresholds': summary_cal,
        'v1_9_native_holdout': summary19_native,
        'fusion': fusion,
        'bootstrap': boot,
        'calibration': cal_audit,
    }
    out_path = os.path.join(OUT_DIR, 'milestone_2_0_1_audit.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'audit -> {out_path}')


if __name__ == '__main__':
    main()
