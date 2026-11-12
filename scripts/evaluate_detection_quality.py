"""Milestone 2.7 — detection-quality evaluation harness.

Evaluates the production v2.0 detection stack on entity-disjoint holdout
users plus synthetic attack scenarios. Internal reproducibility only;
not independent validation.

Usage:
    .venv/Scripts/python.exe scripts/evaluate_detection_quality.py \
        --out reports/detection_quality.json
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

import benchmark_models as bm  # noqa: E402
from data.generate_scenarios import SCENARIOS, generate_scenario  # noqa: E402
from modules.session_intelligence import analyze_session_intelligence  # noqa: E402


def _binary_labels(labels: pd.Series) -> np.ndarray:
    return labels.isin(['suspicious', 'high_risk']).astype(int).to_numpy()


def _prf(y: np.ndarray, pred: np.ndarray) -> Dict:
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': round(prec, 4), 'recall': round(rec, 4),
            'f1': round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 4),
            'fpr': round(fpr, 4)}


def _score_metrics(df: pd.DataFrame) -> Dict:
    """Metrics for one analyzed frame that still carries ``label``."""
    if 'label' not in df.columns or 'combined_score' not in df.columns:
        return {'error': 'missing label or combined_score'}
    y = _binary_labels(df['label'])
    scores = pd.to_numeric(df['combined_score'], errors='coerce') \
        .fillna(0.0).to_numpy()
    high = df['risk_level'].isin(['High', 'Critical']).to_numpy() \
        if 'risk_level' in df.columns else np.zeros(len(df), dtype=bool)
    out = {
        'rows': int(len(df)),
        'labels': df['label'].value_counts().to_dict(),
        'default_risk_level': _prf(y, high.astype(int)),
        'score_auc': bm._auc(y, scores),
        'operating_modes': (df['operating_mode'].value_counts().to_dict()
                            if 'operating_mode' in df.columns else {}),
        'calibration_status': {
            'risk': (df['risk_calibration_status'].mode().iloc[0]
                     if 'risk_calibration_status' in df.columns
                     and len(df['risk_calibration_status'].mode()) else None),
            'detection': (df['detection_calibration_status'].mode().iloc[0]
                          if 'detection_calibration_status' in df.columns
                          and len(df['detection_calibration_status'].mode())
                          else None),
            'threshold_version': (df['threshold_version'].mode().iloc[0]
                                  if 'threshold_version' in df.columns
                                  and len(df['threshold_version'].mode())
                                  else None),
        },
        'score_by_label': {},
    }
    for lab in sorted(df['label'].unique()):
        vals = scores[df['label'].to_numpy() == lab]
        out['score_by_label'][str(lab)] = {
            'count': int(len(vals)),
            'mean': round(float(np.mean(vals)), 4) if len(vals) else None,
            'p95': round(float(np.percentile(vals, 95)), 4)
            if len(vals) else None,
        }
    return out


def _best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> Dict:
    """Select a combined-score threshold on validation users only."""
    candidates = np.unique(scores)
    if len(candidates) > 200:
        candidates = np.quantile(candidates, np.linspace(0, 1, 200))
    best = {'threshold': None, 'f1': -1.0, 'precision': 0.0,
            'recall': 0.0, 'fpr': 0.0}
    for t in candidates:
        m = _prf(y, (scores >= t).astype(int))
        if m['f1'] > best['f1']:
            best = {'threshold': round(float(t), 6), **m}
    return best


def main() -> Dict:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(
        ROOT, 'reports', 'detection_quality.json'))
    args = ap.parse_args()

    df = bm.load_labeled_data()
    splits = bm.make_splits(df)
    users = {k: set(v['user_id']) for k, v in splits.items()}
    for a in users:
        for b in users:
            if a != b:
                assert not (users[a] & users[b]), f'user leak {a}/{b}'

    # Validation-only threshold selection; holdout remains untouched.
    val = analyze_session_intelligence(splits['validation'].copy())
    val_y = _binary_labels(val['label'])
    val_scores = pd.to_numeric(val['combined_score'],
                               errors='coerce').fillna(0.0).to_numpy()
    selected = _best_f1_threshold(val_y, val_scores)

    holdout = analyze_session_intelligence(splits['holdout'].copy())
    hold_y = _binary_labels(holdout['label'])
    hold_scores = pd.to_numeric(holdout['combined_score'],
                                errors='coerce').fillna(0.0).to_numpy()
    threshold_metrics = _prf(hold_y, (hold_scores >= selected['threshold']).astype(int)) \
        if selected['threshold'] is not None else {'error': 'no_threshold'}

    unseen_user = {
        'validation_rows': int(len(splits['validation'])),
        'holdout_rows': int(len(splits['holdout'])),
        'validation_users': int(splits['validation']['user_id'].nunique()),
        'holdout_users': int(splits['holdout']['user_id'].nunique()),
        'entity_disjoint': True,
        'validation_threshold': selected,
        'holdout_default_risk_level': _score_metrics(holdout)['default_risk_level'],
        'holdout_score_auc': _score_metrics(holdout)['score_auc'],
        'holdout_threshold_applied': threshold_metrics,
        'holdout_score_by_label': _score_metrics(holdout)['score_by_label'],
        'holdout_calibration_status': _score_metrics(holdout)['calibration_status'],
    }

    scenarios = {}
    for name in SCENARIOS:
        sdf = generate_scenario(name)
        out = analyze_session_intelligence(sdf)
        scenarios[name] = _score_metrics(out)

    result = {
        'benchmark': 'milestone_2_7_detection_quality',
        'validation_scope': 'internal_reproducibility',
        'independent_validation': 'not_performed_current_data_only',
        'unseen_user_evaluation': unseen_user,
        'synthetic_scenarios': scenarios,
        'false_positive_notes': [
            'Default operating point uses calibrated High/Critical risk levels.',
            'Validation-selected threshold is reported separately and was not tuned on holdout.',
            'Scenario baseline FP rate is the control for synthetic FP reduction checks.',
        ],
        'notes': [
            'Scores are calibrated anomaly scores, not probabilities.',
            'Holdout users are disjoint from all training/calibration users.',
            'Synthetic scenarios are defensive research data only.',
        ],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps({
        'holdout_rows': unseen_user['holdout_rows'],
        'holdout_auc': unseen_user['holdout_score_auc'],
        'holdout_default': unseen_user['holdout_default_risk_level'],
        'threshold_applied': unseen_user['holdout_threshold_applied'],
        'scenarios': {k: (v.get('default_risk_level') or {}).get('fpr')
                      for k, v in scenarios.items()},
        'out': args.out,
    }, indent=2, default=str))
    return result


if __name__ == '__main__':
    main()
