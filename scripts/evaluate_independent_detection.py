"""
Independent-detection evaluation — repeatable version of the Phase 4.3
holdout evaluation.

Runs the PRODUCTION batch pipeline (modules.session_intelligence.
analyze_session_intelligence, v2_0 profile, fingerprint-checked
calibration artifact) over a recovered independent dataset directory
(e.g. data/independent_v2 restored from git bfdca25) and reports
per-channel + combined metrics on the holdout split, plus entity-cohort
slices. The holdout is an evaluation set — this script performs no
tuning; threshold/weight selection must happen on validation splits
elsewhere and be documented.

Usage:
    .venv/Scripts/python scripts/evaluate_independent_detection.py \
        --data-dir <path-to-independent_v2> [--split holdout] \
        [--output reports/independent_detection_evaluation_v2.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from benchmark_models import _auc, _prf  # noqa: E402
from modules.session_intelligence import analyze_session_intelligence  # noqa: E402

SCORE_COLS = {
    'ml': 'anomaly_score_normalized',
    'rules': 'rule_score',
    'behavioral': 'behavioral_deviation_score',
    'session': 'session_deviation_score',
    'sequence': 'sequence_deviation_score',
}
POSITIVE_LABELS = {'suspicious', 'high_risk'}


def _binary_labels(s: pd.Series) -> np.ndarray:
    return s.isin(POSITIVE_LABELS).astype(int).to_numpy()


def _eval_score(y, score, pred=None):
    score = pd.to_numeric(pd.Series(score), errors='coerce').fillna(0.0)
    out = _auc(y, score.to_numpy())
    if pred is None:
        pred = (score >= 0.5).astype(int).to_numpy()
    out.update(_prf(y, np.asarray(pred)))
    return out


def _eval_combined(y, df_scored):
    score = pd.to_numeric(df_scored['combined_score'],
                          errors='coerce').fillna(0.0).to_numpy()
    pred = df_scored['risk_level'].isin(['High', 'Critical']).astype(int)
    out = _auc(y, score)
    out.update(_prf(y, pred.to_numpy()))
    return out


def _slice(df_scored, y, holdout_users):
    mask = df_scored['user_id'].astype(str).isin(holdout_users)
    sub = df_scored[mask]
    ys = y[mask.to_numpy()]
    res = {'n': int(len(sub))}
    for ch, col in SCORE_COLS.items():
        if col in sub.columns:
            res[ch] = _eval_score(ys, sub[col])
    res['combined_v2_0'] = _eval_combined(ys, sub)
    res['score_auc'] = {
        'roc_auc': res['combined_v2_0'].pop('roc_auc', None),
        'pr_auc': res['combined_v2_0'].pop('pr_auc', None),
    }
    return res


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', required=True,
                    help='directory containing the independent dataset '
                         'CSVs (train/calibration/validation/holdout) '
                         'and dataset_manifest.json / entity_cohorts.json')
    ap.add_argument('--split', default='holdout')
    ap.add_argument('--output', default=os.path.join(
        ROOT, 'reports', 'independent_detection_evaluation_v2.json'))
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    holdout_path = os.path.join(data_dir, f'{args.split}.csv')
    if not os.path.exists(holdout_path):
        raise SystemExit(f'holdout CSV not found: {holdout_path}')

    df = pd.read_csv(holdout_path)
    scored = analyze_session_intelligence(df)
    y = _binary_labels(df['label'])

    # entity cohorts for disjoint-user slice
    cohorts_path = os.path.join(data_dir, 'entity_cohorts.json')
    holdout_users = []
    if os.path.exists(cohorts_path):
        cohorts = json.load(open(cohorts_path, encoding='utf-8'))
        holdout_users = cohorts.get('holdout_only_users', [])

    # user overlap vs other splits (train/calibration/validation)
    other_users = set()
    for other in ('train', 'calibration', 'validation'):
        p = os.path.join(data_dir, f'{other}.csv')
        if os.path.exists(p):
            other_users |= set(pd.read_csv(
                p, usecols=['user_id'])['user_id'].astype(str))
    hold_users = sorted(set(df['user_id'].astype(str)) & other_users)

    channels = {}
    for ch, col in SCORE_COLS.items():
        if col in scored.columns:
            channels[ch] = _eval_score(y, scored[col])
    combined = _eval_combined(y, scored)
    combined['score_auc'] = {
        'roc_auc': combined.pop('roc_auc', None),
        'pr_auc': combined.pop('pr_auc', None),
    }
    channels['combined_v2_0'] = combined

    result = {
        'dataset': os.path.relpath(data_dir, ROOT),
        'split': args.split,
        'rows': int(len(df)),
        'user_overlap_with_train_validation': hold_users,
        'entity_disjoint_users': not hold_users,
        'label_counts': df['label'].value_counts().to_dict(),
        'calibration_statuses': {
            'risk': (scored['risk_calibration_status'].mode().iloc[0]
                     if 'risk_calibration_status' in scored.columns
                     and len(scored['risk_calibration_status'].mode())
                     else None),
            'detection': (scored['detection_calibration_status'].mode().iloc[0]
                          if 'detection_calibration_status' in scored.columns
                          and len(scored['detection_calibration_status'].mode())
                          else None),
            'threshold_version': (scored['threshold_version'].dropna().iloc[0]
                                  if 'threshold_version' in scored.columns
                                  and scored['threshold_version'].notna().any()
                                  else None),
        },
        'operating_modes': (scored['operating_mode'].value_counts().to_dict()
                            if 'operating_mode' in scored.columns else {}),
        'channels': channels,
    }
    if holdout_users:
        hu = set(holdout_users)
        result['channels']['slice_holdout_only_users'] = _slice(
            scored, y, hu)
        result['channels']['slice_shared_users'] = _slice(
            scored, y, set(df['user_id'].astype(str)) - hu)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps({k: result[k] for k in
                      ('dataset', 'split', 'rows', 'calibration_statuses')},
                     indent=2))
    print('combined_v2_0:', json.dumps(channels['combined_v2_0']))
    print(f'wrote {args.output}')
    return result


if __name__ == '__main__':
    main()
