"""
Leakage audit for CAID Milestone 2.0 independent_v2 datasets.

Checks split hygiene, identifier/label independence, optional-field coverage,
and timestamp ordering. Writes data/independent_v2/leakage_audit.json.
"""

import json
import os
from typing import Dict

import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                    'data', 'independent_v2')
SPLITS = ['train', 'calibration', 'validation', 'holdout']


def _load(split: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE, f'{split}.csv'))


def _prefix(ip) -> str:
    parts = str(ip).split('.')
    return '.'.join(parts[:3]) if len(parts) >= 3 else str(ip)


def audit() -> Dict:
    report = {'splits': {}, 'checks': {}, 'passed': True}
    frames = {s: _load(s) for s in SPLITS}
    manifest_path = os.path.join(BASE, 'dataset_manifest.json')
    manifest = json.load(open(manifest_path, encoding='utf-8')) if os.path.exists(manifest_path) else {}

    # Split hygiene: holdout-only users must not appear in earlier splits.
    train_users = set(frames['train']['user_id'])
    holdout_users = set(frames['holdout']['user_id'])
    manifest_holdout_only = set(manifest.get('entity_cohorts', {}).get('holdout_only_users', []))
    leaked_users = (train_users & manifest_holdout_only)
    report['checks']['holdout_only_users_absent_from_train'] = {
        'passed': len(leaked_users) == 0,
        'leaked_users': sorted(leaked_users)[:10],
    }
    report['passed'] = report['passed'] and not leaked_users

    # Label vs identifier independence (Cramer's V-ish max abs diff).
    for split, df in frames.items():
        df = df.copy()
        df['ip_prefix'] = df['source_ip'].map(_prefix)
        df['session_prefix'] = df['session_id'].fillna('').astype(str).str.extract(r'^(sess_[^_]+)')[0].fillna('')
        label = df['label'].astype(str)
        checks = {}
        for col in ['ip_prefix', 'session_prefix', 'user_id']:
            ct = pd.crosstab(df[col], label)
            if ct.shape[0] <= 1 or ct.shape[1] <= 1:
                checks[col] = {'max_abs_diff': 0.0, 'note': 'single-level'}
                continue
            # normalized max deviation from uniform label share per identifier
            shares = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0)
            overall = label.value_counts(normalize=True)
            max_diff = float((shares.sub(overall, axis=1).abs().max(axis=1)).max())
            checks[col] = {'max_abs_diff': round(max_diff, 4)}
        report['splits'][split] = {
            'rows': int(len(df)),
            'label_counts': df['label'].value_counts().to_dict(),
            'session_id_coverage': round(float((df['session_id'].fillna('') != '').mean()), 4),
            'auth_result_coverage': round(float((df['auth_result'].fillna('') != '').mean()), 4),
            'identifier_label_checks': checks,
        }

    # Timestamp monotonicity within explicit sessions (where present).
    for split, df in frames.items():
        df = df.copy()
        df['ts'] = pd.to_datetime(df['timestamp'], errors='coerce')
        explicit = df[df['session_id'].fillna('') != ''].copy()
        bad = 0
        for _, grp in explicit.groupby(['user_id', 'session_id']):
            if not grp['ts'].is_monotonic_increasing:
                bad += 1
        report['splits'][split]['non_monotonic_explicit_sessions'] = int(bad)
        if bad:
            report['passed'] = False

    # Required/optional schema.
    required = {'timestamp','user_id','role','source_ip','event_type','courses_accessed',
                'accounts_accessed','api_requests','data_mb','token_age_minutes','after_hours','label'}
    for split, df in frames.items():
        missing = sorted(required - set(df.columns))
        report['splits'][split]['missing_required'] = missing
        if missing:
            report['passed'] = False

    report['passed'] = all(c.get('passed', True) for c in report['checks'].values()) and report['passed']
    out = os.path.join(BASE, 'leakage_audit.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)
    print(json.dumps({'passed': report['passed'], 'out': out}, indent=2))
    return report


if __name__ == '__main__':
    audit()
