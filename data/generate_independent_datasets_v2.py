"""
Generate independent synthetic datasets for CAID Milestone 2.0.

Creates session-structured train/calibration/validation/holdout splits under
data/independent_v2/ WITHOUT modifying the v1.9 datasets.

Design guarantees:
- RFC 5737 documentation IP ranges only (synthetic, defensive research).
- Shared user/IP cohorts across train/calibration/validation plus
  holdout-only cohorts for unseen-entity evaluation.
- Optional session_id and auth_result fields; session_id is present on a
  subset so timeout reconstruction is also exercised.
- Labels are per-event and never encode user IDs, session IDs, or IP
  prefixes. behavior_phase is a generation-only concept stored in the
  manifest, not in the CSV.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

SEED = 20260220
BASE_DATE = datetime(2026, 1, 5)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'independent_v2')

SPLIT_TARGETS = {
    'train': 3000,
    'calibration': 800,
    'validation': 800,
    'holdout': 1500,
}
SPLIT_ORDER = {'train': 0, 'calibration': 1, 'validation': 2, 'holdout': 3}

IP_POOLS = {
    'shared': [f'192.0.2.{i}' for i in range(10, 60)] +
              [f'198.51.100.{i}' for i in range(10, 50)],
    'holdout_only': [f'203.0.113.{i}' for i in range(10, 45)],
}

EVENT_TYPES = [
    'login', 'course_view', 'discussion_post', 'assignment_submit',
    'file_view', 'quiz_submitted', 'grade_view', 'resource_view',
    'module_view', 'page_view', 'settings_view', 'content_view',
]
SENSITIVE_TYPES = ['grade_download', 'student_export', 'course_export',
                   'bulk_download', 'bulk_export']
AUTH_EVENTS = {'login'}

NORMAL_PROBS = {
    'stable_normal': [0.04, 0.05, 0.06, 0.07],
    'normal_to_suspicious': [0.05, 0.10, 0.30, 0.60],
    'suspicious_to_normal': [0.55, 0.35, 0.12, 0.04],
    'mixed': [0.12, 0.14, 0.16, 0.18],
}


def _user_cohorts(rng: np.random.Generator):
    shared = []
    phases = ['stable_normal', 'normal_to_suspicious', 'suspicious_to_normal', 'mixed']
    for i in range(48):
        shared.append({
            'user_id': f'user_{i:03d}',
            'role': 'student' if i % 5 else 'instructor',
            'behavior_phase': phases[i % len(phases)],
            'cohort': 'shared',
        })
    holdout_only = []
    for i in range(16):
        holdout_only.append({
            'user_id': f'user_h{ i:02d}',
            'role': 'student' if i % 4 else 'instructor',
            'behavior_phase': phases[(i + 1) % len(phases)],
            'cohort': 'holdout_only',
        })
    return shared, holdout_only


def _base_metrics(rng: np.random.Generator, event_type: str) -> Dict[str, float]:
    base = {
        'api_requests': rng.integers(20, 140),
        'data_mb': round(float(rng.uniform(0.5, 6.0)), 2),
        'courses_accessed': int(rng.integers(1, 5)),
        'accounts_accessed': int(rng.integers(1, 4)),
        'token_age_minutes': int(rng.integers(5, 240)),
    }
    if event_type in {'grade_download', 'student_export', 'course_export',
                      'bulk_download', 'bulk_export'}:
        base.update({'api_requests': int(rng.integers(120, 320)),
                     'data_mb': round(float(rng.uniform(8, 30)), 2),
                     'courses_accessed': int(rng.integers(4, 10)),
                     'accounts_accessed': int(rng.integers(3, 8))})
    elif event_type == 'login':
        base.update({'api_requests': int(rng.integers(10, 60)),
                     'data_mb': round(float(rng.uniform(0.1, 1.0)), 2)})
    return base


def _suspicious_metrics(rng: np.random.Generator, event_type: str,
                        subtle: bool = False) -> Dict[str, float]:
    m = _base_metrics(rng, event_type)
    if subtle:
        # Sequence-detectable but individually subtle events.
        m['api_requests'] = int(m['api_requests'] * rng.uniform(1.0, 1.6))
        m['data_mb'] = round(m['data_mb'] * rng.uniform(1.0, 1.8), 2)
        return m
    m['api_requests'] = int(min(m['api_requests'] * rng.uniform(2.0, 5.0), 1200))
    m['data_mb'] = round(min(m['data_mb'] * rng.uniform(2.0, 6.0), 120.0), 2)
    m['courses_accessed'] = int(min(m['courses_accessed'] * rng.integers(2, 5), 40))
    m['accounts_accessed'] = int(min(m['accounts_accessed'] * rng.integers(2, 6), 60))
    if event_type in SENSITIVE_TYPES or rng.random() < 0.35:
        m['data_mb'] = round(min(m['data_mb'] * rng.uniform(1.5, 3.0), 200.0), 2)
    return m


def _event_label(rng: np.random.Generator, suspicious: bool) -> str:
    if not suspicious:
        return 'normal'
    return 'high_risk' if rng.random() < 0.35 else 'suspicious'


def _pick_event_type(rng: np.random.Generator, suspicious_session: bool,
                     pattern: str) -> str:
    if suspicious_session:
        if pattern == 'auth_burst_export':
            return 'login' if rng.random() < 0.45 else \
                rng.choice(SENSITIVE_TYPES + ['course_view', 'file_view'])
        if pattern == 'broad_access_export':
            return rng.choice(SENSITIVE_TYPES + ['course_view', 'file_view',
                                                 'assignment_submit'])
        if pattern == 'sensitive_burst':
            return rng.choice(SENSITIVE_TYPES)
        if pattern == 'mixed_sequence':
            return rng.choice(EVENT_TYPES + SENSITIVE_TYPES)
    return str(rng.choice(EVENT_TYPES))


def _make_session_events(rng: np.random.Generator, user: Dict, split: str,
                         session_idx: int, start: datetime,
                         suspicious_prob: float) -> List[Dict]:
    suspicious_session = rng.random() < suspicious_prob
    pattern = str(rng.choice(['auth_burst_export', 'broad_access_export',
                              'sensitive_burst', 'mixed_sequence'])) \
        if suspicious_session else 'normal'
    n_events = int(rng.integers(2, 9)) if suspicious_session else int(rng.integers(1, 7))
    explicit_session = rng.random() < 0.70
    session_id = f'sess_{user["user_id"]}_{session_idx:04d}' if explicit_session else ''

    rows = []
    ts = start
    auth_failures_left = int(rng.integers(2, 4)) if pattern == 'auth_burst_export' else 0
    for i in range(n_events):
        event_type = _pick_event_type(rng, suspicious_session, pattern)
        subtle = suspicious_session and (
            event_type == 'login' or (pattern == 'mixed_sequence' and rng.random() < 0.5))
        metrics = (_suspicious_metrics(rng, event_type, subtle=subtle)
                   if suspicious_session else _base_metrics(rng, event_type))
        label = _event_label(rng, suspicious_session and rng.random() < 0.9)

        auth_result = ''
        if event_type in AUTH_EVENTS:
            if pattern == 'auth_burst_export' and auth_failures_left > 0 and i < 3:
                auth_result = 'failure'
                auth_failures_left -= 1
            else:
                auth_result = 'success' if rng.random() < 0.9 else 'failure'

        hour = ts.hour + ts.minute / 60.0
        rows.append({
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': user['user_id'],
            'role': user['role'],
            'source_ip': str(rng.choice(IP_POOLS['shared'] if user['cohort'] == 'shared'
                                        else IP_POOLS['holdout_only'])),
            'event_type': event_type,
            'courses_accessed': metrics['courses_accessed'],
            'accounts_accessed': metrics['accounts_accessed'],
            'api_requests': metrics['api_requests'],
            'data_mb': metrics['data_mb'],
            'token_age_minutes': metrics['token_age_minutes'],
            'after_hours': int(hour >= 22 or hour < 6),
            'label': label,
            'session_id': session_id,
            'auth_result': auth_result,
        })
        ts += timedelta(seconds=int(rng.integers(15, 240)))
    return rows


def _split_users(split: str, shared, holdout_only):
    if split == 'holdout':
        return shared + holdout_only
    return shared


def _generate_split(rng: np.random.Generator, split: str,
                    shared, holdout_only) -> List[Dict]:
    target = SPLIT_TARGETS[split]
    users = _split_users(split, shared, holdout_only)
    rows: List[Dict] = []
    order = SPLIT_ORDER[split]
    day_offset = {'train': 0, 'calibration': 7, 'validation': 14, 'holdout': 21}[split]
    base = BASE_DATE + timedelta(days=day_offset)

    # Round-robin sessions across users until target reached.
    cursor = {u['user_id']: 0 for u in users}
    while len(rows) < target:
        for user in users:
            if len(rows) >= target:
                break
            uid = user['user_id']
            session_idx = cursor[uid]
            cursor[uid] += 1
            # Spread sessions across the split's 7-day window deterministically.
            day = int(rng.integers(0, 7))
            hour = int(rng.integers(7, 22))
            start = base + timedelta(days=day, hours=hour,
                                     minutes=int(rng.integers(0, 60)))
            phase = user['behavior_phase']
            probs = NORMAL_PROBS[phase]
            suspicious_prob = float(probs[min(order, len(probs) - 1)])
            rows.extend(_make_session_events(rng, user, split, session_idx,
                                             start, suspicious_prob))
    return rows[:target]


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    shared, holdout_only = _user_cohorts(rng)

    manifest = {
        'milestone': '2.0',
        'seed': SEED,
        'base_date': BASE_DATE.isoformat(),
        'split_targets': SPLIT_TARGETS,
        'entity_cohorts': {
            'shared_users': [u['user_id'] for u in shared],
            'holdout_only_users': [u['user_id'] for u in holdout_only],
            'shared_ips': IP_POOLS['shared'],
            'holdout_only_ips': IP_POOLS['holdout_only'],
        },
        'behavior_phases': {u['user_id']: u['behavior_phase'] for u in shared + holdout_only},
        'optional_fields': ['session_id', 'auth_result'],
        'notes': [
            'Synthetic defensive research data only.',
            'Labels are per-event; they do not determine user/session/IP identifiers.',
            'session_id is explicit on a subset; remaining sessions exercise timeout reconstruction.',
            'auth_result is present on login events only.',
        ],
    }

    all_stats = {}
    for split in SPLIT_TARGETS:
        rows = _generate_split(rng, split, shared, holdout_only)
        df = pd.DataFrame(rows)
        # Deterministic column order.
        cols = ['timestamp', 'user_id', 'role', 'source_ip', 'event_type',
                'courses_accessed', 'accounts_accessed', 'api_requests',
                'data_mb', 'token_age_minutes', 'after_hours', 'label',
                'session_id', 'auth_result']
        df = df[cols]
        path = os.path.join(OUT_DIR, f'{split}.csv')
        df.to_csv(path, index=False)
        all_stats[split] = {
            'rows': int(len(df)),
            'users': int(df['user_id'].nunique()),
            'sessions_explicit': int((df['session_id'].fillna('') != '').sum()),
            'auth_result_present': int((df['auth_result'].fillna('') != '').sum()),
            'label_counts': df['label'].value_counts().to_dict(),
        }
        print(f'{split}: {len(df)} rows -> {path}')

    manifest['splits'] = all_stats
    manifest_path = os.path.join(OUT_DIR, 'dataset_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    cohorts_path = os.path.join(OUT_DIR, 'entity_cohorts.json')
    with open(cohorts_path, 'w', encoding='utf-8') as f:
        json.dump(manifest['entity_cohorts'], f, indent=2)
    print(f'manifest -> {manifest_path}')


if __name__ == '__main__':
    main()
