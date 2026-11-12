"""
Independent Dataset Generator for CAID — Milestone 1.9

Generates reproducible, label-independent synthetic security datasets with
entity-disjoint user/IP cohorts and realistic normal/suspicious overlap.

DESIGN GUARANTEES:
- Labels are assigned per-EVENT from behavioral parameters. They NEVER
  determine user IDs, IP addresses, IP prefixes, or row ordering.
- Users and IPs are persistent entities drawn from shared pools; the same
  user/IP can produce normal, suspicious and high-risk events.
- IP addresses are drawn from all three RFC 5737 TEST-NET ranges for EVERY
  label — no prefix encodes a label.
- Entity cohorts: 'shared' entities appear across all splits; 'holdout_only'
  entities appear exclusively in the holdout split (enables unseen-user/IP
  generalization scenarios).
- Behavioral distributions intentionally overlap (e.g. suspicious events can
  occur during business hours from common IPs; normal events can occur
  after hours).

OUTPUT (data/independent/):
- train.csv, calibration.csv, validation.csv, holdout.csv
- dataset_manifest.json — seeds, sizes, entity-overlap matrix, and
  label x {ip_prefix, hour_band, event_type} crosstabs used by the
  leakage audit and calibration role guard.

All data is synthetic. Documentation-only IP ranges (RFC 5737). No real
Canvas infrastructure is accessed or simulated.
"""

import json
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 1900
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'independent')

# Entity cohort sizes
N_SHARED_USERS = 140      # appear in train/calibration/validation/holdout
N_HOLDOUT_USERS = 60      # holdout only (unseen-user scenarios)
N_SHARED_IPS = 120        # used by all splits and all labels
N_HOLDOUT_IPS = 40        # holdout only (unseen-IP scenarios)

# Per-split event counts (approximate; exact counts in manifest)
SPLIT_TARGETS = {
    'train': 3000,
    'calibration': 800,    # predominantly normal (baseline) activity
    'validation': 800,
    'holdout': 1500,
}

# Label mixture per split. Calibration is baseline-dominated by design
# (threshold calibration convention); holdout carries the evaluation mass.
SPLIT_LABEL_MIX = {
    'train': {'normal': 0.72, 'suspicious': 0.18, 'high_risk': 0.10},
    'calibration': {'normal': 0.92, 'suspicious': 0.05, 'high_risk': 0.03},
    'validation': {'normal': 0.60, 'suspicious': 0.27, 'high_risk': 0.13},
    'holdout': {'normal': 0.55, 'suspicious': 0.30, 'high_risk': 0.15},
}

ROLES = ['student', 'instructor', 'ta', 'admin']
ROLE_WEIGHTS = [0.70, 0.15, 0.10, 0.05]

# Event types. The common pool is shared by ALL labels. Rare types appear
# only in the holdout split to exercise unseen event-type handling.
COMMON_EVENT_TYPES = [
    'login', 'course_view', 'resource_view',
    'assignment_submit', 'discussion_post', 'grade_view',
]
SUSPICIOUS_EVENT_TYPES = ['enrollment_view', 'bulk_export']
HIGH_RISK_EVENT_TYPES = [
    'grade_download', 'student_export', 'course_export', 'bulk_download',
]
HOLDOUT_ONLY_EVENT_TYPES = ['lti_launch', 'mastery_check', 'wiki_review']

# RFC 5737 documentation-only TEST-NET ranges. ALL labels draw from ALL
# ranges — prefix carries no label information.
IP_RANGE_24S = ['192.0.2.', '198.51.100.', '203.0.113.']

# Timestamp window shared by every split (no temporal split artifact)
BASE_DATE = pd.Timestamp('2024-01-15')
DATE_SPAN_DAYS = 30

LABELS = ('normal', 'suspicious', 'high_risk')


# ---------------------------------------------------------------------------
# Entity pools
# ---------------------------------------------------------------------------

def _make_users(rng, cohort, count, id_prefix):
    """Create persistent user profiles for one cohort."""
    users = []
    for i in range(count):
        role = rng.choice(ROLES, p=ROLE_WEIGHTS)
        # Per-user suspicious/high-risk propensity. Most users are mostly
        # normal; a minority generate frequent suspicious/high-risk events.
        u = rng.random()
        if u < 0.12:
            propensity = {'normal': 0.55, 'suspicious': 0.30, 'high_risk': 0.15}
        elif u < 0.30:
            propensity = {'normal': 0.80, 'suspicious': 0.15, 'high_risk': 0.05}
        else:
            propensity = {'normal': 0.97, 'suspicious': 0.025, 'high_risk': 0.005}
        users.append({
            'user_id': f'{id_prefix}_{cohort}_{i:04d}',
            'cohort': cohort,
            'role': role,
            'propensity': propensity,
            # Per-user activity scale: some users are simply busier.
            'activity_scale': float(rng.uniform(0.6, 1.8)),
        })
    return users


def _make_ips(rng, cohort, count, id_prefix, exclude=None):
    """Create IP addresses for one cohort across ALL TEST-NET ranges.

    `exclude` is a set of IP strings already assigned to another cohort so
    cohorts stay disjoint.
    """
    exclude = exclude or set()
    ips = []
    used = set(exclude)
    while len(ips) < count:
        prefix = IP_RANGE_24S[rng.integers(0, len(IP_RANGE_24S))]
        ip = f'{prefix}{int(rng.integers(1, 255))}'
        if ip not in used:
            used.add(ip)
            ips.append({'ip': ip, 'cohort': cohort})
    return ips


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------

def _hour_for_label(rng, label):
    """Sample an hour with heavy overlap between labels."""
    r = rng.random()
    if label == 'normal':
        # Mostly business hours, ~12% outside 06:00-22:00
        if r < 0.12:
            return int(rng.integers(0, 24))
        return int(rng.integers(6, 23))
    if label == 'suspicious':
        # Broadly distributed; ~35% outside business hours
        if r < 0.35:
            return int(rng.integers(0, 24))
        return int(rng.integers(6, 23))
    # high_risk: any hour, biased toward night
    if r < 0.55:
        return int(rng.integers(0, 6))
    return int(rng.integers(6, 24))


def _numeric_for_label(rng, label, col):
    """Sample a numeric feature; ranges deliberately overlap across labels."""
    if label == 'normal':
        ranges = {
            'api_requests': (10, 120),
            'data_mb': (0.1, 6.0),
            'courses_accessed': (1, 6),
            'accounts_accessed': (1, 4),
            'token_age_minutes': (5, 150),
        }
    elif label == 'suspicious':
        ranges = {
            'api_requests': (60, 520),
            'data_mb': (3.0, 22.0),
            'courses_accessed': (3, 16),
            'accounts_accessed': (2, 11),
            'token_age_minutes': (60, 540),
        }
    else:  # high_risk
        ranges = {
            'api_requests': (280, 2000),
            'data_mb': (14.0, 100.0),
            'courses_accessed': (9, 50),
            'accounts_accessed': (7, 100),
            'token_age_minutes': (180, 2880),
        }
    lo, hi = ranges[col]
    if col == 'data_mb':
        return round(float(rng.uniform(lo, hi)), 2)
    return int(rng.integers(lo, hi + 1))


def _event_type_for_label(rng, label, split):
    """Sample an event type. Common types shared by all labels; sensitive
    types concentrated in high_risk but not exclusive to it."""
    r = rng.random()
    # Holdout-only rare types appear on every label in the holdout split so
    # 'new event-type combination' scenarios contain both classes. Uses an
    # independent draw so it does not consume the label-mixture draw.
    if split == 'holdout' and rng.random() < 0.04:
        return str(rng.choice(HOLDOUT_ONLY_EVENT_TYPES))
    if label == 'high_risk':
        if r < 0.55:
            return str(rng.choice(HIGH_RISK_EVENT_TYPES))
        if r < 0.70:
            return str(rng.choice(SUSPICIOUS_EVENT_TYPES))
        return str(rng.choice(COMMON_EVENT_TYPES))
    if label == 'suspicious':
        if r < 0.15:
            return str(rng.choice(SUSPICIOUS_EVENT_TYPES))
        if r < 0.25:
            return str(rng.choice(HIGH_RISK_EVENT_TYPES))  # overlap
        return str(rng.choice(COMMON_EVENT_TYPES))
    # normal: overwhelmingly common types; rare overlap incl. a small
    # chance of sensitive event types (e.g. instructors exporting grades)
    if r < 0.02:
        return str(rng.choice(HIGH_RISK_EVENT_TYPES))
    if r < 0.05:
        return str(rng.choice(SUSPICIOUS_EVENT_TYPES))  # occasional overlap
    return str(rng.choice(COMMON_EVENT_TYPES))


def _make_event(rng, user, ip, label, split):
    """Generate one event row for a user/IP/label triple."""
    day_offset = int(rng.integers(0, DATE_SPAN_DAYS))
    hour = _hour_for_label(rng, label)
    minute = int(rng.integers(0, 60))
    second = int(rng.integers(0, 60))
    ts = (BASE_DATE + pd.Timedelta(days=day_offset)
          + pd.Timedelta(hours=hour, minutes=minute, seconds=second))

    after_hours = int(hour < 6 or hour > 22)

    row = {
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'user_id': user['user_id'],
        'role': user['role'],
        'source_ip': ip['ip'],
        'event_type': _event_type_for_label(rng, label, split),
        'after_hours': after_hours,
        'label': label,
    }
    for col in ('api_requests', 'data_mb', 'courses_accessed',
                'accounts_accessed', 'token_age_minutes'):
        row[col] = _numeric_for_label(rng, label, col)
    return row


def _choose_label(rng, user, split):
    """Draw the event label from the user's propensity blended with the
    split-level label mix (keeps split marginals close to targets)."""
    mix = SPLIT_LABEL_MIX[split]
    # 50/50 blend of user propensity and split mix -> overlap + marginals
    p = {lab: 0.5 * user['propensity'][lab] + 0.5 * mix[lab] for lab in LABELS}
    total = sum(p.values())
    r = rng.random() * total
    acc = 0.0
    for lab in LABELS:
        acc += p[lab]
        if r <= acc:
            return lab
    return 'normal'


def _generate_split(rng, split, users, ips):
    """Generate events for one split.

    Shared-cohort users/IPs are eligible in every split; holdout-only
    entities are eligible only in 'holdout'.
    """
    target = SPLIT_TARGETS[split]
    shared_users = [u for u in users if u['cohort'] == 'shared']
    holdout_users = [u for u in users if u['cohort'] == 'holdout_only']
    shared_ips = [i for i in ips if i['cohort'] == 'shared']
    holdout_ips = [i for i in ips if i['cohort'] == 'holdout_only']

    eligible_users = shared_users + (holdout_users if split == 'holdout' else [])
    eligible_ips = shared_ips + (holdout_ips if split == 'holdout' else [])

    rows = []
    # Round-robin users so every eligible user appears in every split they
    # belong to (persistent entities across splits).
    user_cycle = list(eligible_users)
    rng.shuffle(user_cycle)
    ui = 0
    while len(rows) < target:
        user = user_cycle[ui % len(user_cycle)]
        ui += 1
        # Busy users emit more events per turn (activity_scale).
        n_events = 1 + (1 if rng.random() < 0.3 * user['activity_scale'] else 0)
        for _ in range(n_events):
            if len(rows) >= target:
                break
            # IP selection: usually the user's "home" shared IP, sometimes
            # any pool IP — common IPs are used by all labels.
            if rng.random() < 0.6:
                ip = shared_ips[int(rng.integers(0, len(shared_ips)))]
            else:
                ip = eligible_ips[int(rng.integers(0, len(eligible_ips)))]
            label = _choose_label(rng, user, split)
            rows.append(_make_event(rng, user, ip, label, split))

    df = pd.DataFrame(rows)
    # Deterministic shuffle within the split (seeded); no label ordering
    df = df.sample(frac=1.0, random_state=SEED + hash(split) % 10000)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Manifest / audit statistics
# ---------------------------------------------------------------------------

def _crosstab(df, col_a, col_b):
    if col_a not in df.columns or col_b not in df.columns:
        return {}
    ct = pd.crosstab(df[col_a], df[col_b])
    return {str(i): {str(c): int(v) for c, v in row.items()}
            for i, row in ct.iterrows()}


def _ip_prefix(ip: str) -> str:
    parts = str(ip).split('.')
    return '.'.join(parts[:3]) + '.' if len(parts) == 4 else 'unknown'


def build_manifest(splits, users, ips):
    manifest = {
        'generator': 'data/generate_independent_datasets.py',
        'seed': SEED,
        'splits': {},
        'entity_pools': {
            'shared_users': N_SHARED_USERS,
            'holdout_only_users': N_HOLDOUT_USERS,
            'shared_ips': N_SHARED_IPS,
            'holdout_only_ips': N_HOLDOUT_IPS,
        },
        'overlap': {},
        'label_crosstabs': {},
        'quality': {},
    }

    user_to_cohort = {u['user_id']: u['cohort'] for u in users}
    ip_to_cohort = {i['ip']: i['cohort'] for i in ips}

    split_users, split_ips = {}, {}
    for name, df in splits.items():
        manifest['splits'][name] = {
            'file': f'{name}.csv',
            'role': name,
            'rows': int(len(df)),
            'label_counts': {k: int(v) for k, v in
                             df['label'].value_counts().items()},
            'unique_users': int(df['user_id'].nunique()),
            'unique_ips': int(df['source_ip'].nunique()),
        }
        split_users[name] = set(df['user_id'])
        split_ips[name] = set(df['source_ip'])

    # Entity overlap between train and each other split
    for name in ('calibration', 'validation', 'holdout'):
        manifest['overlap'][f'train_vs_{name}'] = {
            'shared_users': len(split_users['train'] & split_users[name]),
            'shared_ips': len(split_ips['train'] & split_ips[name]),
            'holdout_only_users_present': len(
                [u for u in split_users[name]
                 if user_to_cohort.get(u) == 'holdout_only']),
            'holdout_only_ips_present': len(
                [i for i in split_ips[name]
                 if ip_to_cohort.get(i) == 'holdout_only']),
        }
    manifest['overlap']['holdout_unseen_users'] = len(
        [u for u in split_users['holdout']
         if user_to_cohort.get(u) == 'holdout_only'])
    manifest['overlap']['holdout_unseen_ips'] = len(
        [i for i in split_ips['holdout']
         if ip_to_cohort.get(i) == 'holdout_only'])
    # Users/IPs shared between train and holdout (the "known" entities)
    manifest['overlap']['train_holdout_shared_users'] = len(
        split_users['train'] & split_users['holdout'])
    manifest['overlap']['train_holdout_shared_ips'] = len(
        split_ips['train'] & split_ips['holdout'])

    # Label independence evidence: crosstabs on the holdout split
    holdout = splits['holdout'].copy()
    holdout['ip_prefix'] = holdout['source_ip'].map(_ip_prefix)
    holdout['hour_band'] = pd.cut(
        pd.to_datetime(holdout['timestamp']).dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=['night_0-5', 'morning_6-11', 'afternoon_12-17',
                'evening_18-23'])
    manifest['label_crosstabs'] = {
        'label_x_ip_prefix': _crosstab(holdout, 'label', 'ip_prefix'),
        'label_x_hour_band': _crosstab(holdout, 'label', 'hour_band'),
        'label_x_event_type': _crosstab(holdout, 'label', 'event_type'),
    }

    # Quality checks
    all_df = pd.concat(splits.values(), ignore_index=True)
    manifest['quality'] = {
        'total_rows': int(len(all_df)),
        'exact_duplicate_rows': int(all_df.duplicated().sum()),
        'missing_required': int(all_df[
            ['timestamp', 'user_id', 'role', 'source_ip', 'event_type',
             'courses_accessed', 'accounts_accessed', 'api_requests',
             'data_mb', 'token_age_minutes', 'after_hours',
             'label']].isna().any(axis=1).sum()),
        'labels': sorted(all_df['label'].unique().tolist()),
    }
    return manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate():
    rng = np.random.default_rng(SEED)

    shared_users = _make_users(rng, 'shared', N_SHARED_USERS, 'usr')
    holdout_users = _make_users(rng, 'holdout_only', N_HOLDOUT_USERS, 'usr')
    users = shared_users + holdout_users

    shared_ips = _make_ips(rng, 'shared', N_SHARED_IPS, 'ip')
    holdout_ips = _make_ips(
        rng, 'holdout_only', N_HOLDOUT_IPS, 'ip',
        exclude={i['ip'] for i in shared_ips})
    ips = shared_ips + holdout_ips

    splits = {name: _generate_split(rng, name, users, ips)
              for name in SPLIT_TARGETS}

    os.makedirs(BASE_DIR, exist_ok=True)
    for name, df in splits.items():
        df.to_csv(os.path.join(BASE_DIR, f'{name}.csv'), index=False)

    # Entity cohorts sidecar: needed by scenario filtering in evaluation
    # (known vs unseen users/IPs). user_id/ip -> 'shared' | 'holdout_only'.
    cohorts = {
        'users': {u['user_id']: u['cohort'] for u in users},
        'ips': {i['ip']: i['cohort'] for i in ips},
    }
    with open(os.path.join(BASE_DIR, 'entity_cohorts.json'), 'w',
              encoding='utf-8') as f:
        json.dump(cohorts, f, indent=1)

    manifest = build_manifest(splits, users, ips)
    manifest_path = os.path.join(BASE_DIR, 'dataset_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated independent datasets in {BASE_DIR}")
    for name, df in splits.items():
        counts = df['label'].value_counts().to_dict()
        print(f"  {name:12s} rows={len(df):5d} users={df['user_id'].nunique():4d} "
              f"ips={df['source_ip'].nunique():4d} labels={counts}")
    print(f"  manifest -> {manifest_path}")
    print(f"  duplicates={manifest['quality']['exact_duplicate_rows']} "
          f"missing={manifest['quality']['missing_required']}")
    return splits, manifest


if __name__ == '__main__':
    generate()
