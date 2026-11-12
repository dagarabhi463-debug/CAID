"""
Scenario-based synthetic security-log generator for CAID Milestone 2.3.

Produces deliberately-constructed security scenarios (normal, suspicious,
and high-risk users; multiple sessions; shared IPs; repeated sensitive
operations; suspicious sequences) in the raw 14-column log schema so the
full pipeline (anomaly detection -> session analysis -> correlation ->
incidents) can be demonstrated end-to-end.

Design guarantees:
- RFC 5737 documentation IP ranges only (192.0.2.0/24, 198.51.100.0/24,
  203.0.113.0/24). No real Canvas connectivity.
- Deterministic: every scenario accepts a seed and produces identical
  output for identical input.
- Explicit session_id on every event; auth_result on login events.
- Scenarios are scripts (structured user/session narratives), not random
  noise — each is designed to exercise specific correlation types.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'scenarios')

COLUMNS = ['timestamp', 'user_id', 'role', 'source_ip', 'event_type',
           'courses_accessed', 'accounts_accessed', 'api_requests',
           'data_mb', 'token_age_minutes', 'after_hours', 'label',
           'session_id', 'auth_result']

AUTH_EVENTS = {'login'}
NORMAL_EVENT_TYPES = [
    'course_view', 'discussion_post', 'assignment_submit', 'file_view',
    'quiz_submitted', 'grade_view', 'resource_view', 'module_view',
    'page_view', 'settings_view', 'content_view',
]
SENSITIVE_EVENT_TYPES = [
    'grade_download', 'student_export', 'course_export',
    'bulk_download', 'bulk_export',
]

BASE_DATE = datetime(2026, 2, 2)  # Monday


# ----------------------------------------------------------------------
# Row builder helpers
# ----------------------------------------------------------------------

class _Session:
    """Accumulates event rows for one user session."""

    _TIER = {
        'normal': {
            'api_requests': (10, 100), 'data_mb': (0.1, 5.0),
            'courses_accessed': (1, 5), 'accounts_accessed': (1, 3),
            'token_age_minutes': (5, 120), 'after_hours': 0,
            'label': 'normal',
        },
        'suspicious': {
            'api_requests': (150, 500), 'data_mb': (5.1, 30.0),
            'courses_accessed': (6, 15), 'accounts_accessed': (4, 10),
            'token_age_minutes': (121, 480), 'label': 'suspicious',
        },
        'high_risk': {
            'api_requests': (500, 1500), 'data_mb': (30.1, 100.0),
            'courses_accessed': (16, 40), 'accounts_accessed': (11, 60),
            'token_age_minutes': (481, 2000), 'label': 'high_risk',
        },
    }

    def __init__(self, rng: np.random.Generator, user_id: str, role: str,
                 ip: str, session_id: str, start: datetime):
        self.rng = rng
        self.user_id = user_id
        self.role = role
        self.ip = ip
        self.session_id = session_id
        self.ts = start
        self.rows: List[Dict] = []

    def _metrics(self, tier: str, event_type: str) -> Dict:
        spec = self._TIER[tier]
        m = {
            'api_requests': int(self.rng.integers(*spec['api_requests'])),
            'data_mb': round(float(self.rng.uniform(*spec['data_mb'])), 2),
            'courses_accessed': int(self.rng.integers(*spec['courses_accessed'])),
            'accounts_accessed': int(self.rng.integers(*spec['accounts_accessed'])),
            'token_age_minutes': int(self.rng.integers(*spec['token_age_minutes'])),
            'after_hours': spec.get('after_hours', 0),
            'label': spec['label'],
        }
        if event_type in SENSITIVE_EVENT_TYPES and tier != 'normal':
            m['api_requests'] = int(min(m['api_requests'] * 1.5, 1800))
            m['data_mb'] = round(min(m['data_mb'] * 1.8, 150.0), 2)
        return m

    def add(self, event_type: str, tier: str = 'normal',
            minutes_later: Optional[int] = None,
            after_hours: Optional[int] = None) -> Dict:
        m = self._metrics(tier, event_type)
        if after_hours is not None:
            m['after_hours'] = after_hours
        auth_result = ''
        if event_type in AUTH_EVENTS:
            auth_result = 'success' if self.rng.random() < 0.9 else 'failure'
        row = {
            'timestamp': self.ts.strftime('%Y-%m-%d %H:%M:%S'),
            'user_id': self.user_id,
            'role': self.role,
            'source_ip': self.ip,
            'event_type': event_type,
            'courses_accessed': m['courses_accessed'],
            'accounts_accessed': m['accounts_accessed'],
            'api_requests': m['api_requests'],
            'data_mb': m['data_mb'],
            'token_age_minutes': m['token_age_minutes'],
            'after_hours': m['after_hours'],
            'label': m['label'],
            'session_id': self.session_id,
            'auth_result': auth_result,
        }
        self.rows.append(row)
        self.ts += timedelta(
            minutes=int(minutes_later if minutes_later is not None
                        else self.rng.integers(1, 12)),
            seconds=int(self.rng.integers(0, 59)))
        return row

    def add_auth_failure(self, event_type: str = 'login',
                         tier: str = 'suspicious') -> Dict:
        row = self.add(event_type, tier=tier)
        row['auth_result'] = 'failure'
        return row


def _normal_user_sessions(rng: np.random.Generator, user_id: str,
                          role: str, ip: str, n_sessions: int,
                          start_day: int = 0) -> List[Dict]:
    rows: List[Dict] = []
    for s in range(n_sessions):
        day = start_day + int(rng.integers(0, 5))
        sess = _Session(rng, user_id, role, ip,
                        f'sess_{user_id}_{s:02d}',
                        BASE_DATE + timedelta(days=day,
                                              hours=int(rng.integers(9, 20)),
                                              minutes=int(rng.integers(0, 60))))
        n = int(rng.integers(3, 8))
        sess.add('login', 'normal')
        for _ in range(n):
            sess.add(str(rng.choice(NORMAL_EVENT_TYPES)), 'normal')
        rows.extend(sess.rows)
    return rows


def _sensitive_session(rng: np.random.Generator, user_id: str, role: str,
                       ip: str, session_id: str, start: datetime,
                       tier: str, n_sensitive: int = 3,
                       sensitive_types: Optional[List[str]] = None,
                       after_hours: bool = False) -> List[Dict]:
    """A session containing a burst of sensitive operations."""
    types = sensitive_types or SENSITIVE_EVENT_TYPES
    sess = _Session(rng, user_id, role, ip, session_id, start)
    sess.add('login', tier, after_hours=int(after_hours))
    # A few normal-looking events first (blend in), then the burst.
    for _ in range(int(rng.integers(1, 3))):
        sess.add(str(rng.choice(NORMAL_EVENT_TYPES)),
                 'normal' if tier != 'high_risk' else 'suspicious',
                 after_hours=int(after_hours))
    for _ in range(n_sensitive):
        sess.add(str(rng.choice(types)), tier,
                 after_hours=int(after_hours))
    return sess.rows


# ----------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------

def _scenario_baseline(seed: int = 2301) -> pd.DataFrame:
    """Mostly-normal control scenario; should yield few/no incidents."""
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for i in range(8):
        uid = f'norm_{i:02d}'
        ip = f'192.0.2.{10 + i}'
        rows.extend(_normal_user_sessions(
            rng, uid, 'student' if i % 4 else 'instructor', ip,
            n_sessions=int(rng.integers(2, 4))))
    # Two normal students share a dorm IP — must NOT trigger shared_ip.
    for uid in ('norm_dorm_a', 'norm_dorm_b'):
        rows.extend(_normal_user_sessions(
            rng, uid, 'student', '192.0.2.77', n_sessions=2))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scenario_insider_exfiltration(seed: int = 2302) -> pd.DataFrame:
    """Instructor exports grades/rosters across several days and sessions.

    Exercises: repeated_sensitive_operations, long_term_behavior_change,
    repeated_pattern_across_sessions.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    uid, ip = 'prof_kessler', '198.51.100.31'
    # Early window: normal teaching activity.
    for s in range(2):
        day = s
        sess = _Session(rng, uid, 'instructor', ip, f'sess_{uid}_early{s}',
                        BASE_DATE + timedelta(days=day, hours=10))
        sess.add('login', 'normal')
        for _ in range(int(rng.integers(3, 6))):
            sess.add(str(rng.choice(['course_view', 'grade_view',
                                     'assignment_submit', 'module_view'])),
                     'normal')
        rows.extend(sess.rows)
    # Late window: escalating sensitive exports on 4 distinct days.
    for s, day in enumerate([4, 5, 6, 7]):
        start = BASE_DATE + timedelta(days=day, hours=23 if s % 2 else 1)
        rows.extend(_sensitive_session(
            rng, uid, 'instructor', ip, f'sess_{uid}_late{s}', start,
            tier='high_risk' if s >= 2 else 'suspicious',
            n_sensitive=3 + s,
            sensitive_types=['grade_download', 'student_export',
                             'course_export'],
            after_hours=bool(s % 2)))
    # Normal background users for contrast.
    for i in range(4):
        rows.extend(_normal_user_sessions(
            rng, f'bg_{i}', 'student', f'192.0.2.{40 + i}', n_sessions=2))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scenario_credential_stuffing_ring(seed: int = 2303) -> pd.DataFrame:
    """Two accounts on one IP run auth-failure bursts then bulk exports.

    Exercises: shared_ip_activity (corroborated), similar_account_behavior,
    repeated_sensitive_operations.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    shared_ip = '198.51.100.88'
    for uid in ('ring_alice', 'ring_bob'):
        for s, day in enumerate([1, 3]):
            sess = _Session(rng, uid, 'student', shared_ip,
                            f'sess_{uid}_{s}',
                            BASE_DATE + timedelta(days=day, hours=3))
            # Auth-failure burst then sensitive export burst.
            for _ in range(int(rng.integers(2, 4))):
                sess.add_auth_failure('login', tier='suspicious')
            sess.add('login', 'suspicious')
            for _ in range(int(rng.integers(3, 5))):
                sess.add(str(rng.choice(SENSITIVE_EVENT_TYPES)),
                         'high_risk' if s else 'suspicious',
                         after_hours=1)
            rows.extend(sess.rows)
    # Innocent users on other IPs.
    for i in range(4):
        rows.extend(_normal_user_sessions(
            rng, f'innocent_{i}', 'student', f'192.0.2.{60 + i}',
            n_sessions=2))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scenario_after_hours_lab(seed: int = 2304) -> pd.DataFrame:
    """Several students share a lab IP doing sensitive ops after hours.

    Exercises: repeated_sensitive_operations (per user), shared_ip_activity
    (uncorroborated unless profiles also match).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    lab_ip = '192.0.2.200'
    students = [f'lab_{i:02d}' for i in range(4)]
    for uid in students:
        for s, day in enumerate([2, 5]):
            rows.extend(_sensitive_session(
                rng, uid, 'student', lab_ip, f'sess_{uid}_{s}',
                BASE_DATE + timedelta(days=day, hours=23),
                tier='suspicious', n_sensitive=2 + (s % 2),
                after_hours=True))
    for i in range(3):
        rows.extend(_normal_user_sessions(
            rng, f'outside_{i}', 'student', f'198.51.100.{120 + i}',
            n_sessions=2))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scenario_low_and_slow(seed: int = 2305) -> pd.DataFrame:
    """One user's behavior drifts from normal to suspicious over two weeks.

    Exercises: long_term_behavior_change.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    uid, ip = 'drifting_sam', '198.51.100.55'
    # Early window: ~20 normal events ending day 4, leaving a clean time
    # gap before the late window so the median split separates them.
    for s, day in enumerate([0, 2, 4]):
        sess = _Session(rng, uid, 'student', ip, f'sess_{uid}_w1_{s}',
                        BASE_DATE + timedelta(days=day, hours=14))
        sess.add('login', 'normal')
        for _ in range(int(rng.integers(5, 7))):
            sess.add(str(rng.choice(NORMAL_EVENT_TYPES[:4])), 'normal')
        rows.extend(sess.rows)
    # Late window: ~20 events over days 8-10 — auth failures, sensitive
    # ops, after-hours activity, wider event-type mix.
    late_days = [8, 9, 10]
    for s, day in enumerate(late_days):
        start = BASE_DATE + timedelta(days=day,
                                      hours=22 if s % 2 else 15)
        sess = _Session(rng, uid, 'student', ip, f'sess_{uid}_w2_{s}', start)
        sess.add_auth_failure('login', tier='suspicious')
        sess.add_auth_failure('login', tier='suspicious')
        sess.add('login', 'suspicious')
        for _ in range(int(rng.integers(1, 3))):
            sess.add(str(rng.choice(NORMAL_EVENT_TYPES)), 'suspicious',
                     after_hours=int(s % 2))
        for _ in range(3 + s):
            sess.add(str(rng.choice(SENSITIVE_EVENT_TYPES)), 'suspicious',
                     after_hours=int(s % 2))
        rows.extend(sess.rows)
    for i in range(4):
        rows.extend(_normal_user_sessions(
            rng, f'stable_{i}', 'student', f'192.0.2.{80 + i}',
            n_sessions=2))
    return pd.DataFrame(rows, columns=COLUMNS)


def _scenario_mixed_campaign(seed: int = 2306) -> pd.DataFrame:
    """Combined campaign: insider + credential ring + after-hours lab.

    The primary demo scenario — designed to produce multiple distinct
    incidents across all five correlation types.
    """
    component_generators = [
        ('insider_exfiltration', _scenario_insider_exfiltration),
        ('credential_stuffing_ring', _scenario_credential_stuffing_ring),
        ('after_hours_lab', _scenario_after_hours_lab),
        ('low_and_slow', _scenario_low_and_slow),
    ]
    if seed == SCENARIOS['mixed_campaign']['seed']:
        # Default mixed campaign composes the canonical component
        # scenarios so documented coverage (including low_and_slow drift)
        # is preserved. Custom seeds keep deterministic offset behavior.
        parts = [gen(SCENARIOS[name]['seed']) for name, gen in
                 component_generators]
    else:
        parts = [gen(seed + i) for i, (_, gen) in
                 enumerate(component_generators)]
    # Component user prefixes are distinct, so no renaming is needed.
    df = pd.concat(parts, ignore_index=True)
    return df[COLUMNS]


# ----------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------

SCENARIOS: Dict[str, Dict] = {
    'baseline': {
        'title': 'Baseline (mostly normal)',
        'description': 'Control scenario — normal users, one shared dorm '
                       'IP. Expect few or no correlation findings.',
        'generator': _scenario_baseline,
        'seed': 2301,
    },
    'insider_exfiltration': {
        'title': 'Insider data exfiltration',
        'description': 'Instructor exports grades across sessions and days '
                       '— repeated sensitive ops + behavior drift.',
        'generator': _scenario_insider_exfiltration,
        'seed': 2302,
    },
    'credential_stuffing_ring': {
        'title': 'Credential stuffing ring',
        'description': 'Two accounts share an IP, run auth-failure bursts, '
                       'then bulk exports — corroborated shared-IP lead.',
        'generator': _scenario_credential_stuffing_ring,
        'seed': 2303,
    },
    'after_hours_lab': {
        'title': 'After-hours lab access',
        'description': 'Several students share a lab IP doing sensitive '
                       'ops late at night.',
        'generator': _scenario_after_hours_lab,
        'seed': 2304,
    },
    'low_and_slow': {
        'title': 'Low-and-slow drift',
        'description': 'One user gradually shifts from normal to '
                       'suspicious behavior over two weeks.',
        'generator': _scenario_low_and_slow,
        'seed': 2305,
    },
    'mixed_campaign': {
        'title': 'Mixed campaign (demo)',
        'description': 'Insider + credential ring + lab + drifter combined '
                       '— the full demonstration scenario.',
        'generator': _scenario_mixed_campaign,
        'seed': 2306,
    },
}

DEFAULT_SCENARIO_KEY = 'default_sample'


def generate_scenario(name: str,
                      seed: Optional[int] = None) -> pd.DataFrame:
    """Generate a named scenario as a raw-log-schema DataFrame."""
    if name not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {name}. '
                         f'Available: {sorted(SCENARIOS)}')
    spec = SCENARIOS[name]
    return spec['generator'](seed if seed is not None else spec['seed'])


def scenario_names() -> List[str]:
    return list(SCENARIOS)


def save_scenario(name: str, seed: Optional[int] = None,
                  out_dir: str = SCENARIO_DIR) -> str:
    """Generate a scenario and persist it as CSV under data/scenarios/."""
    df = generate_scenario(name, seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{name}.csv')
    df.to_csv(path, index=False)
    return path
