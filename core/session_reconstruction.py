"""
Session reconstruction for CAID — Milestone 2.0

Groups security events into deterministic sessions and derives two feature
views:

1. CAUSAL session-prefix features attached to every event. These use only
   events up to and including the current event, so earlier predictions can
   never be influenced by future events.
2. Completed-session summaries for dashboards and retrospective
   session-level evaluation. These are NOT used for per-event scoring.

Session identity rules:
- A non-empty explicit ``session_id`` column wins when present.
- Otherwise events are grouped per ``user_id`` using a configurable
  inactivity timeout.
- Missing timestamps are isolated into deterministic per-event sessions;
  no duration or rate is fabricated for them.
- Duplicate rows are preserved and flagged via ``is_duplicate_event``.

All data is synthetic/defensive research data. No label, IP-prefix, or
synthetic-label-derived field is used as a feature.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SENSITIVE_EVENT_TYPES = (
    'grade_download',
    'student_export',
    'course_export',
    'bulk_download',
    'bulk_export',
)

DEFAULT_AUTH_COLUMN = 'auth_result'
DEFAULT_USER_COLUMN = 'user_id'
DEFAULT_TIMESTAMP_COLUMN = 'timestamp'
DEFAULT_EVENT_COLUMN = 'event_type'
DEFAULT_IP_COLUMN = 'source_ip'
DEFAULT_SESSION_COLUMN = 'session_id'

# Scalar event fields used for deterministic duplicate detection. Pipeline
# outputs (lists/dicts such as risk_factors/evidence) are intentionally
# excluded so hashing never hits unhashable values.
DUPLICATE_KEY_COLUMNS = (
    'timestamp', 'user_id', 'role', 'source_ip', 'event_type',
    'courses_accessed', 'accounts_accessed', 'api_requests', 'data_mb',
    'token_age_minutes', 'after_hours', 'label', 'session_id', 'auth_result',
)

# Ordered causal prefix features consumed by SessionBaselineManager and the
# optional session-aware ML experiment. Order is persisted in artifacts.
SESSION_PREFIX_FEATURES: List[str] = [
    'session_prefix_event_index',
    'session_prefix_event_count',
    'session_prefix_duration_seconds',
    'session_prefix_event_rate',
    'session_prefix_unique_event_types',
    'session_prefix_api_requests',
    'session_prefix_data_volume_mb',
    'session_prefix_unique_ips',
    'session_prefix_ip_switch_count',
    'session_prefix_sensitive_event_count',
    'session_prefix_after_hours_count',
    'session_prefix_auth_failures',
    'session_prefix_auth_successes',
    'session_prefix_seconds_since_previous',
]

# Completed-session summary columns (retrospective; dashboard/eval only).
SESSION_SUMMARY_FEATURES: List[str] = [
    'session_event_count',
    'session_unique_event_types',
    'session_api_requests',
    'session_data_volume_mb',
    'session_unique_ips',
    'session_sensitive_event_count',
    'session_after_hours_count',
    'session_ip_switch_count',
    'session_duration_seconds',
    'session_event_rate',
    'session_auth_failures',
    'session_auth_successes',
]


@dataclass
class SessionConfig:
    """Configuration for deterministic session reconstruction."""

    inactivity_timeout_seconds: float = 1800.0
    explicit_session_column: str = DEFAULT_SESSION_COLUMN
    user_column: str = DEFAULT_USER_COLUMN
    timestamp_column: str = DEFAULT_TIMESTAMP_COLUMN
    event_column: str = DEFAULT_EVENT_COLUMN
    ip_column: str = DEFAULT_IP_COLUMN
    auth_column: str = DEFAULT_AUTH_COLUMN
    sensitive_event_types: Tuple[str, ...] = DEFAULT_SENSITIVE_EVENT_TYPES
    missing_timestamp_policy: str = 'isolate'
    duplicate_policy: str = 'flag'
    tie_break: str = 'content_hash'

    def to_dict(self) -> Dict:
        return {
            'inactivity_timeout_seconds': float(self.inactivity_timeout_seconds),
            'explicit_session_column': self.explicit_session_column,
            'user_column': self.user_column,
            'timestamp_column': self.timestamp_column,
            'event_column': self.event_column,
            'ip_column': self.ip_column,
            'auth_column': self.auth_column,
            'sensitive_event_types': list(self.sensitive_event_types),
            'missing_timestamp_policy': self.missing_timestamp_policy,
            'duplicate_policy': self.duplicate_policy,
            'tie_break': self.tie_break,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'SessionConfig':
        if not data:
            return cls()
        kwargs = dict(cls().to_dict())
        kwargs.update({k: v for k, v in data.items() if k in kwargs})
        if 'sensitive_event_types' in kwargs:
            kwargs['sensitive_event_types'] = tuple(kwargs['sensitive_event_types'])
        return cls(**kwargs)


def _norm_id(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'unknown'
    text = str(value).strip()
    return text if text else 'unknown'


def _norm_token(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ''
    text = str(value).strip()
    return '' if text.lower() in {'nan', 'none', 'nat'} else text


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _content_hash(row: pd.Series, cols: List[str]) -> str:
    payload = []
    for col in cols:
        val = row.get(col)
        if isinstance(val, float) and np.isnan(val):
            val = None
        payload.append(str(val))
    blob = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def _is_failure(value) -> bool:
    return str(value).strip().lower() in {'failure', 'failed', 'fail', 'false', '0', 'denied'}


def _is_success(value) -> bool:
    return str(value).strip().lower() in {'success', 'succeeded', 'true', '1', 'allow', 'allowed'}


class SessionReconstructor:
    """Deterministic session reconstruction and feature extraction."""

    def __init__(self, config: Optional[SessionConfig] = None):
        self.config = config or SessionConfig()

    # ------------------------------------------------------------------
    # Session assignment
    # ------------------------------------------------------------------

    def _assign_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = df.copy()
        n = len(out)
        if n == 0:
            out['session_id'] = pd.Series(dtype=object)
            out['session_event_index'] = pd.Series(dtype=int)
            out['session_timestamp_missing'] = pd.Series(dtype=bool)
            out['is_duplicate_event'] = pd.Series(dtype=bool)
            return out

        out['_orig_pos'] = np.arange(n)
        out['_user_key'] = out.get(cfg.user_column, pd.Series('unknown', index=out.index)).map(_norm_id)
        out['_ts'] = pd.to_datetime(out.get(cfg.timestamp_column), errors='coerce') if cfg.timestamp_column in out.columns else pd.NaT

        explicit_col = cfg.explicit_session_column
        explicit = out.get(explicit_col, pd.Series('', index=out.index)).map(_norm_token) if explicit_col in out.columns else pd.Series('', index=out.index)
        has_explicit = explicit.ne('')

        session_ids = pd.Series('', index=out.index, dtype=object)
        out['session_timestamp_missing'] = out['_ts'].isna()

        # Explicit sessions: trust the supplied identifier, scoped by user so
        # identical session_id values across users cannot merge.
        if has_explicit.any():
            session_ids.loc[has_explicit] = (
                'explicit:' + out.loc[has_explicit, '_user_key'] + ':' + explicit.loc[has_explicit]
            )

        # Timed, non-explicit rows: per-user inactivity timeout.
        timed_mask = (~has_explicit) & out['_ts'].notna()
        if timed_mask.any():
            timed = out.loc[timed_mask].copy()
            hash_cols = [c for c in (cfg.user_column, cfg.timestamp_column,
                                     cfg.event_column, cfg.ip_column,
                                     cfg.auth_column) if c in timed.columns]
            timed['_tie'] = [_content_hash(r, hash_cols) for _, r in timed.iterrows()]
            timed = timed.sort_values(['_user_key', '_ts', '_tie', '_orig_pos'],
                                      kind='mergesort')
            assigned = {}
            timeout = float(cfg.inactivity_timeout_seconds)
            for user_id, grp in timed.groupby('_user_key', sort=False):
                ordinal = 0
                prev_ts = None
                current_id = None
                for pos, ts in zip(grp['_orig_pos'].to_numpy(), grp['_ts'].to_numpy()):
                    ts_val = pd.Timestamp(ts)
                    if current_id is None or prev_ts is None or \
                            (ts_val - prev_ts).total_seconds() > timeout:
                        ordinal += 1
                        start = ts_val.isoformat()
                        current_id = f'timeout:{user_id}:{start}:{ordinal:04d}'
                    assigned[int(pos)] = current_id
                    prev_ts = ts_val
            for pos, sid in assigned.items():
                session_ids.iloc[int(pos)] = sid

        # Missing timestamps without explicit IDs are isolated so no duration,
        # ordering, or transition is fabricated.
        missing_mask = (~has_explicit) & out['_ts'].isna()
        if missing_mask.any():
            for pos in np.flatnonzero(missing_mask.to_numpy()):
                user_id = out['_user_key'].iloc[int(pos)]
                session_ids.iloc[int(pos)] = f'missing_ts:{user_id}:{int(pos):08d}'

        out['session_id'] = session_ids.fillna('unknown')

        # Duplicate flag: deterministic content hash over scalar event fields.
        if cfg.duplicate_policy == 'flag':
            key_cols = [c for c in DUPLICATE_KEY_COLUMNS if c in out.columns]
            if key_cols:
                hashes = pd.Series(
                    [_content_hash(r, key_cols) for _, r in out.iterrows()],
                    index=out.index)
                counts = hashes.map(hashes.value_counts())
                out['is_duplicate_event'] = counts > 1
            else:
                out['is_duplicate_event'] = False
        else:
            out['is_duplicate_event'] = False

        out = out.drop(columns=[c for c in ('_orig_pos', '_user_key', '_ts', '_tie') if c in out.columns])
        return out

    # ------------------------------------------------------------------
    # Causal prefix features
    # ------------------------------------------------------------------

    def _add_prefix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = df.copy()
        n = len(out)
        prefix_cols = list(dict.fromkeys(SESSION_PREFIX_FEATURES + [
            'session_prefix_event_type', 'session_prefix_prev_event_type',
            'session_prefix_transition_key', 'session_prefix_has_timestamp',
            'session_prefix_ip_switch_count',
        ]))
        for col in prefix_cols:
            if col not in out.columns:
                out[col] = np.nan
            # Object dtype avoids lossy setitem when mixing ints/floats/strings.
            out[col] = out[col].astype(object)

        sensitive = set(cfg.sensitive_event_types)
        ts = pd.to_datetime(out.get(cfg.timestamp_column), errors='coerce') \
            if cfg.timestamp_column in out.columns else pd.Series(pd.NaT, index=out.index)

        # Process each session in deterministic order. Within a session, timed
        # events sort by timestamp; missing timestamps keep original order last.
        work = out.copy()
        work['_ts'] = ts
        work['_orig'] = np.arange(n)
        # Deterministic tie-break for identical timestamps so shuffled input
        # order cannot change causal prefix ordering. Label/session columns are
        # excluded so ordering never depends on evaluation metadata.
        tie_cols = [c for c in DUPLICATE_KEY_COLUMNS
                    if c in work.columns and c not in {'label', 'session_id'}]
        work['_tie'] = ([_content_hash(r, tie_cols) for _, r in work.iterrows()]
                        if tie_cols else '')
        work = work.sort_values(['session_id', '_ts', '_tie', '_orig'],
                                na_position='last', kind='mergesort')

        prefix_values: Dict[str, list] = {c: [] for c in prefix_cols}
        order_positions = work.index.to_numpy()

        for _, grp in work.groupby('session_id', sort=True):
            event_types: set = set()
            ips: List[str] = []
            ip_set: set = set()
            first_ts = None
            prev_ts = None
            prev_event = ''
            prev_ip = None
            ip_switches = 0
            api_sum = 0.0
            data_sum = 0.0
            sensitive_count = 0
            after_hours_count = 0
            auth_fail = 0
            auth_success = 0
            event_count = 0

            for row in grp.itertuples(index=False):
                row_d = row._asdict()
                event_count += 1
                event_type = _norm_token(row_d.get(cfg.event_column, '')) or 'unknown'
                ip = _norm_id(row_d.get(cfg.ip_column, 'unknown'))
                event_types.add(event_type)
                ip_set.add(ip)
                if prev_ip is not None and ip != prev_ip:
                    ip_switches += 1
                prev_ip = ip

                api_sum += _safe_float(row_d.get('api_requests', 0.0))
                data_sum += _safe_float(row_d.get('data_mb', 0.0))
                if event_type in sensitive:
                    sensitive_count += 1
                after_hours_count += int(_safe_float(row_d.get('after_hours', 0.0)) > 0)
                auth_val = row_d.get(cfg.auth_column)
                if auth_val is not None and str(auth_val).strip() != '':
                    if _is_failure(auth_val):
                        auth_fail += 1
                    elif _is_success(auth_val):
                        auth_success += 1

                cur_ts = row_d.get('_ts')
                cur_ts = pd.Timestamp(cur_ts) if pd.notna(cur_ts) else None
                if cur_ts is not None and first_ts is None:
                    first_ts = cur_ts
                duration = np.nan
                rate = np.nan
                since_prev = np.nan
                has_ts = int(cur_ts is not None)
                if cur_ts is not None and first_ts is not None:
                    duration = float((cur_ts - first_ts).total_seconds())
                    elapsed_hours = max(duration / 3600.0, 1.0 / 3600.0)
                    rate = float(event_count / elapsed_hours)
                if cur_ts is not None and prev_ts is not None:
                    since_prev = float((cur_ts - prev_ts).total_seconds())
                if cur_ts is not None:
                    prev_ts = cur_ts

                transition = f'{prev_event}->{event_type}' if prev_event else ''
                vals = {
                    'session_prefix_event_index': event_count,
                    'session_prefix_event_count': event_count,
                    'session_prefix_duration_seconds': duration,
                    'session_prefix_event_rate': rate,
                    'session_prefix_unique_event_types': len(event_types),
                    'session_prefix_api_requests': api_sum,
                    'session_prefix_data_volume_mb': data_sum,
                    'session_prefix_unique_ips': len(ip_set),
                    'session_prefix_ip_switch_count': ip_switches,
                    'session_prefix_sensitive_event_count': sensitive_count,
                    'session_prefix_after_hours_count': after_hours_count,
                    'session_prefix_auth_failures': auth_fail,
                    'session_prefix_auth_successes': auth_success,
                    'session_prefix_seconds_since_previous': since_prev,
                    'session_prefix_event_type': event_type,
                    'session_prefix_prev_event_type': prev_event,
                    'session_prefix_transition_key': transition,
                    'session_prefix_has_timestamp': has_ts,
                }
                for col in prefix_cols:
                    prefix_values[col].append(vals.get(col, ''))
                prev_event = event_type

        # prefix_values were appended in `work` row order; align by index.
        prefix_df = pd.DataFrame(prefix_values, index=work.index)
        for col in prefix_cols:
            out[col] = prefix_df[col].reindex(out.index)

        # Fill numeric prefix columns cleanly.
        numeric_prefix = [c for c in SESSION_PREFIX_FEATURES]
        for col in numeric_prefix:
            out[col] = pd.to_numeric(out[col], errors='coerce')
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconstruct(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return events with deterministic session IDs and causal prefix features."""
        if df is None:
            raise ValueError('Cannot reconstruct sessions for None DataFrame')
        assigned = self._assign_sessions(df)
        return self._add_prefix_features(assigned)

    def session_summaries(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Return completed-session summaries (retrospective use only)."""
        if events_df is None or len(events_df) == 0 or 'session_id' not in events_df.columns:
            return pd.DataFrame(columns=['session_id', 'user_id'] + SESSION_SUMMARY_FEATURES)
        cfg = self.config
        ev = events_df.copy()
        ev['_event_type_norm'] = ev.get(cfg.event_column, pd.Series('', index=ev.index)).map(
            lambda x: _norm_token(x) or 'unknown')
        ev['_is_sensitive'] = ev['_event_type_norm'].isin(set(cfg.sensitive_event_types)).astype(int)
        ev['_after_hours'] = pd.to_numeric(ev.get('after_hours', 0), errors='coerce').fillna(0) > 0
        ev['_auth_fail'] = ev.get(cfg.auth_column, pd.Series('', index=ev.index)).map(_is_failure).astype(int)
        ev['_auth_success'] = ev.get(cfg.auth_column, pd.Series('', index=ev.index)).map(_is_success).astype(int)
        ev['_ts'] = pd.to_datetime(ev.get(cfg.timestamp_column), errors='coerce') \
            if cfg.timestamp_column in ev.columns else pd.NaT

        def _auth_any(s):
            return int(((s == 'failure') | (s == 'failed') | (s == 'fail') |
                        (s == 'success') | (s == 'succeeded') |
                        (s == 'true') | (s == 'false') |
                        (s == '1') | (s == '0') |
                        (s == 'allow') | (s == 'allowed') |
                        (s == 'denied')).sum()) if len(s) else 0

        agg = ev.groupby('session_id', sort=True).agg(
            user_id=(cfg.user_column, 'first'),
            session_event_count=('session_id', 'size'),
            session_unique_event_types=('_event_type_norm', 'nunique'),
            session_api_requests=('api_requests', lambda s: float(pd.to_numeric(s, errors='coerce').fillna(0).sum())),
            session_data_volume_mb=('data_mb', lambda s: float(pd.to_numeric(s, errors='coerce').fillna(0).sum())),
            session_unique_ips=(cfg.ip_column, 'nunique'),
            session_sensitive_event_count=('_is_sensitive', 'sum'),
            session_after_hours_count=('_after_hours', 'sum'),
            session_ip_switch_count=('session_prefix_ip_switch_count', 'max'),
            session_auth_failures=('_auth_fail', 'sum'),
            session_auth_successes=('_auth_success', 'sum'),
            session_first_ts=('_ts', 'min'),
            session_last_ts=('_ts', 'max'),
            session_has_missing_ts=('session_timestamp_missing', 'any'),
            session_explicit_id=('session_id', lambda s: str(s.iloc[0]).startswith('explicit:')),
        ).reset_index()

        duration = (agg['session_last_ts'] - agg['session_first_ts']).dt.total_seconds()
        agg['session_duration_seconds'] = duration.where(duration.notna(), np.nan)
        hours = (agg['session_duration_seconds'] / 3600.0).replace(0, np.nan)
        agg['session_event_rate'] = agg['session_event_count'] / hours
        return agg

    def reconstruct_and_summarize(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        events = self.reconstruct(df)
        return events, self.session_summaries(events)


def prepare_session_features(df: pd.DataFrame,
                             config: Optional[SessionConfig] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (events_with_prefix_features, session_summaries)."""
    recon = SessionReconstructor(config)
    return recon.reconstruct_and_summarize(df)
