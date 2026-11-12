"""
Canonical streaming event schema and validation — Milestone 2.5.

Every ingested record normalizes to the existing security-log schema plus a
streaming envelope (``event_id``, ``ingested_at_utc``, ``source``). Envelope
fields are carried for identity/dedup/lineage only and are NEVER fed to
detectors and never influence scores.

Production stream events must NOT contain synthetic benchmark ``label`` or
ground-truth outcomes — ``validate_event`` rejects them unless the caller
explicitly opts in (``allow_labels=True``, intended for tests only).
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import pandas as pd

from utils.validators import VALID_EVENT_TYPES, REQUIRED_COLUMNS

# Log-schema fields detectors consume (session_id/auth_result optional).
LOG_FIELDS = (
    'timestamp', 'user_id', 'role', 'source_ip', 'event_type',
    'courses_accessed', 'accounts_accessed', 'api_requests', 'data_mb',
    'token_age_minutes', 'after_hours', 'session_id', 'auth_result',
)

NUMERIC_FIELDS = (
    'courses_accessed', 'accounts_accessed', 'api_requests', 'data_mb',
    'token_age_minutes', 'after_hours',
)

_OPTIONAL_STR_FIELDS = ('role', 'source_ip', 'session_id', 'auth_result')

_EVENT_TYPE_VOCAB = {t.lower(): t for t in VALID_EVENT_TYPES}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_timestamp(value) -> pd.Timestamp:
    """Fast scalar ISO-8601 parse -> pd.Timestamp (NaT on failure).

    ``datetime.fromisoformat`` is ~10-50x cheaper than ``pd.to_datetime``
    for scalar strings; falls back to pandas for exotic formats. Output
    values are identical to ``pd.to_datetime`` for ISO inputs.
    """
    try:
        return pd.Timestamp(datetime.fromisoformat(str(value).strip()))
    except Exception:
        pass
    try:
        ts = pd.to_datetime(value, errors='coerce')
        return pd.Timestamp(ts) if pd.notna(ts) else pd.NaT
    except Exception:
        return pd.NaT


def _norm_str(value) -> str:
    if value is None:
        return ''
    if isinstance(value, float) and pd.isna(value):
        return ''
    return str(value).strip()


def _content_payload(fields: Dict[str, object]) -> str:
    """Canonical JSON of the log-schema fields for hashing/lineage."""
    blob = json.dumps(fields, sort_keys=True, default=str,
                      ensure_ascii=False)
    return blob


def compute_event_id(source: str, event_ts: str, user_id: str,
                     session_id: str, event_type: str,
                     payload_fields: Dict[str, object]) -> str:
    """
    Deterministic event identity:
    ``sha1(source | event_ts | user_id | session_id | event_type |
    payload-hash)[:16]`` — content-derived, replay-safe.
    """
    payload_hash = hashlib.sha256(
        _content_payload(payload_fields).encode('utf-8')).hexdigest()[:16]
    joined = '|'.join([
        str(source), str(event_ts), str(user_id), str(session_id or ''),
        str(event_type), payload_hash,
    ])
    return hashlib.sha1(joined.encode('utf-8')).hexdigest()[:16]


@dataclass
class StreamingEvent:
    """Canonical normalized event: log schema + streaming envelope."""

    # Envelope
    event_id: str
    ingested_at_utc: str
    source: str
    # Log schema (detectors consume only these)
    timestamp: str
    user_id: str
    role: str = ''
    source_ip: str = ''
    event_type: str = ''
    courses_accessed: float = 0.0
    accounts_accessed: float = 0.0
    api_requests: float = 0.0
    data_mb: float = 0.0
    token_age_minutes: float = 0.0
    after_hours: int = 0
    session_id: str = ''
    auth_result: str = ''
    # Ordering metadata (not detector inputs)
    event_ts: Optional[pd.Timestamp] = field(default=None, repr=False)
    ordering_key: Tuple = field(default=None, repr=False)
    # Lineage for replayed events
    replay_of: str = ''

    def to_log_dict(self) -> Dict[str, object]:
        """Log-schema fields only — the shape detectors consume."""
        out = {
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'role': self.role,
            'source_ip': self.source_ip,
            'event_type': self.event_type,
            'courses_accessed': self.courses_accessed,
            'accounts_accessed': self.accounts_accessed,
            'api_requests': self.api_requests,
            'data_mb': self.data_mb,
            'token_age_minutes': self.token_age_minutes,
            'after_hours': self.after_hours,
            'session_id': self.session_id,
            'auth_result': self.auth_result,
        }
        return out

    def to_dict(self) -> Dict[str, object]:
        out = self.to_log_dict()
        out.update({
            'event_id': self.event_id,
            'ingested_at_utc': self.ingested_at_utc,
            'source': self.source,
        })
        if self.replay_of:
            out['replay_of'] = self.replay_of
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> 'StreamingEvent':
        """Rebuild an event from its serialized dict (queue/replay)."""
        if not isinstance(data, dict):
            raise ValueError('Event payload must be a dict')
        log = {k: data.get(k) for k in LOG_FIELDS}
        ts_raw = _norm_str(log.get('timestamp'))
        ts = _to_timestamp(ts_raw)
        ev = cls(
            event_id=_norm_str(data.get('event_id')),
            ingested_at_utc=_norm_str(data.get('ingested_at_utc')),
            source=_norm_str(data.get('source')),
            timestamp=ts_raw,
            user_id=_norm_str(log.get('user_id')),
            role=_norm_str(log.get('role')),
            source_ip=_norm_str(log.get('source_ip')),
            event_type=_norm_str(log.get('event_type')),
            courses_accessed=float(log.get('courses_accessed') or 0.0),
            accounts_accessed=float(log.get('accounts_accessed') or 0.0),
            api_requests=float(log.get('api_requests') or 0.0),
            data_mb=float(log.get('data_mb') or 0.0),
            token_age_minutes=float(log.get('token_age_minutes') or 0.0),
            after_hours=int(float(log.get('after_hours') or 0.0)),
            session_id=_norm_str(log.get('session_id')),
            auth_result=_norm_str(log.get('auth_result')),
            replay_of=_norm_str(data.get('replay_of')),
        )
        ev.event_ts = ts if pd.notna(ts) else None
        ev.ordering_key = (ev.event_ts, ev.event_id)
        return ev


class BoundedDedupLRU:
    """Bounded LRU of seen event_ids for duplicate detection."""

    def __init__(self, maxsize: int = 100_000):
        self.maxsize = int(maxsize)
        self._seen: OrderedDict[str, None] = OrderedDict()

    def __len__(self) -> int:
        return len(self._seen)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._seen

    def check_and_add(self, event_id: str) -> bool:
        """Return True if the event_id was already seen (duplicate)."""
        if event_id in self._seen:
            self._seen.move_to_end(event_id)
            return True
        self._seen[event_id] = None
        while len(self._seen) > self.maxsize:
            self._seen.popitem(last=False)
        return False

    def snapshot_keys(self) -> List[str]:
        return list(self._seen.keys())

    def restore_keys(self, keys) -> None:
        self._seen = OrderedDict()
        for key in list(keys or [])[-self.maxsize:]:
            self._seen[str(key)] = None


def validate_event(raw: object, *, source: str = 'api',
                   allow_labels: bool = False,
                   ingested_at_utc: Optional[str] = None,
                   replay_of: str = '') -> Tuple[Optional[StreamingEvent], Optional[str]]:
    """
    Validate and normalize a raw event dict into a StreamingEvent.

    Returns ``(event, None)`` on success or ``(None, reason)`` on rejection.
    Rejections must be routed to the DLQ path — never silently dropped.
    """
    if not isinstance(raw, dict):
        return None, 'not_a_dict'

    if not allow_labels and 'label' in raw:
        return None, 'label_present_in_production_stream'

    missing = [c for c in REQUIRED_COLUMNS if c not in raw]
    if missing:
        return None, f'missing_fields:{",".join(missing)}'

    timestamp_raw = _norm_str(raw.get('timestamp'))
    ts = _to_timestamp(timestamp_raw)
    if not timestamp_raw or pd.isna(ts):
        return None, 'invalid_timestamp'

    user_id = _norm_str(raw.get('user_id'))
    if not user_id:
        return None, 'missing_user_id'

    event_type_raw = _norm_str(raw.get('event_type'))
    canonical = _EVENT_TYPE_VOCAB.get(event_type_raw.lower())
    if canonical is None:
        return None, f'unknown_event_type:{event_type_raw}'

    normalized: Dict[str, object] = {}
    for col in NUMERIC_FIELDS:
        val = raw.get(col, 0)
        try:
            num = float(val)
        except (TypeError, ValueError):
            return None, f'invalid_numeric:{col}'
        if not pd.notna(num):
            num = 0.0
        normalized[col] = num
    normalized['after_hours'] = int(normalized['after_hours'] != 0)

    log_fields: Dict[str, object] = {
        'timestamp': ts.isoformat() if hasattr(ts, 'isoformat') else timestamp_raw,
        'user_id': user_id,
        'role': _norm_str(raw.get('role')),
        'source_ip': _norm_str(raw.get('source_ip')),
        'event_type': canonical,
        'courses_accessed': normalized['courses_accessed'],
        'accounts_accessed': normalized['accounts_accessed'],
        'api_requests': normalized['api_requests'],
        'data_mb': normalized['data_mb'],
        'token_age_minutes': normalized['token_age_minutes'],
        'after_hours': normalized['after_hours'],
        'session_id': _norm_str(raw.get('session_id')),
        'auth_result': _norm_str(raw.get('auth_result')).lower(),
    }

    # Strip label from payload hash inputs — labels are never part of
    # event identity in the production stream.
    payload_fields = {k: log_fields[k] for k in LOG_FIELDS}

    event_id = compute_event_id(
        source=str(source), event_ts=log_fields['timestamp'],
        user_id=user_id, session_id=log_fields['session_id'],
        event_type=canonical, payload_fields=payload_fields)

    event = StreamingEvent(
        event_id=event_id,
        ingested_at_utc=ingested_at_utc or _utcnow_iso(),
        source=str(source),
        timestamp=log_fields['timestamp'],
        user_id=user_id,
        role=log_fields['role'],
        source_ip=log_fields['source_ip'],
        event_type=canonical,
        courses_accessed=normalized['courses_accessed'],
        accounts_accessed=normalized['accounts_accessed'],
        api_requests=normalized['api_requests'],
        data_mb=normalized['data_mb'],
        token_age_minutes=normalized['token_age_minutes'],
        after_hours=normalized['after_hours'],
        session_id=log_fields['session_id'],
        auth_result=log_fields['auth_result'],
        replay_of=replay_of,
    )
    event.event_ts = pd.Timestamp(ts)
    event.ordering_key = (event.event_ts, event.event_id)
    return event, None
