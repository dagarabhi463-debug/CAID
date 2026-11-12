"""
Incremental session & sequence state — Milestone 2.5.

Replicates, event-by-event, the causal semantics of the batch pipeline's
``SessionReconstructor`` (session assignment + prefix features) and
``SequenceAnalyzer.analyze`` (per-session causal window scoring), so
streaming scores match batch scores on identical ordered input.

Ordering policy (documented):
- Per-user timed (non-explicit-session) events are buffered and emitted in
  ``(event_ts, assign_tie_hash, arrival_seq)`` order — the same order the
  batch reconstructor uses.
- Explicit-``session_id`` events are buffered per ``(user, session)`` and
  emitted in ``(event_ts, prefix_tie_hash, arrival_seq)`` order.
- An event is emittable once the per-stream max event_ts has advanced
  ``reorder_window_seconds`` beyond it, or it has waited
  ``max_hold_seconds`` (whichever first) — bounding added latency.
- Events arriving with ``event_ts`` earlier than the last emitted ts for
  their stream cannot be inserted without retroactively changing already
  scored events; they are returned as late events for the DLQ/replay path.
- Missing timestamps get isolated sessions immediately (batch parity:
  ``missing_ts:{user}:{seq}``).

Known documented divergence: ``is_duplicate_event`` in batch flags ALL
occurrences of duplicated content in the whole dataset; streaming flags
all occurrences whose content hash is observed >= 2 times causally —
counting duplicates co-pending in the same drain batch AND prior
emitted occurrences. A duplicate separated from its first occurrence by
more than the reorder/hold window (first already committed) flags only
the later occurrence(s) — a bounded, documented divergence. The stream
hash also excludes ``label`` (production events carry none; batch drops
it from the key columns when absent anyway).
"""

from __future__ import annotations

import time
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.session_reconstruction import (
    DUPLICATE_KEY_COLUMNS, SESSION_PREFIX_FEATURES, SessionConfig,
    _content_hash, _is_failure, _is_success, _norm_id, _norm_token,
    _safe_float,
)
from core.session_baseline import SessionBaselineManager
from core.sequence_detection import (
    COMPONENT_WEIGHTS, SequenceAnalyzer, SequenceConfig,
)
from core.streaming.event_schema import StreamingEvent

# Columns hashed for session-assignment tie-breaks (mirrors
# SessionReconstructor._assign_sessions).
_ASSIGN_HASH_COLS = ('user_id', 'timestamp', 'event_type', 'source_ip',
                     'auth_result')
# Columns hashed for prefix-ordering tie-breaks (mirrors
# SessionReconstructor._add_prefix_features: DUPLICATE_KEY_COLUMNS minus
# label/session_id).
_PREFIX_HASH_COLS = tuple(c for c in DUPLICATE_KEY_COLUMNS
                          if c not in {'label', 'session_id'})
# Columns hashed for is_duplicate_event (mirrors _assign_sessions: all
# DUPLICATE_KEY_COLUMNS present — session_id resolved, label absent in
# the production stream).
_DUP_HASH_COLS = DUPLICATE_KEY_COLUMNS

# Extra prefix columns emitted alongside SESSION_PREFIX_FEATURES.
_PREFIX_EXTRA_COLS = (
    'session_prefix_event_type', 'session_prefix_prev_event_type',
    'session_prefix_transition_key', 'session_prefix_has_timestamp',
)

SENSITIVE_EVENT_TYPES_DEFAULT = (
    'grade_download', 'student_export', 'course_export',
    'bulk_download', 'bulk_export',
)


@dataclass
class _Pending:
    event: StreamingEvent
    sort_key: Tuple
    ingest_monotonic: float


@dataclass
class _SessionState:
    """Running causal state for one session (prefix + sequence window)."""
    session_id: str
    user_id: str
    # prefix aggregates (mirrors _add_prefix_features loop)
    event_types: set = field(default_factory=set)
    ips: List[str] = field(default_factory=list)
    ip_set: set = field(default_factory=set)
    first_ts: Optional[pd.Timestamp] = None
    prev_ts: Optional[pd.Timestamp] = None
    prev_event: str = ''
    prev_ip: Optional[str] = None
    ip_switches: int = 0
    api_sum: float = 0.0
    data_sum: float = 0.0
    sensitive_count: int = 0
    after_hours_count: int = 0
    auth_fail: int = 0
    auth_success: int = 0
    event_count: int = 0
    # sequence window state (mirrors SequenceAnalyzer.analyze loop)
    window: deque = field(default_factory=deque)


class _OrderingBuffer:
    """Per-stream reorder buffer with bounded latency."""

    def __init__(self, reorder_window_s: float, max_hold_s: float,
                 max_size: int):
        self.reorder_window_s = float(reorder_window_s)
        self.max_hold_s = float(max_hold_s)
        self.max_size = int(max_size)
        self.pending: List[_Pending] = []
        self.max_ts: Optional[pd.Timestamp] = None
        self.emitted_max_ts: Optional[pd.Timestamp] = None
        self.emitted_count = 0
        self.late_count = 0
        self.overflow_count = 0

    def add(self, event: StreamingEvent, sort_key: Tuple) -> None:
        self.pending.append(_Pending(event=event, sort_key=sort_key,
                                     ingest_monotonic=time.monotonic()))
        ts = event.event_ts
        if ts is not None and (self.max_ts is None or ts > self.max_ts):
            self.max_ts = ts
        if len(self.pending) > self.max_size:
            # Bound memory: expire the oldest entry by wall-clock age so the
            # next drain emits it first (ordering preserved — it is the
            # smallest sort key).
            self.pending.sort(key=lambda p: p.sort_key)
            self.pending[0].ingest_monotonic = 0.0
            self.overflow_count += 1

    def is_late(self, event_ts: pd.Timestamp) -> bool:
        return (self.emitted_max_ts is not None
                and event_ts < self.emitted_max_ts)

    def drain(self, now_monotonic: Optional[float] = None) -> List[_Pending]:
        """Emit the longest emittable prefix in sort order."""
        if not self.pending:
            return []
        now = now_monotonic if now_monotonic is not None else time.monotonic()
        self.pending.sort(key=lambda p: p.sort_key)
        frontier = (self.max_ts - pd.Timedelta(seconds=self.reorder_window_s)
                    ) if self.max_ts is not None else None
        out: List[_Pending] = []
        for p in self.pending:
            ts = p.event.event_ts
            by_frontier = (frontier is not None and ts is not None
                           and ts <= frontier)
            by_age = (now - p.ingest_monotonic) >= self.max_hold_s
            if not (by_frontier or by_age):
                break
            if (self.emitted_max_ts is not None and ts is not None
                    and ts < self.emitted_max_ts):
                # Should not happen (is_late filters at ingest), but never
                # emit out of order.
                break
            out.append(p)
        if out:
            self.pending = self.pending[len(out):]
            last_ts = out[-1].event.event_ts
            if last_ts is not None and (self.emitted_max_ts is None
                                        or last_ts > self.emitted_max_ts):
                self.emitted_max_ts = last_ts
            self.emitted_count += len(out)
        return out

    def force_flush(self) -> List[_Pending]:
        """Emit everything pending (shutdown/replay boundary)."""
        self.pending.sort(key=lambda p: p.sort_key)
        out = self.pending
        self.pending = []
        for p in out:
            ts = p.event.event_ts
            if ts is not None and (self.emitted_max_ts is None
                                   or ts > self.emitted_max_ts):
                self.emitted_max_ts = ts
        self.emitted_count += len(out)
        return out


class IncrementalSessionTracker:
    """
    Incremental replica of session reconstruction + causal sequence
    scoring for the streaming pipeline.

    Args:
        session_baseline_manager: fitted SessionBaselineManager (frozen
            artifact — never refit during inference).
        sequence_analyzer: fitted SequenceAnalyzer (frozen artifact).
        session_config / sequence_config: taken from the artifacts when
            not explicitly provided.
        reorder_window_seconds: event-time reorder horizon per stream.
        max_hold_seconds: max wall-clock buffering before forced emit.
        max_reorder_buffer: per-stream pending bound (overflow expires the
            oldest entry early; counted, never silently dropped).
        max_late_seconds: reserved for the late-event policy surface; the
            operative rule is "earlier than the emitted frontier".
    """

    def __init__(self, session_baseline_manager: SessionBaselineManager,
                 sequence_analyzer: SequenceAnalyzer,
                 session_config: Optional[SessionConfig] = None,
                 sequence_config: Optional[SequenceConfig] = None,
                 reorder_window_seconds: float = 0.25,
                 max_hold_seconds: float = 1.0,
                 max_reorder_buffer: int = 10_000,
                 max_late_seconds: float = 300.0):
        self.baseline_manager = session_baseline_manager
        self.sequence_analyzer = sequence_analyzer
        self.config = session_config or SessionConfig.from_dict(
            session_baseline_manager.session_config)
        self.seq_config = sequence_config or sequence_analyzer.config
        self.reorder_window_seconds = float(reorder_window_seconds)
        self.max_hold_seconds = float(max_hold_seconds)
        self.max_reorder_buffer = int(max_reorder_buffer)
        self.max_late_seconds = float(max_late_seconds)
        self.timeout = float(self.config.inactivity_timeout_seconds)
        self.sensitive = set(self.seq_config.sensitive_event_types
                             or SENSITIVE_EVENT_TYPES_DEFAULT)

        # Ordering domains: 'timed' per user_key; 'explicit' per
        # (user_key, session_id); 'missing' handled immediately.
        self._timed: Dict[str, _OrderingBuffer] = {}
        self._explicit: Dict[Tuple[str, str], _OrderingBuffer] = {}
        self._sessions: Dict[str, _SessionState] = {}
        # Per-user timed-session assignment state
        self._user_assign: Dict[str, dict] = {}
        self._arrival_seq = 0
        self.late_events = deque(maxlen=10_000)
        # Traceability for DLQ: event_id -> ordering-domain context captured
        # at the moment the event was declared late. Bounded like the queue.
        self.late_details: "OrderedDict[str, dict]" = OrderedDict()
        self._late_details_max = 10_000
        # Causal content-hash set for is_duplicate_event (bounded).
        self.content_seen: "OrderedDict[str, None]" = OrderedDict()
        self._content_seen_max = 100_000

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _record_late(self, event: StreamingEvent, ordering_domain: str,
                     buf: "_OrderingBuffer") -> None:
        self.late_events.append(event)
        detail = {
            'ordering_domain': ordering_domain,
            'emitted_frontier_ts': (
                buf.emitted_max_ts.isoformat()
                if buf.emitted_max_ts is not None else None),
            'event_ts': event.event_ts.isoformat()
            if event.event_ts is not None else None,
        }
        self.late_details[event.event_id] = detail
        self.late_details.move_to_end(event.event_id)
        while len(self.late_details) > self._late_details_max:
            self.late_details.popitem(last=False)

    def ingest(self, event: StreamingEvent
               ) -> Tuple[List[Dict], List[StreamingEvent]]:
        """
        Offer one validated event. Returns (scored_row_dicts, late_events)
        — rows are emitted only when their causal position is final.
        """
        self._arrival_seq += 1
        rows: List[Dict] = []
        late: List[StreamingEvent] = []

        ts = event.event_ts
        if ts is None:
            # Missing timestamp: isolated session, immediate (parity with
            # batch missing_ts handling).
            sid = (f"missing_ts:{_norm_id(event.user_id)}:"
                   f"{self._arrival_seq:08d}")
            rows.append(self._emit_single(event, sid))
            return rows, late

        user_key = _norm_id(event.user_id)
        sid_raw = _norm_token(event.session_id)

        if sid_raw:
            domain_key = (user_key, sid_raw)
            buf = self._explicit.get(domain_key)
            if buf is None:
                buf = self._explicit[domain_key] = _OrderingBuffer(
                    self.reorder_window_seconds, self.max_hold_seconds,
                    self.max_reorder_buffer)
            session_id = f'explicit:{user_key}:{sid_raw}'
            sort_key = (ts, self._prefix_hash(event), self._arrival_seq)
            if buf.is_late(ts):
                buf.late_count += 1
                late.append(event)
                self._record_late(
                    event, f'explicit:{user_key}:{sid_raw}', buf)
                return rows, late
            buf.add(event, sort_key)
            drained = buf.drain()
            if drained:
                rows.extend(self._emit_explicit_batch(drained, session_id))
            return rows, late

        # Timed non-explicit domain (per user).
        buf = self._timed.get(user_key)
        if buf is None:
            buf = self._timed[user_key] = _OrderingBuffer(
                self.reorder_window_seconds, self.max_hold_seconds,
                self.max_reorder_buffer)
        assign_key = (ts, self._assign_hash(event), self._arrival_seq)
        if buf.is_late(ts):
            buf.late_count += 1
            late.append(event)
            self._record_late(event, f'timed:{user_key}', buf)
            return rows, late
        buf.add(event, assign_key)
        drained = buf.drain()
        if drained:
            rows.extend(self._emit_timed_batch(user_key, drained))
        return rows, late

    def sweep(self, now_monotonic: Optional[float] = None
              ) -> Tuple[List[Dict], List[StreamingEvent]]:
        """Emit events whose max_hold has elapsed (idle-flush)."""
        rows: List[Dict] = []
        late: List[StreamingEvent] = []
        for user_key, buf in list(self._timed.items()):
            drained = buf.drain(now_monotonic)
            if drained:
                rows.extend(self._emit_timed_batch(user_key, drained))
        for (user_key, sid_raw), buf in list(self._explicit.items()):
            drained = buf.drain(now_monotonic)
            if drained:
                rows.extend(self._emit_explicit_batch(
                    drained, f'explicit:{user_key}:{sid_raw}'))
        return rows, late

    def flush_all(self) -> Tuple[List[Dict], List[StreamingEvent]]:
        """Force-emit everything pending (shutdown / replay boundary)."""
        rows: List[Dict] = []
        late: List[StreamingEvent] = []
        for user_key, buf in list(self._timed.items()):
            drained = buf.force_flush()
            if drained:
                rows.extend(self._emit_timed_batch(user_key, drained))
        for (user_key, sid_raw), buf in list(self._explicit.items()):
            drained = buf.force_flush()
            if drained:
                rows.extend(self._emit_explicit_batch(
                    drained, f'explicit:{user_key}:{sid_raw}'))
        return rows, late

    def pending_counts(self) -> Dict[str, int]:
        return {
            'timed_pending': sum(len(b.pending) for b in self._timed.values()),
            'explicit_pending': sum(len(b.pending)
                                    for b in self._explicit.values()),
            'sessions': len(self._sessions),
            'late_events': len(self.late_events),
        }

    # ------------------------------------------------------------------
    # Hash helpers (exact replicas of batch tie-breaks)
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_for(event: StreamingEvent, cols: Tuple[str, ...]) -> str:
        """Content hash identical to ``_content_hash(pd.Series(d), cols)``
        but without constructing a pandas Series per call (hot path)."""
        import hashlib
        import json
        d = event.log_dict()
        present = [c for c in cols if c in d]
        payload = []
        for col in present:
            val = d.get(col)
            if isinstance(val, float) and np.isnan(val):
                val = None
            payload.append(str(val))
        blob = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:16]

    def _assign_hash(self, event: StreamingEvent) -> str:
        return self._hash_for(event, _ASSIGN_HASH_COLS)

    def _prefix_hash(self, event: StreamingEvent) -> str:
        return self._hash_for(event, _PREFIX_HASH_COLS)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    def _state_for(self, session_id: str, user_id: str) -> _SessionState:
        st = self._sessions.get(session_id)
        if st is None:
            st = _SessionState(session_id=session_id,
                               user_id=_norm_id(user_id))
            st.window = deque(maxlen=int(self.seq_config.max_window_events))
            self._sessions[session_id] = st
        return st

    def _dup_hash_for(self, event: StreamingEvent,
                      session_id: str) -> str:
        """Resolved-content duplicate hash (session_id included)."""
        d = dict(event.log_dict())
        d['session_id'] = session_id
        present = [c for c in _DUP_HASH_COLS if c in d]
        return _content_hash(d, present)

    def _note_dup_hashes(self, hashes) -> None:
        for h in hashes:
            if h in self.content_seen:
                self.content_seen.move_to_end(h)
            else:
                self.content_seen[h] = None
            while len(self.content_seen) > self._content_seen_max:
                self.content_seen.popitem(last=False)

    def _dup_flags_for(self, pairs) -> List[bool]:
        """Batch-aligned duplicate flags for a drain set.

        ``pairs`` is a list of ``(event, session_id)`` in emission order.
        An occurrence is flagged when its content hash is observed >= 2
        times causally — counting co-pending members of this drain AND
        previously emitted occurrences — matching the batch contract
        "flag every occurrence of duplicated content" within the causal
        window. Hashes are registered into ``content_seen`` as a side
        effect (bounded LRU).
        """
        hashes = [self._dup_hash_for(ev, sid) for ev, sid in pairs]
        counts = Counter(hashes)
        flags = [counts[h] + (1 if h in self.content_seen else 0) >= 2
                 for h in hashes]
        self._note_dup_hashes(hashes)
        return flags

    def _emit_single(self, event: StreamingEvent, session_id: str,
                     is_dup: Optional[bool] = None) -> Dict:
        """Emit one event whose session/prefix position is final."""
        if is_dup is None:
            # Isolated emission (e.g. missing timestamp): causal 2nd+
            # semantics — the first occurrence cannot be retroactively
            # flagged.
            h = self._dup_hash_for(event, session_id)
            is_dup = h in self.content_seen
            self._note_dup_hashes([h])
        st = self._state_for(session_id, event.user_id)
        return self._advance_and_score(event, st, session_id, is_dup)

    def _emit_explicit_batch(self, drained: List[_Pending],
                             session_id: str) -> List[Dict]:
        """Emit an explicit-session drain set with batch-aligned dup flags."""
        pairs = [(p.event, session_id) for p in drained]
        flags = iter(self._dup_flags_for(pairs))
        st = self._state_for(session_id, drained[0].event.user_id)
        return [self._advance_and_score(p.event, st, session_id, next(flags))
                for p in drained]

    def _emit_timed_batch(self, user_key: str,
                          drained: List[_Pending]) -> List[Dict]:
        """Assign timeout sessions in assign order, then score each
        session's members in prefix order."""
        # Pass 1: session assignment. drained is already in assign sort
        # order (drain() returns the sorted emittable prefix).
        assign = self._user_assign.setdefault(
            user_key, {'current': None, 'prev_ts': None, 'ordinal': 0})
        assigned: List[Tuple[_Pending, str]] = []
        for p in drained:
            event = p.event
            ts = event.event_ts
            cur = assign['current']
            prev_ts = assign['prev_ts']
            if (cur is None or prev_ts is None
                    or (ts - prev_ts).total_seconds() > self.timeout):
                assign['ordinal'] += 1
                start = ts.isoformat() if ts is not None else 'unknown'
                cur = (f'timeout:{user_key}:{start}:'
                       f'{assign["ordinal"]:04d}')
                assign['current'] = cur
            assign['prev_ts'] = ts
            assigned.append((p, cur))

        # Pass 2: per-session prefix ordering + scoring.
        by_session: "OrderedDict[str, List[_Pending]]" = OrderedDict()
        for p, sid in assigned:
            by_session.setdefault(sid, []).append(p)
        rows: List[Dict] = []
        for sid, members in by_session.items():
            # Prefix order: (ts, prefix_tie, arrival). Re-sort members of
            # this session by the prefix tie-break for exact parity.
            members.sort(key=lambda p: (
                p.event.event_ts if p.event.event_ts is not None
                else pd.Timestamp.max,
                self._prefix_hash(p.event),
                p.sort_key[2]))
            user_id = members[0].event.user_id
            st = self._state_for(sid, user_id)
            flags = iter(self._dup_flags_for(
                [(p.event, sid) for p in members]))
            for p in members:
                rows.append(self._advance_and_score(
                    p.event, st, sid, next(flags)))
        return rows

    # ------------------------------------------------------------------
    # Per-event causal computation (prefix features + sequence score)
    # ------------------------------------------------------------------

    def _advance_and_score(self, event: StreamingEvent,
                           st: _SessionState, session_id: str,
                           is_dup: bool) -> Dict:
        ev = event
        ts = ev.event_ts

        # --- prefix aggregates (mirror _add_prefix_features loop) ---------
        st.event_count += 1
        event_type = _norm_token(ev.event_type) or 'unknown'
        ip = _norm_id(ev.source_ip)
        st.event_types.add(event_type)
        st.ips.append(ip)
        st.ip_set.add(ip)
        if st.prev_ip is not None and ip != st.prev_ip:
            st.ip_switches += 1
        st.prev_ip = ip
        st.api_sum += _safe_float(ev.api_requests)
        st.data_sum += _safe_float(ev.data_mb)
        if event_type in self.sensitive:
            st.sensitive_count += 1
        st.after_hours_count += int(_safe_float(ev.after_hours) > 0)
        auth_val = ev.auth_result
        if auth_val is not None and str(auth_val).strip() != '':
            if _is_failure(auth_val):
                st.auth_fail += 1
            elif _is_success(auth_val):
                st.auth_success += 1

        # Bug-compatible with batch: SessionReconstructor._add_prefix_features
        # reads row_d.get('_ts') from itertuples(), which renames '_ts' to a
        # positional field — so cur_ts is ALWAYS None in batch and
        # duration/event_rate/seconds_since_previous/has_timestamp are
        # always NaN/0. Replicated exactly for score parity.
        duration = np.nan
        rate = np.nan
        since_prev = np.nan
        has_ts = 0

        transition = (f'{st.prev_event}->{event_type}'
                      if st.prev_event else '')
        prefix_vals = {
            'session_prefix_event_index': st.event_count,
            'session_prefix_event_count': st.event_count,
            'session_prefix_duration_seconds': duration,
            'session_prefix_event_rate': rate,
            'session_prefix_unique_event_types': len(st.event_types),
            'session_prefix_api_requests': st.api_sum,
            'session_prefix_data_volume_mb': st.data_sum,
            'session_prefix_unique_ips': len(st.ip_set),
            'session_prefix_ip_switch_count': st.ip_switches,
            'session_prefix_sensitive_event_count': st.sensitive_count,
            'session_prefix_after_hours_count': st.after_hours_count,
            'session_prefix_auth_failures': st.auth_fail,
            'session_prefix_auth_successes': st.auth_success,
            'session_prefix_seconds_since_previous': since_prev,
            'session_prefix_event_type': event_type,
            'session_prefix_prev_event_type': st.prev_event,
            'session_prefix_transition_key': transition,
            'session_prefix_has_timestamp': has_ts,
        }
        prev_event_for_seq = st.prev_event
        st.prev_event = event_type

        # --- session deviation (reuse frozen baseline scorer) -------------
        row_for_session = dict(ev.log_dict())
        row_for_session.update(prefix_vals)
        session_score, session_factors, session_missing, session_scope = \
            self.baseline_manager.score_row(row_for_session)

        # --- sequence deviation (mirror SequenceAnalyzer.analyze) --------
        seq_out = self._sequence_score(
            ev, st, ts, event_type, ip, prev_event_for_seq,
            is_dup, rate, st.sensitive_count)

        # --- assemble emitted row -----------------------------------------
        row = dict(ev.log_dict())
        # Envelope fields: identity/lineage only — never detector inputs.
        row['event_id'] = ev.event_id
        row['ingested_at_utc'] = ev.ingested_at_utc
        row['source'] = ev.source
        if ev.replay_of:
            row['replay_of'] = ev.replay_of
        row['session_id'] = session_id
        row['is_duplicate_event'] = is_dup
        for col in SESSION_PREFIX_FEATURES:
            row[col] = prefix_vals.get(col, np.nan)
        for col in _PREFIX_EXTRA_COLS:
            row[col] = prefix_vals.get(col, '')
        row['session_deviation_score'] = session_score
        row['session_factors'] = session_factors
        row['session_score_missing'] = session_missing
        row['session_baseline_scope'] = session_scope
        row.update(seq_out)
        return row

    def _sequence_score(self, ev: StreamingEvent, st: _SessionState,
                        cur_ts, event_type: str, ip: str,
                        prev_event: str, is_dup: bool,
                        prefix_rate, prefix_sensitive_count) -> Dict:
        analyzer = self.sequence_analyzer
        cfg = self.seq_config
        sensitive = self.sensitive

        auth_val = ev.auth_result
        auth_present = auth_val is not None and str(auth_val).strip() != ''
        cur = {
            'event_type': event_type,
            'sensitive': event_type in sensitive,
            'auth_present': auth_present,
            'auth_fail': auth_present and _is_failure(auth_val),
            'auth_success': auth_present and _is_success(auth_val),
            # Bug-compatible with batch: window entries carry 'source_ip',
            # not 'ip', so .get('ip') is always None -> the condition
            # reduces to (ip_switches > 0 and window non-empty).
            'ip_switch': (st.ip_switches > 0 and bool(st.window)
                          and st.window[-1].get('ip') != ip),
            'accounts_accessed': _safe_float(ev.accounts_accessed),
            'courses_accessed': _safe_float(ev.courses_accessed),
            'api_requests': _safe_float(ev.api_requests),
            'data_mb': _safe_float(ev.data_mb),
            'source_ip': ip,
        }

        win_events = []
        if cur_ts is not None and pd.notna(cur_ts):
            for e in st.window:
                ets = e.get('_ts')
                if ets is not None and pd.notna(ets) and \
                        0 <= (cur_ts - ets).total_seconds() \
                        <= cfg.window_seconds:
                    win_events.append(e)
        else:
            win_events = list(st.window)
        if cfg.ignore_duplicates:
            win_events = [e for e in win_events if not e.get('duplicate')]

        model, scope = analyzer._model_for_user(ev.user_id)
        transition_score, trans_missing, trans_ev = \
            analyzer._transition_score(model, str(prev_event or ''),
                                       event_type)
        rate_score, rate_missing = analyzer._rate_score(
            model, _safe_float_or_nan(prefix_rate))
        sens_score, sens_missing = analyzer._sensitive_score(
            model, _safe_float_or_nan(prefix_sensitive_count))
        pattern_score, triggered, pattern_ev = analyzer._eval_definitions(
            win_events, cur)

        components = {}
        factors = []
        evidence = []
        if not trans_missing and np.isfinite(transition_score):
            components['transition_rarity'] = transition_score
            factors.append({'factor': 'Transition Rarity',
                            'value': transition_score,
                            'weight': COMPONENT_WEIGHTS['transition_rarity'],
                            'impact': 'positive' if transition_score > 0
                            else 'none',
                            'detail': trans_ev})
            if transition_score > 0.5:
                evidence.append({
                    'channel': 'sequence', 'evidence_name': 'rare_transition',
                    'observed_value': transition_score,
                    'expected_baseline_or_threshold':
                        'user/global transition model',
                    'contribution_or_severity': transition_score,
                    'availability': 'available',
                    'explanation': f"Unusual transition "
                                   f"{trans_ev.get('transition', '')}"})
        else:
            factors.append({'factor': 'Transition Rarity', 'value': None,
                            'impact': 'missing', 'missing': True})

        if not rate_missing and np.isfinite(rate_score):
            components['event_rate_deviation'] = rate_score
            factors.append({'factor': 'Event Rate Deviation',
                            'value': rate_score,
                            'weight': COMPONENT_WEIGHTS[
                                'event_rate_deviation'],
                            'impact': 'positive' if rate_score > 0
                            else 'none'})
        else:
            factors.append({'factor': 'Event Rate Deviation', 'value': None,
                            'impact': 'missing', 'missing': True})

        if not sens_missing and np.isfinite(sens_score):
            components['sensitive_sequence_deviation'] = sens_score
            factors.append({'factor': 'Sensitive Sequence Deviation',
                            'value': sens_score,
                            'weight': COMPONENT_WEIGHTS[
                                'sensitive_sequence_deviation'],
                            'impact': 'positive' if sens_score > 0
                            else 'none'})
        else:
            factors.append({'factor': 'Sensitive Sequence Deviation',
                            'value': None, 'impact': 'missing',
                            'missing': True})

        components['pattern_match'] = pattern_score
        if pattern_score > 0:
            factors.append({'factor': 'Sequence Pattern Match',
                            'value': pattern_score,
                            'weight': COMPONENT_WEIGHTS['pattern_match'],
                            'impact': 'positive'})
            for pe in pattern_ev:
                evidence.append({
                    'channel': 'sequence', 'evidence_name': pe['pattern'],
                    'observed_value': pe['score'],
                    'expected_baseline_or_threshold': pe['kind'],
                    'contribution_or_severity': pe['score'],
                    'availability': 'available',
                    'explanation': pe.get('detail', pe['pattern'])})
        else:
            factors.append({'factor': 'Sequence Pattern Match', 'value': 0.0,
                            'weight': COMPONENT_WEIGHTS['pattern_match'],
                            'impact': 'none'})

        weights = cfg.weights or COMPONENT_WEIGHTS
        total_w = sum(weights.get(k, 0.0) for k in components)
        if total_w <= 0:
            score = float('nan')
            miss = True
        else:
            score = float(np.clip(
                sum(components[k] * weights.get(k, 0.0) for k in components)
                / total_w, 0.0, 1.0))
            miss = False

        # Append current event to causal window for subsequent events.
        st.window.append({
            **cur,
            '_ts': cur_ts if cur_ts is not None and pd.notna(cur_ts)
            else None,
            'duplicate': bool(is_dup),
        })

        return {
            'sequence_deviation_score': score,
            'sequence_score_missing': miss,
            'sequence_factors': factors,
            'sequence_evidence': evidence,
            'sequence_triggered_patterns': triggered,
            'sequence_baseline_scope': scope if not miss else 'insufficient',
        }


def _safe_float_or_nan(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return v if np.isfinite(v) else float('nan')
