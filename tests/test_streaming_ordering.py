"""Ordering-policy tests — Milestone 2.5.1.

Covers the documented ordering contract of IncrementalSessionTracker:
per-domain reorder buffers, emitted-frontier late detection, bounded
latency, and DLQ traceability (reason + event_id + frontier detail).
"""

import pandas as pd
import pytest

from core.streaming.event_schema import StreamingEvent, validate_event
from core.streaming.session_state import IncrementalSessionTracker


def _ev(user, ts, event_type='login', session_id='', **kw):
    raw = {
        'timestamp': ts,
        'user_id': user,
        'role': 'student',
        'source_ip': '192.0.2.1',
        'event_type': event_type,
        'courses_accessed': 1,
        'accounts_accessed': 1,
        'api_requests': 1,
        'data_mb': 1.0,
        'token_age_minutes': 1,
        'after_hours': 0,
        'session_id': session_id,
    }
    raw.update(kw)
    ev, reason = validate_event(raw, source='ordering-test')
    assert ev is not None, reason
    return ev


@pytest.fixture(scope='module')
def tracker():
    from core.streaming.event_processor import StreamingDetectionProcessor
    art = StreamingDetectionProcessor.load_artifacts(
        tracker_kwargs={'reorder_window_seconds': 0.25,
                        'max_hold_seconds': 0.5})
    return art.tracker


class TestLateDetection:
    def test_straggler_before_emitted_frontier_is_late(self, tracker):
        # Immediate-emit config: window=0, hold=0 -> every ingest drains.
        tr = type(tracker)(
            tracker.baseline_manager, tracker.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        base = pd.Timestamp('2024-03-01T10:00:00')
        for i, minute in enumerate((0, 10, 20)):
            ev = _ev('u_late', (base + pd.Timedelta(minutes=minute))
                     .strftime('%Y-%m-%dT%H:%M:%S'),
                     event_type='course_view')
            rows, late = tr.ingest(ev)
            assert not late
        # Straggler with ts between first and last emitted -> late.
        ev = _ev('u_late', (base + pd.Timedelta(minutes=5))
                 .strftime('%Y-%m-%dT%H:%M:%S'), event_type='file_view')
        rows, late = tr.ingest(ev)
        assert rows == [] and len(late) == 1
        assert late[0].event_id == ev.event_id
        detail = tr.late_details[ev.event_id]
        assert detail['ordering_domain'] == 'timed:u_late'
        assert detail['emitted_frontier_ts'] is not None
        assert detail['event_ts'] < detail['emitted_frontier_ts']

    def test_late_event_never_emitted_or_resurrected(self, tracker):
        tr = type(tracker)(
            tracker.baseline_manager, tracker.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        base = pd.Timestamp('2024-03-01T11:00:00')
        evs = [_ev('u_late2', (base + pd.Timedelta(minutes=m))
                    .strftime('%Y-%m-%dT%H:%M:%S'), event_type='course_view')
               for m in (0, 30)]
        for ev in evs:
            tr.ingest(ev)
        straggler = _ev('u_late2', (base + pd.Timedelta(minutes=10))
                        .strftime('%Y-%m-%dT%H:%M:%S'),
                        event_type='file_view')
        _rows, late = tr.ingest(straggler)
        rows, _ = tr.flush_all()
        emitted_ids = {r['event_id'] for r in rows}
        assert straggler.event_id not in emitted_ids
        assert straggler.event_id in tr.late_details

    def test_explicit_session_domain_independent(self, tracker):
        tr = type(tracker)(
            tracker.baseline_manager, tracker.sequence_analyzer,
            reorder_window_seconds=0.0, max_hold_seconds=0.0)
        base = pd.Timestamp('2024-03-01T12:00:00')
        for m in (0, 30):
            ev = _ev('u_exp', (base + pd.Timedelta(minutes=m))
                     .strftime('%Y-%m-%dT%H:%M:%S'),
                     event_type='course_view', session_id='sessA')
            tr.ingest(ev)
        # Straggler in the SAME explicit session -> late.
        ev = _ev('u_exp', (base + pd.Timedelta(minutes=10))
                 .strftime('%Y-%m-%dT%H:%M:%S'),
                 event_type='file_upload', session_id='sessA')
        _rows, late = tr.ingest(ev)
        assert len(late) == 1
        assert tr.late_details[ev.event_id]['ordering_domain'] == \
            'explicit:u_exp:sessA'
        # Same user, DIFFERENT session -> different domain, not late.
        ev2 = _ev('u_exp', (base + pd.Timedelta(minutes=5))
                  .strftime('%Y-%m-%dT%H:%M:%S'),
                  event_type='quiz_view', session_id='sessB')
        rows2, late2 = tr.ingest(ev2)
        assert not late2


class TestReorderWindow:
    def test_in_window_reorder_preserves_ts_order(self, tracker):
        tr = type(tracker)(
            tracker.baseline_manager, tracker.sequence_analyzer,
            reorder_window_seconds=120.0, max_hold_seconds=0.5)
        base = pd.Timestamp('2024-03-01T13:00:00')
        e1 = _ev('u_re', (base).strftime('%Y-%m-%dT%H:%M:%S'),
                 event_type='login')
        e2 = _ev('u_re', (base + pd.Timedelta(seconds=200))
                 .strftime('%Y-%m-%dT%H:%M:%S'), event_type='logout')
        e3 = _ev('u_re', (base + pd.Timedelta(seconds=100))
                 .strftime('%Y-%m-%dT%H:%M:%S'), event_type='page_view')
        rows = []
        for ev in (e1, e2, e3):
            r, late = tr.ingest(ev)
            assert not late
            rows.extend(r)
        r, _ = tr.flush_all()
        rows.extend(r)
        assert [x['event_id'] for x in rows] == \
            [e1.event_id, e3.event_id, e2.event_id]

    def test_monotonic_feed_produces_no_late_events(self, tracker):
        tr = type(tracker)(
            tracker.baseline_manager, tracker.sequence_analyzer,
            reorder_window_seconds=0.25, max_hold_seconds=0.5)
        base = pd.Timestamp('2024-03-01T14:00:00')
        late_total = 0
        rows_total = 0
        for i in range(50):
            ev = _ev(f'u_mono_{i % 5}',
                     (base + pd.Timedelta(seconds=i * 7))
                     .strftime('%Y-%m-%dT%H:%M:%S'),
                     event_type='login' if i % 3 else 'logout')
            rows, late = tr.ingest(ev)
            rows_total += len(rows)
            late_total += len(late)
        rows, _ = tr.flush_all()
        rows_total += len(rows)
        assert late_total == 0
        assert rows_total == 50
        assert not tr.late_details
