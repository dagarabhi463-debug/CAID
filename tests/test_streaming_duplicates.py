"""Duplicate-semantics tests — Milestone 2.5.1.

Contract (aligned to batch ``is_duplicate_event`` within causal bounds):
an event is flagged when its resolved-content hash is observed >= 2
times causally — counting co-pending drain-batch members AND previously
emitted occurrences. A duplicate separated from its first occurrence by
more than the reorder window flags only the later occurrence(s).
"""

import pandas as pd
import pytest

from core.streaming.event_schema import validate_event
from core.streaming.event_checkpoint import StreamCheckpoint


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
    ev, reason = validate_event(raw, source='dup-test')
    assert ev is not None, reason
    return ev


def _dup_pair(user='u_dup', ts='2024-04-01T10:00:00', session_id='s1'):
    """Two byte-identical raw events -> identical deterministic event_id."""
    return _ev(user, ts, event_type='course_view', session_id=session_id,
               source_ip='198.51.100.7', api_requests=3)


@pytest.fixture(scope='module')
def proto_tracker():
    from core.streaming.event_processor import StreamingDetectionProcessor
    art = StreamingDetectionProcessor.load_artifacts(
        tracker_kwargs={'reorder_window_seconds': 1.0,
                        'max_hold_seconds': 1.0})
    return art.tracker


def _mk(tracker, **kw):
    return type(tracker)(
        tracker.baseline_manager, tracker.sequence_analyzer, **kw)


def _flags(rows):
    return {r['event_id']: r['is_duplicate_event'] for r in rows}


class TestDuplicateContract:
    def test_single_occurrence_not_duplicate(self, proto_tracker):
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        ev = _ev('u_single', '2024-04-01T10:00:00')
        rows, _ = tr.ingest(ev)
        assert not rows[0]['is_duplicate_event']

    def test_two_identical_co_pending_both_flagged(self, proto_tracker):
        # Large window+hold: both duplicates stay pending until flush,
        # landing in the same drain batch -> BOTH flagged (batch parity).
        tr = _mk(proto_tracker, reorder_window_seconds=3600.0,
                 max_hold_seconds=3600.0)
        e1 = _dup_pair()
        e2 = _dup_pair()  # identical content AND identical event_id
        for ev in (e1, e2):
            tr.ingest(ev)
        rows, _ = tr.flush_all()
        flags = _flags(rows)
        assert len(rows) == 2
        assert all(flags[r['event_id']] for r in rows)

    def test_three_identical_all_flagged(self, proto_tracker):
        tr = _mk(proto_tracker, reorder_window_seconds=3600.0,
                 max_hold_seconds=3600.0)
        for _ in range(3):
            tr.ingest(_dup_pair())
        rows, _ = tr.flush_all()
        assert len(rows) == 3
        assert all(r['is_duplicate_event'] for r in rows)

    def test_timestamp_difference_breaks_duplicate(self, proto_tracker):
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        e1 = _dup_pair()
        rows1, _ = tr.ingest(e1)
        assert not rows1[0]['is_duplicate_event']
        # Different timestamp -> different resolved content -> not a dup.
        e2 = _dup_pair(ts='2024-04-01T10:00:01')
        rows2, _ = tr.ingest(e2)
        assert not rows2[0]['is_duplicate_event']

    def test_true_duplicate_after_first_emitted_flags_second(
            self, proto_tracker):
        # First occurrence committed; an identical later arrival is the
        # documented residual case -> only the later occurrence flagged.
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        e1 = _dup_pair()
        rows1, _ = tr.ingest(e1)
        assert not rows1[0]['is_duplicate_event']
        e2 = _dup_pair()
        rows2, _ = tr.ingest(e2)
        assert rows2[0]['is_duplicate_event'] is True

    def test_duplicate_after_checkpoint_restore(self, proto_tracker, tmp_path):
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        e1 = _dup_pair()
        tr.ingest(e1)
        cp = StreamCheckpoint(str(tmp_path / 'ckpt.json'))
        cp.save(tracker=tr)

        tr2 = _mk(proto_tracker, reorder_window_seconds=0.0,
                  max_hold_seconds=0.0)
        assert cp.restore_tracker(tr2)
        e2 = _dup_pair()
        rows, _ = tr2.ingest(e2)
        assert rows[0]['is_duplicate_event'] is True

    def test_replay_of_preserves_duplicate_flagging(self, proto_tracker):
        # replay_of lineage does not alter the content hash: a replayed
        # duplicate of already-seen content is still flagged.
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        e1 = _dup_pair()
        tr.ingest(e1)
        raw = e1.to_dict()
        raw['replay_of'] = e1.event_id
        ev2, reason = validate_event(raw, source='dup-test',
                                     replay_of=e1.event_id)
        assert ev2 is not None, reason
        rows, _ = tr.ingest(ev2)
        assert rows[0]['is_duplicate_event'] is True
        assert rows[0]['replay_of'] == e1.event_id

    def test_distinct_content_not_duplicate(self, proto_tracker):
        tr = _mk(proto_tracker, reorder_window_seconds=0.0,
                 max_hold_seconds=0.0)
        e1 = _ev('u_dist', '2024-04-01T10:00:00', event_type='login')
        e2 = _ev('u_dist', '2024-04-01T10:00:01', event_type='logout')
        rows = []
        for ev in (e1, e2):
            r, _ = tr.ingest(ev)
            rows.extend(r)
        assert all(not r['is_duplicate_event'] for r in rows)
