"""Tests for core/streaming/event_schema.py — Milestone 2.5."""

import pytest

from core.streaming.event_schema import (
    BoundedDedupLRU, StreamingEvent, compute_event_id, validate_event)


def _raw(**over):
    base = {
        'timestamp': '2024-01-28 05:58:04',
        'user_id': 'user_1',
        'role': 'student',
        'source_ip': '192.0.2.1',
        'event_type': 'login',
        'courses_accessed': 3,
        'accounts_accessed': 1,
        'api_requests': 12,
        'data_mb': 1.5,
        'token_age_minutes': 30,
        'after_hours': 0,
        'session_id': '',
        'auth_result': 'success',
    }
    base.update(over)
    return base


class TestValidateEvent:
    def test_valid_event_normalizes(self):
        ev, reason = validate_event(_raw(), source='test')
        assert reason is None and ev is not None
        assert ev.event_id and len(ev.event_id) == 16
        assert ev.user_id == 'user_1'
        assert ev.event_type == 'login'
        assert ev.ordering_key[0] is not None

    def test_event_id_deterministic(self):
        e1, _ = validate_event(_raw(), source='s')
        e2, _ = validate_event(_raw(), source='s')
        assert e1.event_id == e2.event_id
        e3, _ = validate_event(_raw(user_id='user_2'), source='s')
        assert e3.event_id != e1.event_id

    def test_label_rejected_in_production(self):
        ev, reason = validate_event(_raw(label='normal'), source='s')
        assert ev is None and reason == 'label_present_in_production_stream'
        ev2, reason2 = validate_event(_raw(label='normal'), source='s',
                                      allow_labels=True)
        assert ev2 is not None and reason2 is None

    def test_missing_required_field_rejected(self):
        raw = _raw()
        del raw['user_id']
        ev, reason = validate_event(raw)
        assert ev is None and 'missing_fields' in reason

    def test_bad_timestamp_rejected(self):
        ev, reason = validate_event(_raw(timestamp='not-a-date'))
        assert ev is None and reason == 'invalid_timestamp'

    def test_unknown_event_type_rejected(self):
        ev, reason = validate_event(_raw(event_type='not_a_real_event'))
        assert ev is None and reason.startswith('unknown_event_type')

    def test_event_type_canonicalized(self):
        ev, _ = validate_event(_raw(event_type='LOGIN'))
        assert ev is not None and ev.event_type == 'login'

    def test_invalid_numeric_rejected(self):
        ev, reason = validate_event(_raw(api_requests='abc'))
        assert ev is None and reason == 'invalid_numeric:api_requests'

    def test_missing_optional_field_defaults(self):
        raw = _raw()
        del raw['session_id']
        ev, reason = validate_event(raw)
        assert ev is not None and ev.session_id == ''

    def test_missing_required_numeric_rejected(self):
        raw = _raw()
        del raw['data_mb']
        ev, reason = validate_event(raw)
        assert ev is None and 'missing_fields' in reason

    def test_non_dict_rejected(self):
        ev, reason = validate_event(['not', 'a', 'dict'])
        assert ev is None and reason == 'not_a_dict'


class TestDedupLRU:
    def test_duplicate_detected(self):
        d = BoundedDedupLRU(maxsize=10)
        assert d.check_and_add('a') is False
        assert d.check_and_add('a') is True
        assert d.check_and_add('b') is False

    def test_bounded_eviction(self):
        d = BoundedDedupLRU(maxsize=3)
        for i in range(5):
            d.check_and_add(f'id{i}')
        assert len(d) == 3
        assert d.check_and_add('id0') is False  # evicted


class TestSerialization:
    def test_roundtrip(self):
        ev, _ = validate_event(_raw(), source='s')
        ev2 = StreamingEvent.from_dict(ev.to_dict())
        assert ev2.event_id == ev.event_id
        assert ev2.user_id == ev.user_id
        assert ev2.event_ts == ev.event_ts
