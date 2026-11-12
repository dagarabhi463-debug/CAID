"""Milestone 2.0.1 causal-correctness and determinism tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session_reconstruction import SessionReconstructor, SessionConfig
from core.sequence_detection import SequenceAnalyzer


def ev(ts, user='u1', session='', et='course_view', ip='192.0.2.10',
       auth='', api=50, data=1.0, courses=2, accounts=2, **kw):
    row = {
        'timestamp': ts, 'user_id': user, 'role': 'student',
        'source_ip': ip, 'event_type': et,
        'courses_accessed': courses, 'accounts_accessed': accounts,
        'api_requests': api, 'data_mb': data, 'token_age_minutes': 30,
        'after_hours': 0, 'label': 'normal', 'session_id': session,
        'auth_result': auth,
    }
    row.update(kw)
    return row


def frame(rows):
    return pd.DataFrame(rows)


class TestSessionCausality:
    def test_prefix_excludes_future_events(self):
        df = frame([
            ev('2026-01-05 08:00:00', session='s1', api=10),
            ev('2026-01-05 08:01:00', session='s1', api=100),
            ev('2026-01-05 08:02:00', session='s1', api=200),
        ])
        out = SessionReconstructor().reconstruct(df)
        assert out['session_prefix_api_requests'].tolist() == [10.0, 110.0, 310.0]
        assert out['session_prefix_event_count'].tolist() == [1, 2, 3]

    def test_same_timestamp_deterministic_under_shuffle(self):
        df = frame([
            ev('2026-01-05 08:00:00', session='s1', et='aaa_event', ip='192.0.2.11'),
            ev('2026-01-05 08:00:00', session='s1', et='bbb_event', ip='192.0.2.12'),
        ])
        recon = SessionReconstructor()
        a = recon.reconstruct(df)
        b = recon.reconstruct(df.iloc[::-1].reset_index(drop=True))
        key = ['timestamp', 'user_id', 'event_type', 'source_ip']
        m = a[key + ['session_prefix_event_index']].merge(
            b[key + ['session_prefix_event_index']], on=key, suffixes=('_a', '_b'))
        assert (m['session_prefix_event_index_a'] == m['session_prefix_event_index_b']).all()
        assert sorted(m['session_prefix_event_index_a']) == [1, 2]

    def test_shuffled_input_same_scores_for_unique_timestamps(self):
        rows = [ev(f'2026-01-05 08:{i:02d}:00', session=f's{i%2}') for i in range(8)]
        df = frame(rows)
        shuffled = df.sample(frac=1.0, random_state=7).reset_index(drop=True)
        recon = SessionReconstructor()
        a = recon.reconstruct(df)
        b = recon.reconstruct(shuffled)
        cols = ['session_id', 'session_prefix_event_count',
                'session_prefix_api_requests', 'session_prefix_event_type']
        key = ['timestamp', 'user_id', 'event_type', 'source_ip']
        m = a[key + cols].merge(b[key + cols], on=key, suffixes=('_a', '_b'))
        for c in cols:
            assert (m[f'{c}_a'].astype(str) == m[f'{c}_b'].astype(str)).all()

    def test_timeout_boundary(self):
        cfg = SessionConfig(inactivity_timeout_seconds=60)
        df = frame([
            ev('2026-01-05 08:00:00', session=''),
            ev('2026-01-05 08:01:00', session=''),   # exactly 60s -> same
            ev('2026-01-05 08:02:01', session=''),   # 61s after previous -> new
        ])
        out = SessionReconstructor(cfg).reconstruct(df)
        assert out['session_id'].nunique() == 2
        assert out.loc[0, 'session_id'] == out.loc[1, 'session_id']
        assert out.loc[2, 'session_id'] != out.loc[1, 'session_id']

    def test_missing_timestamp_isolated_no_transition(self):
        df = frame([
            ev('2026-01-05 08:00:00', et='course_view'),
            ev('', et='file_view'),
        ])
        out = SessionReconstructor().reconstruct(df)
        miss = out[out['session_timestamp_missing']]
        assert len(miss) == 1
        assert str(miss.iloc[0]['session_id']).startswith('missing_ts:')
        assert miss.iloc[0]['session_prefix_event_count'] == 1
        assert miss.iloc[0]['session_prefix_prev_event_type'] in ('', None)

    def test_duplicate_events_flagged_and_preserved(self):
        row = ev('2026-01-05 08:00:00', session='s1')
        df = frame([row, dict(row)])
        out = SessionReconstructor().reconstruct(df)
        assert len(out) == 2
        assert out['is_duplicate_event'].all()

    def test_multiple_explicit_sessions_reset_prefix(self):
        df = frame([
            ev('2026-01-05 08:00:00', session='s1'),
            ev('2026-01-05 08:01:00', session='s1'),
            ev('2026-01-05 08:05:00', session='s2'),
        ])
        out = SessionReconstructor().reconstruct(df)
        assert out['session_prefix_event_count'].tolist() == [1, 2, 1]

    def test_late_arriving_event_orders_by_timestamp(self):
        df = frame([
            ev('2026-01-05 08:00:00', session='s1', et='first'),
            ev('2026-01-05 08:02:00', session='s1', et='third'),
            ev('2026-01-05 08:01:00', session='s1', et='second'),  # late input
        ])
        out = SessionReconstructor().reconstruct(df)
        by_ts = out.sort_values('timestamp')
        assert by_ts['session_prefix_event_type'].tolist() == ['first', 'second', 'third']
        assert by_ts['session_prefix_event_count'].tolist() == [1, 2, 3]


class TestSequenceCausality:
    def _train(self):
        rows = []
        for s in range(6):
            base = pd.Timestamp('2026-01-05 08:00:00') + pd.Timedelta(hours=s)
            for i, et in enumerate(['login', 'course_view', 'file_view']):
                rows.append(ev((base + pd.Timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
                               session=f's{s}', et=et, auth='success' if et == 'login' else ''))
        return frame(rows)

    def test_truncation_does_not_change_earlier_scores(self):
        train = self._train()
        analyzer = SequenceAnalyzer().fit(train)
        rows = [
            ev('2026-01-06 09:00:00', user='uX', session='sx', et='login', auth='failure'),
            ev('2026-01-06 09:01:00', user='uX', session='sx', et='login', auth='failure'),
            ev('2026-01-06 09:02:00', user='uX', session='sx', et='login', auth='success'),
            ev('2026-01-06 09:03:00', user='uX', session='sx', et='bulk_export'),
        ]
        df = frame(rows)
        full = analyzer.analyze(df)
        trunc = analyzer.analyze(df.iloc[:3].copy())
        cols = ['sequence_deviation_score', 'sequence_triggered_patterns',
                'sequence_baseline_scope']
        for c in cols:
            a = full.iloc[:3][c].astype(str).tolist()
            b = trunc[c].astype(str).tolist()
            assert a == b, c
        # Later suspicious event should not be required for earlier causal scores.
        assert full.iloc[0]['sequence_deviation_score'] is not None
