"""Milestone 2.1 cross-session threat correlation tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cross_session_correlation import (
    CorrelationConfig,
    CrossSessionCorrelator,
)


def _events(rows):
    """Build an events DataFrame from compact row tuples."""
    base = {
        'timestamp': '2026-02-01 10:00:00',
        'user_id': 'u1',
        'role': 'student',
        'source_ip': '198.51.100.1',
        'event_type': 'course_view',
        'courses_accessed': 1,
        'accounts_accessed': 1,
        'api_requests': 5,
        'data_mb': 1.0,
        'token_age_minutes': 30,
        'after_hours': 0,
        'label': 'normal',
        'session_id': 's1',
        'auth_result': 'success',
        'combined_score': 0.1,
        'sequence_deviation': 0.1,
        'session_deviation': 0.1,
        'risk_level': 'Low',
        'sequence_triggered_patterns': [],
    }
    out = []
    for overrides in rows:
        r = dict(base)
        r.update(overrides)
        out.append(r)
    return pd.DataFrame(out)


def _session_events(user, session, n=3, start='2026-02-01 08:00:00',
                    event_type='course_view', ip='198.51.100.1',
                    combined=0.9, patterns=None, sensitive=False):
    rows = []
    for i in range(n):
        rows.append({
            'user_id': user,
            'session_id': session,
            'timestamp': f'{start[:10]} {8 + i // 60:02d}:{i % 60:02d}:00',
            'source_ip': ip,
            'event_type': 'grade_download' if sensitive and i == 0 else event_type,
            'combined_score': combined,
            'risk_level': 'High' if combined >= 0.5 else 'Low',
            'sequence_triggered_patterns': patterns or [],
        })
    return rows


class TestRepeatedPatternsAcrossSessions:
    def test_same_pattern_in_two_sessions(self):
        rows = (_session_events('u1', 's1', patterns=['p1']) +
                _session_events('u1', 's2', start='2026-02-02 08:00:00',
                                patterns=['p1']))
        result = CrossSessionCorrelator().correlate(_events(rows))
        corr = result['correlations']
        hits = corr[corr['correlation_type'] == 'repeated_pattern_across_sessions']
        assert len(hits) == 1
        assert hits.iloc[0]['entity_a'] == 'u1'
        assert set(hits.iloc[0]['sessions']) == {'s1', 's2'}
        ev = hits.iloc[0]['evidence'][0]
        assert ev['signal'] == 'repeated_sequence_pattern'
        assert 's1' in ev['explanation'] and 's2' in ev['explanation']

    def test_elevated_deviation_without_named_pattern(self):
        rows = (_session_events('u2', 's1', patterns=[]) +
                _session_events('u2', 's2', start='2026-02-02 08:00:00',
                                patterns=[]))
        result = CrossSessionCorrelator().correlate(_events(rows))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'repeated_pattern_across_sessions']
        assert len(hits) == 1
        assert hits.iloc[0]['evidence'][0]['signal'] == 'repeated_elevated_deviation'

    def test_single_session_does_not_correlate(self):
        rows = _session_events('u3', 's1', patterns=['p1'])
        result = CrossSessionCorrelator().correlate(_events(rows))
        assert len(result['correlations']) == 0 or \
            'repeated_pattern_across_sessions' not in \
            set(result['correlations']['correlation_type'])


class TestSimilarAccountBehavior:
    def _pair_rows(self):
        rows = []
        for uid in ('alice', 'bob'):
            for s in ('s1', 's2'):
                for i, et in enumerate(['grade_download', 'bulk_download',
                                        'course_export', 'course_view']):
                    rows.append({
                        'user_id': uid,
                        'session_id': f'{uid}_{s}',
                        'timestamp': f'2026-02-0{s[-1]} 0{i}:00:00',
                        'source_ip': f'198.51.100.{10 + (0 if uid == "alice" else 1)}',
                        'event_type': et,
                        'combined_score': 0.8,
                        'risk_level': 'High',
                        'sequence_triggered_patterns': ['sensitive_burst'],
                    })
        return rows

    def test_similar_suspicious_accounts_correlate(self):
        result = CrossSessionCorrelator().correlate(_events(self._pair_rows()))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'similar_account_behavior']
        assert len(hits) >= 1
        assert set(hits.iloc[0]['entities']) == {'alice', 'bob'}

    def test_dissimilar_accounts_do_not_correlate(self):
        rows = []
        for uid, ets in (('alice', ['grade_download'] * 6),
                         ('carol', ['discussion_post', 'page_view',
                                    'settings_view'] * 2)):
            for i, et in enumerate(ets):
                rows.append({
                    'user_id': uid,
                    'session_id': f'{uid}_s1',
                    'timestamp': f'2026-02-01 0{i}:00:00',
                    'event_type': et,
                    'combined_score': 0.8,
                    'risk_level': 'High',
                    'sequence_triggered_patterns': ['sensitive_burst'],
                })
        result = CrossSessionCorrelator().correlate(_events(rows))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'similar_account_behavior']
        assert len(hits) == 0


class TestRepeatedSensitiveOperations:
    def test_sensitive_ops_across_days(self):
        rows = []
        for day in ('01', '02', '03'):
            rows += _session_events('u9', f's_{day}', n=2, sensitive=True,
                                    start=f'2026-02-{day} 08:00:00',
                                    combined=0.2)
        result = CrossSessionCorrelator().correlate(_events(rows))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'repeated_sensitive_operations']
        assert len(hits) == 1
        assert hits.iloc[0]['entity_a'] == 'u9'
        assert '2026-02-01' in hits.iloc[0]['evidence'][0]['explanation']

    def test_single_day_does_not_correlate(self):
        rows = _session_events('u10', 's1', n=4, sensitive=True,
                               start='2026-02-01 08:00:00', combined=0.2)
        result = CrossSessionCorrelator().correlate(_events(rows))
        hits = result['correlations']
        assert 'repeated_sensitive_operations' not in \
            set(hits['correlation_type']) if len(hits) else True


class TestLongTermBehaviorChange:
    def test_early_vs_late_shift(self):
        cfg = CorrelationConfig(change_threshold=0.3)
        rows = []
        # Early window: calm behavior.
        for i in range(6):
            rows.append({
                'user_id': 'drift', 'session_id': 'early',
                'timestamp': f'2026-01-0{i + 1} 09:00:00',
                'event_type': 'course_view', 'combined_score': 0.1,
            })
        # Late window: sensitive ops, failures, after-hours, more event types.
        for i, et in enumerate(['grade_download', 'bulk_download',
                                'course_export', 'quiz_submitted',
                                'assignment_submit', 'grade_download']):
            rows.append({
                'user_id': 'drift', 'session_id': 'late',
                'timestamp': f'2026-02-0{i + 1} 23:00:00',
                'event_type': et, 'after_hours': 1,
                'auth_result': 'failure' if i % 2 else 'success',
                'combined_score': 0.6,
            })
        result = CrossSessionCorrelator(cfg).correlate(_events(rows))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'long_term_behavior_change']
        assert len(hits) == 1
        assert hits.iloc[0]['entity_a'] == 'drift'

    def test_stable_user_does_not_correlate(self):
        rows = []
        for i in range(12):
            rows.append({
                'user_id': 'steady', 'session_id': 's1' if i < 6 else 's2',
                'timestamp': f'2026-01-{i + 1:02d} 09:00:00',
                'event_type': 'course_view', 'combined_score': 0.1,
            })
        result = CrossSessionCorrelator(
            CorrelationConfig(change_threshold=0.3)).correlate(_events(rows))
        hits = result['correlations']
        assert 'long_term_behavior_change' not in \
            set(hits['correlation_type']) if len(hits) else True


class TestSharedIpActivity:
    def _shared_ip_rows(self, uid_a='sh1', uid_b='sh2'):
        rows = (_session_events(uid_a, 'a1', n=3, ip='203.0.113.50',
                                event_type='course_view', combined=0.9,
                                patterns=[]) +
                _session_events(uid_b, 'b1', n=3, ip='203.0.113.50',
                                event_type='settings_view', combined=0.9,
                                patterns=[]))
        return rows

    def test_shared_ip_alone_is_not_proof(self):
        result = CrossSessionCorrelator().correlate(_events(self._shared_ip_rows()))
        hits = result['correlations']
        hits = hits[hits['correlation_type'] == 'shared_ip_activity']
        assert len(hits) == 1
        assert hits.iloc[0]['shared_ip_only'] is True or \
            bool(hits.iloc[0]['shared_ip_only'])
        assert hits.iloc[0]['correlation_score'] <= 0.4
        assert '203.0.113.50' in str(hits.iloc[0]['evidence'])

    def test_shared_ip_escalates_with_corroboration(self):
        # Same suspicious profile + same pattern on shared IP → corroborated.
        rows = []
        for uid in ('ca', 'cb'):
            for s in ('s1', 's2'):
                for i, et in enumerate(['grade_download', 'bulk_download',
                                        'course_export', 'course_view']):
                    rows.append({
                        'user_id': uid, 'session_id': f'{uid}_{s}',
                        'timestamp': f'2026-02-0{i % 2 + 1} 0{i}:00:00',
                        'source_ip': '203.0.113.99',
                        'event_type': et, 'combined_score': 0.8,
                        'risk_level': 'High',
                        'sequence_triggered_patterns': ['sensitive_burst'],
                    })
        result = CrossSessionCorrelator().correlate(_events(rows))
        hits = result['correlations']
        ip_hits = hits[hits['correlation_type'] == 'shared_ip_activity']
        assert len(ip_hits) == 1
        assert not bool(ip_hits.iloc[0]['shared_ip_only'])
        assert ip_hits.iloc[0]['correlation_score'] > 0.4


class TestRobustness:
    def test_empty_dataframe(self):
        result = CrossSessionCorrelator().correlate(pd.DataFrame())
        assert len(result['correlations']) == 0
        assert 'correlation_id' in result['correlations'].columns

    def test_deterministic_ids(self):
        rows = (_session_events('u1', 's1', patterns=['p1']) +
                _session_events('u1', 's2', start='2026-02-02 08:00:00',
                                patterns=['p1']))
        df = _events(rows)
        a = CrossSessionCorrelator().correlate(df)['correlations']
        b = CrossSessionCorrelator().correlate(df)['correlations']
        assert a['correlation_id'].tolist() == b['correlation_id'].tolist()

    def test_missing_session_column_reconstructs(self):
        rows = _session_events('u1', 's1', patterns=['p1']) + \
               _session_events('u1', 's2', start='2026-02-02 08:00:00',
                               patterns=['p1'])
        df = _events(rows).drop(columns=['session_id'])
        result = CrossSessionCorrelator().correlate(df)
        hits = result['correlations']
        assert 'repeated_pattern_across_sessions' in \
            set(hits['correlation_type'])

    def test_no_suspicious_activity_no_findings(self):
        rows = []
        for uid in ('n1', 'n2'):
            rows += _session_events(uid, f'{uid}_s1', n=3, ip='198.51.100.7',
                                    event_type='course_view', combined=0.05,
                                    patterns=[])
        result = CrossSessionCorrelator().correlate(_events(rows))
        corr = result['correlations']
        # Only possible finding would be shared_ip_activity, which requires
        # at least one suspicious account — expect zero findings.
        assert len(corr) == 0
