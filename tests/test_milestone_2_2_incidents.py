"""Milestone 2.2 threat investigation & incident report tests."""

import json
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.incident_manager import (
    IncidentConfig,
    IncidentManager,
    IncidentStore,
)
from core.incident_report import generate_incident_csv, generate_incident_pdf


def _finding(fid, ftype='repeated_pattern_across_sessions', entities=None,
             sessions=None, score=0.7, shared_ip_only=False,
             ips=None, note=''):
    evidence = []
    for ip in ips or []:
        evidence.append({
            'signal': 'shared_source_ip', 'observed_value': ip,
            'expected_baseline_or_threshold': '>= 2 users',
            'explanation': f'shared ip {ip}',
        })
    if not evidence:
        evidence.append({
            'signal': 'repeated_sequence_pattern', 'observed_value': 'p1',
            'expected_baseline_or_threshold': '>= 2 sessions',
            'explanation': 'pattern repeated',
        })
    return {
        'correlation_id': fid,
        'correlation_type': ftype,
        'entity_a': (entities or ['u1'])[0],
        'entity_b': (entities or ['u1'])[1] if len(entities or []) > 1 else '',
        'entities': entities or ['u1'],
        'sessions': sessions or ['s1'],
        'correlation_score': score,
        'shared_ip_only': shared_ip_only,
        'evidence': evidence,
        'operating_note': note,
    }


def _corr_df(findings):
    cols = ['correlation_id', 'correlation_type', 'entity_a', 'entity_b',
            'entities', 'sessions', 'correlation_score', 'shared_ip_only',
            'evidence', 'operating_note']
    return pd.DataFrame(findings, columns=cols)


@pytest.fixture
def mgr(tmp_path):
    cfg = IncidentConfig(store_path=str(tmp_path / 'incidents.json'))
    return IncidentManager(config=cfg, store=IncidentStore(cfg.store_path))


class TestAutoGrouping:
    def test_shared_user_groups_findings(self, mgr):
        findings = [
            _finding('c1', entities=['alice'], sessions=['s1']),
            _finding('c2', entities=['alice', 'bob'], sessions=['s2'],
                     ftype='similar_account_behavior'),
            _finding('c3', entities=['carol'], sessions=['s9']),
        ]
        incs = mgr.build_incidents(_corr_df(findings))
        # similar_account_behavior user pairs are type-scoped (Milestone
        # 2.3): they form their own cluster rather than merging with
        # single-user findings for the same account.
        assert len(incs) == 3
        carol = next(i for i in incs if 'carol' in i['entity_users'])
        assert len(carol['findings']) == 1

    def test_similar_account_findings_cluster_together(self, mgr):
        findings = [
            _finding('c1', entities=['alice', 'bob'], sessions=['sx1'],
                     ftype='similar_account_behavior'),
            _finding('c2', entities=['bob', 'carol'], sessions=['sx2'],
                     ftype='similar_account_behavior'),
            _finding('c3', entities=['alice'], sessions=['s1']),
        ]
        incs = mgr.build_incidents(_corr_df(findings))
        assert len(incs) == 2
        sim_inc = max(incs, key=lambda i: len(i['findings']))
        assert len(sim_inc['findings']) == 2
        assert set(sim_inc['entity_users']) == {'alice', 'bob', 'carol'}

    def test_shared_ip_links_findings(self, mgr):
        findings = [
            _finding('c1', entities=['u1'], sessions=['s1'],
                     ftype='shared_ip_activity', shared_ip_only=True,
                     ips=['203.0.113.5']),
            _finding('c2', entities=['u2'], sessions=['s2'],
                     ftype='shared_ip_activity', shared_ip_only=True,
                     ips=['203.0.113.5']),
            _finding('c3', entities=['u3'], sessions=['s3'],
                     ftype='repeated_sensitive_operations'),
        ]
        incs = mgr.build_incidents(_corr_df(findings))
        assert len(incs) == 2
        ip_inc = next(i for i in incs if len(i['findings']) == 2)
        assert ip_inc['entity_ips'] == ['203.0.113.5']

    def test_no_grouping_when_disabled(self, mgr):
        cfg = IncidentConfig(store_path=mgr.config.store_path,
                             auto_group_enabled=False)
        mgr2 = IncidentManager(config=cfg, store=mgr.store)
        findings = [_finding('c1', entities=['u1'], sessions=['s1']),
                    _finding('c2', entities=['u1'], sessions=['s2'])]
        incs = mgr2.build_incidents(_corr_df(findings))
        assert len(incs) == 2

    def test_deterministic_incident_ids(self, mgr):
        findings = [_finding('c1', entities=['u1'], sessions=['s1']),
                    _finding('c2', entities=['u1'], sessions=['s2'])]
        a = mgr.build_incidents(_corr_df(findings))
        b = IncidentManager(config=mgr.config,
                            store=IncidentStore(mgr.config.store_path)) \
            .build_incidents(_corr_df(findings))
        assert [i['incident_id'] for i in a] == \
               [i['incident_id'] for i in b]

    def test_empty_correlations_returns_existing(self, mgr):
        assert mgr.build_incidents(pd.DataFrame()) == []
        assert mgr.build_incidents(None) == []


class TestLifecycle:
    def test_create_add_note_set_status(self, mgr):
        mgr._findings = {'c1': _finding('c1', entities=['u1'])}
        inc = mgr.create_from_findings(['c1'], notes='looks suspicious')
        assert inc is not None and inc['source'] == 'manual'
        assert inc['notes'][0]['text'] == 'looks suspicious'
        mgr.set_status(inc['incident_id'], 'Investigating')
        mgr.add_note(inc['incident_id'], 'escalated to IR')
        reloaded = IncidentStore(mgr.config.store_path)
        found = next(i for i in reloaded.incidents
                     if i['incident_id'] == inc['incident_id'])
        assert found['status'] == 'Investigating'
        assert len(found['notes']) == 2
        assert found['notes'][1]['text'] == 'escalated to IR'

    def test_merge_combines_findings_and_notes(self, mgr):
        findings = [_finding('c1', entities=['u1'], sessions=['s1']),
                    _finding('c2', entities=['u2'], sessions=['s2']),
                    _finding('c3', entities=['u3'], sessions=['s3'])]
        incs = mgr.build_incidents(_corr_df(findings), auto_group=False)
        assert len(incs) == 3
        mgr.add_note(incs[0]['incident_id'], 'first note')
        merged = mgr.merge([i['incident_id'] for i in incs[:2]])
        assert merged is not None
        assert len(merged['findings']) == 2
        assert merged['entity_users'] == ['u1', 'u2']
        assert any('Merged incidents' in n['text'] for n in merged['notes'])
        assert any(n['text'] == 'first note' for n in merged['notes'])
        assert mgr.store.get(incs[0]['incident_id']) is None

    def test_split_moves_findings(self, mgr):
        findings = [_finding('c1', entities=['u1'], sessions=['s1']),
                    _finding('c2', entities=['u1'], sessions=['s2'])]
        incs = mgr.build_incidents(_corr_df(findings))
        inc = incs[0]
        new_inc = mgr.split(inc['incident_id'], ['c2'])
        assert new_inc is not None
        assert [f['correlation_id'] for f in new_inc['findings']] == ['c2']
        updated = mgr.store.get(inc['incident_id'])
        assert [f['correlation_id'] for f in updated['findings']] == ['c1']
        # Splitting the only remaining finding is rejected.
        assert mgr.split(inc['incident_id'], ['c1']) is None

    def test_invalid_status_rejected(self, mgr):
        mgr._findings = {'c1': _finding('c1')}
        inc = mgr.create_from_findings(['c1'])
        assert mgr.set_status(inc['incident_id'], 'Bogus') is inc or \
            mgr.store.get(inc['incident_id'])['status'] == 'Open'

    def test_upsert_preserves_analyst_state(self, mgr):
        findings = [_finding('c1', entities=['u1'], sessions=['s1']),
                    _finding('c2', entities=['u1'], sessions=['s2'])]
        mgr.build_incidents(_corr_df(findings))
        inc = mgr.store.incidents[0]
        mgr.set_status(inc['incident_id'], 'Contained')
        mgr.add_note(inc['incident_id'], 'triaged')
        # Rebuild with same findings → same ID, state preserved.
        mgr.build_incidents(_corr_df(findings))
        reloaded = mgr.store.incidents[0]
        assert reloaded['status'] == 'Contained'
        assert len(reloaded['notes']) == 1
        assert len(mgr.store.incidents) == 1


class TestSeverity:
    def test_threshold_boundaries(self, mgr):
        assert mgr.severity_for([_finding('a', score=0.3)]) == 'Low'
        assert mgr.severity_for([_finding('a', score=0.5)]) == 'Medium'
        assert mgr.severity_for([_finding('a', score=0.7)]) == 'High'
        assert mgr.severity_for([_finding('a', score=0.95)]) == 'Critical'
        # Max score across findings wins.
        assert mgr.severity_for([_finding('a', score=0.2),
                                 _finding('b', score=0.9)]) == 'Critical'
        assert mgr.severity_for([]) == 'Low'


class TestStore:
    def test_missing_file_starts_empty(self, tmp_path):
        store = IncidentStore(str(tmp_path / 'nope.json'))
        assert store.incidents == [] and store.load_error is None

    def test_corrupt_file_recovers(self, tmp_path):
        p = tmp_path / 'bad.json'
        p.write_text('{not json', encoding='utf-8')
        store = IncidentStore(str(p))
        assert store.incidents == []
        assert store.load_error is not None


class TestReports:
    def _incident(self, mgr):
        findings = [
            _finding('c1', entities=['alice', 'bob'],
                     sessions=['s1', 's2'], score=0.85,
                     ftype='similar_account_behavior', ips=['203.0.113.9']),
            _finding('c2', entities=['alice'], sessions=['s3'],
                     score=0.5, shared_ip_only=True,
                     ftype='shared_ip_activity', ips=['203.0.113.9'],
                     note='Shared IP only — insufficient evidence'),
        ]
        mgr.build_incidents(_corr_df(findings))
        return mgr.store.incidents[0]

    def test_csv_contains_sections(self, mgr):
        inc = self._incident(mgr)
        data = generate_incident_csv(inc)
        text = data.decode('utf-8')
        assert inc['incident_id'] in text
        assert 'similar_account_behavior' in text
        assert 'Analyst' not in text  # no notes yet
        assert 'shared_source_ip' in text
        assert 'disclaimer' in text

    def test_csv_sanitizes_formulas(self, mgr):
        inc = self._incident(mgr)
        inc['title'] = '=cmd|calc'
        data = generate_incident_csv(inc).decode('utf-8')
        assert "'=cmd" in data

    def test_pdf_is_valid(self, mgr):
        inc = self._incident(mgr)
        pdf = generate_incident_pdf(inc)
        assert pdf[:5] == b'%PDF-'
        assert len(pdf) > 1000

    def test_pdf_shared_ip_caveat(self, mgr):
        findings = [_finding('c1', entities=['u1', 'u2'],
                             sessions=['s1'], shared_ip_only=True,
                             ftype='shared_ip_activity', ips=['198.51.100.1'],
                             note='Shared IP only')]
        mgr.build_incidents(_corr_df(findings))
        inc = mgr.store.incidents[0]
        pdf = generate_incident_pdf(inc)
        assert pdf[:5] == b'%PDF-'
        csv_text = generate_incident_csv(inc).decode('utf-8')
        assert 'shared_ip_only' in csv_text


class TestEvidenceTimeline:
    def test_timeline_filters_and_orders(self, mgr):
        findings = [_finding('c1', entities=['u1'],
                             sessions=['sess_a', 'sess_b'])]
        mgr.build_incidents(_corr_df(findings))
        inc = mgr.store.incidents[0]
        events = pd.DataFrame([
            {'timestamp': '2026-02-02 10:00:00', 'user_id': 'u1',
             'session_id': 'sess_b', 'event_type': 'x',
             'combined_score': 0.5, 'risk_level': 'High'},
            {'timestamp': '2026-02-01 09:00:00', 'user_id': 'u1',
             'session_id': 'sess_a', 'event_type': 'y',
             'combined_score': 0.2, 'risk_level': 'Low'},
            {'timestamp': '2026-02-01 08:00:00', 'user_id': 'other',
             'session_id': 'zzz', 'event_type': 'z',
             'combined_score': 0.9, 'risk_level': 'High'},
        ])
        tl = mgr.evidence_timeline(inc['incident_id'], events)
        assert len(tl) == 2
        assert tl.iloc[0]['session_id'] == 'sess_a'  # time-ordered
        assert set(tl.iloc[0]['finding_ids']) == {'c1'}

    def test_timeline_empty_without_events(self, mgr):
        mgr._findings = {'c1': _finding('c1', sessions=['s1'])}
        inc = mgr.create_from_findings(['c1'])
        assert len(mgr.evidence_timeline(inc['incident_id'], None)) == 0
        assert len(mgr.evidence_timeline(inc['incident_id'],
                                         pd.DataFrame())) == 0


class TestListOrdering:
    def test_severity_then_recency(self, mgr):
        findings = [
            _finding('c1', entities=['u1'], sessions=['s1'], score=0.9),
            _finding('c2', entities=['u2'], sessions=['s2'], score=0.2),
            _finding('c3', entities=['u3'], sessions=['s3'], score=0.5),
        ]
        mgr.build_incidents(_corr_df(findings), auto_group=False)
        ordered = mgr.list_incidents()
        assert ordered[0]['severity'] == 'Critical'
        assert ordered[-1]['severity'] == 'Low'
        # Filters
        assert all(i['severity'] == 'High' for i in
                   mgr.list_incidents(severity='High'))
