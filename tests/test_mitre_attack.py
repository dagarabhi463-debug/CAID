"""Tests for core.mitre_attack — STIX parsing and CAID mappings."""

import json

import pytest

from core.mitre_attack import MitreAttackStore


def _tech(tid, name, tactic='initial-access', sub=False, revoked=False,
          deprecated=False):
    return {
        'type': 'attack-pattern',
        'name': name,
        'description': f'{name} description',
        'external_references': [{
            'source_name': 'mitre-attack',
            'external_id': tid,
            'url': f'https://attack.mitre.org/techniques/{tid.replace(".", "/")}',
        }],
        'kill_chain_phases': [{
            'kill_chain_name': 'mitre-attack',
            'phase_name': tactic,
        }],
        'x_mitre_is_subtechnique': sub,
        'x_mitre_attack_spec_version': '3.3.0',
        'revoked': revoked,
        'x_mitre_deprecated': deprecated,
    }


@pytest.fixture
def bundle():
    return {
        'type': 'bundle',
        'id': 'bundle--test',
        'spec_version': '2.0',
        'objects': [
            _tech('T1000', 'Test Technique'),
            _tech('T1000.001', 'Test Sub-technique', sub=True),
            _tech('T2000', 'Revoked Technique', revoked=True),
            _tech('T3000', 'Deprecated Technique', deprecated=True),
            {'type': 'course-of-action', 'name': 'not a technique'},
        ],
    }


@pytest.fixture
def store(tmp_path, bundle):
    raw = json.dumps(bundle).encode('utf-8')
    s = MitreAttackStore(cache_dir=str(tmp_path),
                         fetcher=lambda: raw)
    assert s.ensure_data(allow_download=True)
    return s


def test_load_parses_techniques(store):
    assert 'T1000' in store.techniques
    assert 'T1000.001' in store.techniques
    assert 'T2000' not in store.techniques  # revoked filtered
    assert 'T3000' not in store.techniques  # deprecated filtered


def test_subtechnique_flag(store):
    assert store.get_technique('T1000.001').is_subtechnique is True
    assert store.get_technique('T1000').is_subtechnique is False


def test_tactic_assignment(store):
    tech = store.get_technique('T1000')
    assert 'Initial Access' in tech.tactics
    assert 'T1000' in store.matrix['Initial Access']


def test_framework_version_recorded(store):
    assert store.framework_version
    assert '3.3.0' in store.framework_version
    assert store.retrieved_at


def test_get_missing_technique(store):
    assert store.get_technique('T9999') is None


def test_map_event_sensitive_type(store):
    row = {'event_type': 'bulk_download', 'user_id': 'u1',
           'timestamp': '2024-01-01', 'combined_score': 0.9,
           'risk_level': 'High', 'triggered_rules': []}
    maps = store.map_event(row)
    ids = {m['technique_id'] for m in maps}
    assert 'T1213' in ids
    assert all(m['confidence'] in ('confirmed', 'potential') for m in maps)
    assert all(m['mapping_source'] == 'caid_curated_v1' for m in maps)
    assert all(m['framework_version'] for m in maps)


def test_map_event_login_failure(store):
    row = {'event_type': 'login', 'auth_result': 'failure',
           'triggered_rules': []}
    maps = store.map_event(row)
    assert any(m['technique_id'] == 'T1110' for m in maps)


def test_map_event_normal_activity_unmapped(store):
    row = {'event_type': 'page_view', 'triggered_rules': []}
    assert store.map_event(row) == []


def test_map_finding_correlation(store):
    finding = {'correlation_type': 'repeated_sensitive_operations',
               'correlation_id': 'corr_1', 'entities': ['u1'],
               'sessions': ['s1'], 'correlation_score': 0.8,
               'shared_ip_only': False, 'evidence': []}
    maps = store.map_finding(finding)
    assert any(m['technique_id'] == 'T1213' and
               m['confidence'] == 'confirmed' for m in maps)


def test_shared_ip_only_produces_no_mapping(store):
    finding = {'correlation_type': 'shared_ip_activity',
               'correlation_id': 'corr_2', 'entities': ['u1', 'u2'],
               'sessions': [], 'correlation_score': 0.3,
               'shared_ip_only': True, 'evidence': []}
    assert store.map_finding(finding) == []


def test_map_incident_aggregates(store):
    incident = {'incident_id': 'inc_1', 'findings': [
        {'correlation_type': 'repeated_sensitive_operations',
         'correlation_id': 'c1', 'entities': ['u1'], 'sessions': ['s1'],
         'correlation_score': 0.8, 'shared_ip_only': False, 'evidence': []},
        {'correlation_type': 'shared_ip_activity',
         'correlation_id': 'c2', 'entities': ['u1', 'u2'], 'sessions': [],
         'correlation_score': 0.3, 'shared_ip_only': True, 'evidence': []},
    ]}
    maps = store.map_incident(incident)
    assert maps  # at least the sensitive-ops mappings
    assert all(m['confidence'] in ('confirmed', 'potential') for m in maps)


def test_coverage_stats(store):
    cov = store.coverage()
    assert cov['available'] is True
    assert 'bulk_download_pattern' in cov['mapped_rules']
    assert 'high_api_requests' in cov['unmapped_rules']
    assert cov['confirmed_mappings'] > 0
    assert cov['potential_mappings'] > 0
    # Fixture bundle has no T1213 etc., so per-tactic counts may be empty;
    # just verify the structure is present.
    assert isinstance(cov['techniques_by_tactic'], dict)


def test_download_failure_uses_no_cache(tmp_path, bundle):
    def bad_fetch():
        raise OSError('network down')
    s = MitreAttackStore(cache_dir=str(tmp_path), fetcher=bad_fetch)
    assert s.ensure_data(allow_download=True) is False
    assert s.available is False
    assert s.load_error


def test_corrupt_cache_recovers_via_download(tmp_path, bundle):
    raw = json.dumps(bundle).encode('utf-8')
    s = MitreAttackStore(cache_dir=str(tmp_path), fetcher=lambda: raw)
    # Write garbage to bundle path first.
    import os
    os.makedirs(str(tmp_path), exist_ok=True)
    with open(s.bundle_path, 'wb') as fh:
        fh.write(b'garbage')
    # ensure_data tries cache (fails) then downloads fresh.
    assert s.ensure_data(allow_download=True) is True
    assert 'T1000' in s.techniques


def test_no_download_without_flag(tmp_path, bundle):
    raw = json.dumps(bundle).encode('utf-8')
    s = MitreAttackStore(cache_dir=str(tmp_path), fetcher=lambda: raw)
    assert s.ensure_data(allow_download=False) is False
