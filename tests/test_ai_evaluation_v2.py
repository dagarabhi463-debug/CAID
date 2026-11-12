"""Tests for Phase 5.1 AI evaluation v2: repeatability, evidence validation,
ATT&CK validation, and new metrics."""

import json

import pandas as pd
import pytest

from core.ai_evaluation import (
    aggregate_repeated_runs,
    build_scenarios,
    check_empty_evidence_response,
    run_scenario_repeatedly,
    validate_attack_mapping,
    validate_evidence_references_strict,
)


class _JsonClient:
    """Fake client returning a fixed structured JSON payload."""

    model = 'fake-model'
    last_stats = {'total_tokens': 5}

    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def resolve_model(self):
        return 'fake-model'

    def generate(self, prompt, model=None, system=None, stream=False,
                 options=None):
        self.prompts.append(prompt)
        if stream:
            return iter([self.payload])
        return self.payload


def _good_payload():
    return json.dumps({
        'assessment': 'Bulk download anomaly for user_001',
        'observations': ['Combined score 0.9, risk High',
                         'Bulk Download Pattern rule fired'],
        'evidence_refs': ['E1', 'E2'],
        'hypotheses': ['Possible data staging'],
        'limitations': ['Single event'],
        'confidence': 'medium'})


def _resp_with_evidence():
    return {
        'text': 'test',
        'model': 'fake',
        'degraded': False,
        'error': None,
        'elapsed_ms': 12.3,
        'usage': {'total_tokens': 10},
        'structured': {
            'assessment': 'test',
            'observations': ['Score 0.9'],
            'evidence_refs': ['E1'],
            'hypotheses': [],
            'limitations': [],
            'confidence': 'high',
            'dropped_evidence_refs': 0,
        },
        'structured_error': None,
        'context': {
            'facts': {'user_id': 'user_001'},
            'evidence': ['Score 0.9, risk High'],
            'limitations': [],
        },
        'evidence_used': [],
    }


# ----------------------------------------------------------------------
# Evidence validation
# ----------------------------------------------------------------------


def test_validate_evidence_references_strict_valid():
    resp = _resp_with_evidence()
    result = validate_evidence_references_strict(resp)
    assert result['valid'] is True
    assert result['invalid_refs'] == []
    assert result['unsupported_observations'] == []


def test_validate_evidence_references_strict_invalid_ref():
    resp = _resp_with_evidence()
    resp['structured']['evidence_refs'] = ['E1', 'E99', 'E-1']
    result = validate_evidence_references_strict(resp)
    assert result['valid'] is False
    assert 'E99' in result['invalid_refs']
    assert 'E-1' in result['invalid_refs']


def test_validate_evidence_references_strict_unsupported_obs():
    resp = _resp_with_evidence()
    resp['structured']['observations'] = ['Score 0.9 with user_999 exfiltrated 500GB']
    result = validate_evidence_references_strict(resp)
    assert result['valid'] is False
    assert len(result['unsupported_observations']) == 1
    assert 'user_999' in result['unsupported_observations'][0]['fabricated_ids']


def test_validate_evidence_references_strict_empty_evidence():
    resp = _resp_with_evidence()
    resp['context']['evidence'] = []
    resp['context']['facts'] = {}
    resp['structured']['observations'] = ['Some observation']
    result = validate_evidence_references_strict(resp)
    assert result['valid'] is False
    assert 'empty evidence' in result['reason']


def test_validate_evidence_references_strict_empty_with_caveat():
    resp = _resp_with_evidence()
    resp['context']['evidence'] = []
    resp['context']['facts'] = {}
    resp['structured']['observations'] = ['Insufficient evidence available']
    resp['structured']['limitations'] = ['No evidence to analyze']
    resp['structured']['evidence_refs'] = []  # No refs when empty
    result = validate_evidence_references_strict(resp)
    assert result['valid'] is True
    assert result['valid_ref_count'] == 0


# ----------------------------------------------------------------------
# ATT&CK validation
# ----------------------------------------------------------------------


def test_validate_attack_mapping_found_in_evidence():
    response = 'Technique T1213 was observed in the logs.'
    evidence = 'ATT&CK T1213 Data from Information Repositories detected.'
    result = validate_attack_mapping(response, evidence)
    assert 't1213' in result['found_techniques']
    assert result['unsupported_techniques'] == []


def test_validate_attack_mapping_unsupported():
    response = 'Technique T1059 was used.'
    evidence = 'No ATT&CK mappings available.'
    result = validate_attack_mapping(response, evidence)
    assert 't1059' in result['found_techniques']
    assert 't1059' in result['unsupported_techniques']


def test_validate_attack_mapping_expected():
    response = 'Technique T1213 and T1567 were observed.'
    evidence = 'ATT&CK T1213 and T1567 mappings.'
    result = validate_attack_mapping(response, evidence,
                                     expected_mappings=['T1213', 'T1567'])
    assert result['mapping_accuracy'] == 1.0
    assert result['missing_expected'] == []


def test_validate_attack_mapping_missing_expected():
    response = 'Technique T1213 was observed.'
    evidence = 'ATT&CK T1213 mapping.'
    result = validate_attack_mapping(response, evidence,
                                     expected_mappings=['T1213', 'T1567'])
    assert result['mapping_accuracy'] < 1.0
    assert 't1567' in result['missing_expected']


# ----------------------------------------------------------------------
# Empty evidence check
# ----------------------------------------------------------------------


def test_check_empty_evidence_response_not_applicable():
    resp = _resp_with_evidence()
    result = check_empty_evidence_response(resp)
    assert result['applicable'] is False
    assert result['passed'] is None


def test_check_empty_evidence_response_without_caveat():
    resp = _resp_with_evidence()
    resp['context']['evidence'] = []
    resp['context']['facts'] = {}
    resp['text'] = 'This is a response.'
    result = check_empty_evidence_response(resp)
    assert result['applicable'] is True
    assert result['passed'] is False


def test_check_empty_evidence_response_with_caveat():
    resp = _resp_with_evidence()
    resp['context']['evidence'] = []
    resp['context']['facts'] = {}
    resp['text'] = 'Insufficient evidence available for analysis.'
    result = check_empty_evidence_response(resp)
    assert result['applicable'] is True
    assert result['passed'] is True


# ----------------------------------------------------------------------
# Repeated runs
# ----------------------------------------------------------------------


def test_run_scenario_repeatedly_single_scenario():
    scenarios = {s['id']: s for s in build_scenarios()}
    scenario = scenarios['event_bulk_download']
    client = _JsonClient(_good_payload())

    result = run_scenario_repeatedly(scenario, client, model='fake', runs=3)

    assert result['scenario_id'] == 'event_bulk_download'
    assert result['runs'] == 3
    assert len(result['individual_results']) == 3
    assert result['summary']['total_runs'] == 3
    assert result['summary']['successful_runs'] == 3
    assert result['summary']['scenario_pass_rate'] == 1.0


def test_run_scenario_repeatedly_records_indices():
    scenarios = {s['id']: s for s in build_scenarios()}
    scenario = scenarios['event_bulk_download']
    client = _JsonClient(_good_payload())

    result = run_scenario_repeatedly(scenario, client, model='fake', runs=2)

    assert result['individual_results'][0]['run_index'] == 0
    assert result['individual_results'][1]['run_index'] == 1
    assert 'run_timestamp' in result['individual_results'][0]


def test_aggregate_repeated_runs():
    scenarios = {s['id']: s for s in build_scenarios()}
    client = _JsonClient(_good_payload())

    # Run 2 scenarios with 2 runs each
    # Use event_bulk_download and ask_event_count (both have no required caveats)
    sr1 = run_scenario_repeatedly(scenarios['event_bulk_download'],
                                  client, model='fake', runs=2)
    sr2 = run_scenario_repeatedly(scenarios['ask_event_count'],
                                  client, model='fake', runs=2)

    result = aggregate_repeated_runs([sr1, sr2])

    assert result['summary']['total_runs'] == 4
    assert result['summary']['scenario_count'] == 2
    assert result['summary']['overall_pass_rate'] == 1.0
    assert result['summary']['repeatability_rate'] == 1.0
    assert len(result['scenario_results']) == 2


def test_aggregate_repeated_runs_with_failures():
    scenarios = {s['id']: s for s in build_scenarios()}

    # Create a client that returns bad response
    bad_payload = json.dumps({
        'assessment': 'user_999 attacked',
        'observations': ['Score 0.42'],
        'evidence_refs': ['E1'],
        'hypotheses': [],
        'limitations': [],
        'confidence': 'high'})

    client_bad = _JsonClient(bad_payload)
    client_good = _JsonClient(_good_payload())

    sr1 = run_scenario_repeatedly(scenarios['event_bulk_download'],
                                  client_bad, model='fake', runs=2)
    sr2 = run_scenario_repeatedly(scenarios['finding_shared_ip'],
                                  client_good, model='fake', runs=2)

    result = aggregate_repeated_runs([sr1, sr2])

    # Should have degraded pass rate due to fabrication in sr1
    assert result['summary']['overall_pass_rate'] < 1.0
    assert result['summary']['scenario_count'] == 2


# ----------------------------------------------------------------------
# Scenario catalog
# ----------------------------------------------------------------------


def test_build_scenarios_v2_count():
    scenarios = build_scenarios()
    # Phase 5.1: 9 original + 31 new = 40 scenarios
    assert len(scenarios) == 40


def test_scenario_ids_unique():
    scenarios = build_scenarios()
    ids = [s['id'] for s in scenarios]
    assert len(ids) == len(set(ids))


def test_scenario_categories_present():
    scenarios = build_scenarios()
    ids = [s['id'] for s in scenarios]

    # Check for v2 categories
    assert any('evidence_' in id for id in ids)
    assert any('incident_' in id for id in ids)
    assert any('mitre_' in id for id in ids)
    assert any('injection_' in id for id in ids)
    assert any('numeric_' in id for id in ids)
    assert any('reliability_' in id for id in ids)


# ----------------------------------------------------------------------
# New metrics in aggregate
# ----------------------------------------------------------------------


def test_aggregate_includes_v2_metrics():
    from core.ai_evaluation import aggregate
    results = [
        {'degraded': False, 'term_recall': 1.0, 'fabricated_entities': [],
         'unsupported_numbers': [], 'observations_unsupported': 0,
         'forbidden_hits': [], 'caveat_required': True, 'caveat_ok': True,
         'structured': {'structured_requested': True, 'parsed': True,
                        'dropped_refs': 0},
         'elapsed_ms': 100.0, 'usage': {'total_tokens': 10}},
    ]
    s = aggregate(results)
    assert 'heuristic_detected_fabrication_rate' in s
    assert 'evidence_reference_precision' in s
    assert 'structured_schema_validity_rate' in s
    assert 'caveat_compliance_rate' in s


# ----------------------------------------------------------------------
# Phase 5.1b: evidence-ref scores, attack stages, waivers
# ----------------------------------------------------------------------

from core.ai_evaluation import (  # noqa: E402
    _critical_failure_categories,
    evidence_ref_scores,
    validate_attack_stages,
)


def test_evidence_ref_scores_precision_recall():
    resp = _resp_with_evidence()
    resp['structured']['evidence_refs'] = ['E1']
    # expected_terms '0.9' appears in evidence item 1 only
    out = evidence_ref_scores(resp, ['0.9'])
    assert out['expected_refs'] == ['E1']
    assert out['precision'] == 1.0
    assert out['recall'] == 1.0


def test_evidence_ref_scores_partial():
    resp = _resp_with_evidence()
    resp['context']['evidence'] = [
        'Score 0.9 high', 'Rule bulk_download', 'Session sess_aaa']
    resp['structured']['evidence_refs'] = ['E1']
    out = evidence_ref_scores(resp, ['0.9', 'bulk_download'])
    assert out['expected_refs'] == ['E1', 'E2']
    assert out['precision'] == 1.0
    assert out['recall'] == 0.5


def test_evidence_ref_scores_none_when_no_expected():
    resp = _resp_with_evidence()
    out = evidence_ref_scores(resp, [])
    assert out['precision'] is None and out['recall'] is None


def test_validate_attack_stages_required_missing():
    resp = _resp_with_evidence()
    out = validate_attack_stages(resp, required=True)
    assert out['valid'] is False
    assert out['n_stages'] == 0


def test_validate_attack_stages_valid():
    resp = _resp_with_evidence()
    resp['structured']['attack_stages'] = [
        {'stage': 'collection', 'status': 'observed',
         'evidence_refs': ['E1']},
        {'stage': 'exfiltration', 'status': 'hypothesized',
         'evidence_refs': []}]
    out = validate_attack_stages(resp, required=True)
    assert out['valid'] is True
    assert out['n_stages'] == 2


def test_validate_attack_stages_bad_ref():
    resp = _resp_with_evidence()
    resp['structured']['attack_stages'] = [
        {'stage': 'x', 'status': 'observed', 'evidence_refs': ['E99']}]
    out = validate_attack_stages(resp, required=True)
    assert out['valid'] is False


def test_critical_failure_categories_and_waivers():
    base = {'degraded': False, 'fabricated_entities': [],
            'forbidden_hits': [], 'unsupported_numbers': [],
            'observations_unsupported': 0,
            'structured': {'structured_requested': True,
                           'parsed': False},
            'evidence_validation': {'valid': True},
            'caveat_required': False, 'caveat_ok': None,
            'attack_mapping_validation': {'mapping_accuracy': 1.0},
            'deterministic_check': None}
    cats = _critical_failure_categories(base)
    assert 'schema' in cats and 'evidence' not in cats
    waived = dict(base, waived_checks=['schema'])
    assert _critical_failure_categories(waived) == set()


def test_hunt_scenario_has_ground_truth():
    scenarios = {s['id']: s for s in build_scenarios()}
    truth = scenarios['hunt_failed_auth']['truth']
    assert 'user_002' in truth['expected_terms']


def test_attack_chain_requires_stages():
    scenarios = {s['id']: s for s in build_scenarios()}
    for sid in ['attack_chain', 'incident_mitre',
                'incident_uncertain_chain']:
        assert scenarios[sid]['truth'].get('require_attack_stages')


def test_reliability_scenarios_waive_expected_failures():
    scenarios = {s['id']: s for s in build_scenarios()}
    for sid in ['reliability_timeout', 'reliability_unavailable',
                'reliability_fallback']:
        assert 'degraded' in scenarios[sid]['truth']['waive_checks']
