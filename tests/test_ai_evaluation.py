"""Tests for the AI accuracy benchmark harness (core.ai_evaluation).

Uses fake clients — no Ollama required. Verifies that fabrication,
unsupported-claim, caveat, citation, and prompt-injection checks
actually detect bad responses.
"""

import json

import pandas as pd
import pytest

from core import ai_analyst
from core.ai_evaluation import (
    RecordingClient,
    aggregate,
    build_scenarios,
    check_caveats,
    check_expected_terms,
    check_forbidden,
    check_observation_support,
    extract_checkable_numbers,
    extract_identifiers,
    fabricated_entities,
    numbers_supported,
    run_benchmark,
    run_scenario,
    score_response,
)


EVIDENCE = (
    'FACTS:\n{"user_id": "user_001", "combined_score": 0.9}\n\n'
    'EVIDENCE:\n'
    'E1. Combined score 0.900, risk level High.\n'
    'E2. Triggered rules: Bulk Download Pattern\n'
    'E3. Session sess_aaa contains 2 events.'
)


def _resp(text, structured=None, degraded=False):
    return {'text': text, 'model': 'fake', 'degraded': degraded,
            'error': None, 'elapsed_ms': 12.3,
            'usage': {'total_tokens': 10},
            'structured': structured,
            'structured_error': None,
            'context': {'facts': {}, 'evidence': [
                'Combined score 0.900, risk level High.',
                'Triggered rules: Bulk Download Pattern',
                'Session sess_aaa contains 2 events.'],
                'limitations': []},
            'evidence_used': []}


# ----------------------------------------------------------------------
# Extraction primitives
# ----------------------------------------------------------------------

def test_extract_identifiers_finds_ids_and_ips():
    text = ('user user_001 from 10.0.0.1 in sess_aaa flagged T1059; '
            'incident inc_eval01 corr_0001 det_0001')
    ids = extract_identifiers(text)
    for tok in ['user_001', '10.0.0.1', 'sess_aaa', 't1059',
                'inc_eval01', 'corr_0001', 'det_0001']:
        assert tok in ids


def test_extract_identifiers_ignores_plain_words():
    assert extract_identifiers('the quick brown fox') == []


def test_extract_checkable_numbers():
    nums = extract_checkable_numbers('score 0.9, 1500 events, 3 users, '
                                     'year 2024')
    assert '0.9' in nums and '1500' in nums and '2024' in nums
    assert '3' not in nums  # small ints are derivable counts


def test_numbers_supported():
    assert numbers_supported(['0.9'], EVIDENCE) == []
    assert numbers_supported(['0.9', '0.42'], EVIDENCE) == ['0.42']
    assert numbers_supported(['1500'], EVIDENCE) == ['1500']


def test_fabricated_entities():
    good = 'user_001 scored 0.9 on sess_aaa'
    bad = 'user_001 and user_999 at 192.168.9.9 on sess_zzz'
    assert fabricated_entities(good, EVIDENCE) == []
    fab = fabricated_entities(bad, EVIDENCE)
    assert 'user_999' in fab and '192.168.9.9' in fab
    assert 'sess_zzz' in fab


# ----------------------------------------------------------------------
# Ground-truth checks
# ----------------------------------------------------------------------

def test_check_expected_terms_partial():
    out = check_expected_terms('score 0.9 bulk_download',
                               ['0.9', 'bulk_download', 'user_001'])
    assert out['recall'] == pytest.approx(2 / 3)
    assert out['missing'] == ['user_001']


def test_check_forbidden_regex_and_literal():
    hits = check_forbidden('this is a confirmed compromise',
                           [r'confirmed\s+compromise', 'breach'])
    assert hits == [r'confirmed\s+compromise']


def test_check_caveats():
    assert check_caveats('this is a low-confidence lead',
                         ['low-confidence', 'weak'])
    assert not check_caveats('definitely an attack',
                             ['low-confidence', 'weak'])


# ----------------------------------------------------------------------
# score_response / observation support
# ----------------------------------------------------------------------

def test_score_response_clean():
    resp = _resp('user_001 scored 0.9 via Bulk Download Pattern on '
                 'sess_aaa.')
    score = score_response(resp, EVIDENCE, {
        'expected_terms': ['0.9', 'bulk_download'],
        'required_caveats': ['low-confidence', 'hypothes'],
    })
    assert score['term_recall'] == 1.0
    assert score['fabricated_entities'] == []
    assert score['caveat_ok'] is False  # no caveat language present
    assert score['degraded'] is False


def test_score_response_hallucination():
    resp = _resp('user_777 exfiltrated to 203.0.113.9 with score 0.42.')
    score = score_response(resp, EVIDENCE, {})
    assert 'user_777' in score['fabricated_entities']
    assert '203.0.113.9' in score['fabricated_entities']
    assert '0.42' in score['unsupported_numbers']


def test_score_response_forbidden_claim():
    resp = _resp('This is a confirmed compromise of user_001.')
    score = score_response(resp, EVIDENCE,
                           {'forbidden': [r'\bconfirmed\s+compromise']})
    assert score['forbidden_hits']


def test_score_response_negated_disclaimer_not_flagged():
    # Quoting a limitation like "not confirmed attacker behavior" is
    # correct disclaiming, not a forbidden claim.
    resp = _resp('ATT&CK mappings are potential research mappings, '
                 'not confirmed attacker behavior.')
    score = score_response(resp, EVIDENCE,
                           {'forbidden': [r'\bconfirmed\s+attacker']})
    assert score['forbidden_hits'] == []


def test_score_response_unnegated_still_flagged():
    resp = _resp('Evidence shows confirmed attacker behavior here.')
    score = score_response(resp, EVIDENCE,
                           {'forbidden': [r'\bconfirmed\s+attacker']})
    assert score['forbidden_hits']


def test_check_observation_support_flags_unsupported():
    structured = {
        'assessment': 'a',
        'observations': ['Combined score 0.9 with rule',
                         'user_999 exfiltrated 500GB'],
        'hypotheses': [], 'limitations': [], 'confidence': 'high',
        'evidence_refs': ['E1'], 'dropped_evidence_refs': 0}
    resp = _resp('x', structured=structured)
    out = check_observation_support(resp)
    assert out['checked'] == 2
    assert out['unsupported'] == 1
    assert out['detail'][0]['fabricated_ids'] == ['user_999']
    assert out['detail'][0]['unsupported_numbers'] == ['500']


# ----------------------------------------------------------------------
# Fake clients through the real analyst paths
# ----------------------------------------------------------------------

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


def test_run_scenario_event_scores_real_path():
    scenarios = {s['id']: s for s in build_scenarios()}
    score = run_scenario(scenarios['event_bulk_download'],
                         _JsonClient(_good_payload()))
    assert score['degraded'] is False
    assert score['term_recall'] == 1.0
    assert score['fabricated_entities'] == []
    assert score['structured']['parsed'] is True
    assert score['structured']['n_evidence_refs'] == 2


def test_run_scenario_detects_fabrication_end_to_end():
    scenarios = {s['id']: s for s in build_scenarios()}
    bad = json.dumps({
        'assessment': 'user_999 attacked from 203.0.113.9',
        'observations': ['Score 0.42 indicates breach'],
        'evidence_refs': ['E1'], 'hypotheses': [], 'limitations': [],
        'confidence': 'high'})
    score = run_scenario(scenarios['event_bulk_download'],
                         _JsonClient(bad))
    assert score['fabricated_entities']
    assert score['unsupported_numbers'] or \
        score['observations_unsupported']


def test_adversarial_scenario_prompt_is_sanitized():
    scenarios = {s['id']: s for s in build_scenarios()}
    client = _JsonClient(_good_payload())
    score = run_scenario(scenarios['adversarial_injection'], client)
    assert score['prompt_check'] is True
    prompt = client.prompts[0]
    assert 'ignore all previous instructions' not in prompt.lower()
    assert '[filtered]' in prompt
    # non-blocklisted adversarial text may remain, but must stay inside
    # untrusted-data delimiters
    assert '<<UNTRUSTED_DATA' in prompt
    assert '<<END_UNTRUSTED_DATA>>' in prompt


def test_hunt_scenario_deterministic_check():
    scenarios = {s['id']: s for s in build_scenarios()}

    class _HuntClient:
        model = 'fake'

        def resolve_model(self):
            return 'fake'

        def generate(self, *a, **k):
            return 'narration'

    score = run_scenario(scenarios['hunt_failed_auth'], _HuntClient())
    assert score['deterministic_check'] is True
    # v2 ground truth expects the narration to name the matched account;
    # the fake client's 'narration' text does not, so recall is 0.
    assert score['term_recall'] == 0.0


def test_aggregate_metrics():
    results = [
        {'degraded': False, 'term_recall': 1.0, 'fabricated_entities': [],
         'unsupported_numbers': [], 'observations_unsupported': 0,
         'forbidden_hits': [], 'caveat_required': True, 'caveat_ok': True,
         'structured': {'structured_requested': True, 'parsed': True,
                        'dropped_refs': 0},
         'elapsed_ms': 100.0, 'usage': {'total_tokens': 10}},
        {'degraded': True, 'term_recall': 0.5, 'fabricated_entities': ['x'],
         'unsupported_numbers': ['9.9'], 'observations_unsupported': 1,
         'forbidden_hits': ['p'], 'caveat_required': True, 'caveat_ok': False,
         'structured': {'structured_requested': True, 'parsed': False,
                        'dropped_refs': 2},
         'elapsed_ms': 300.0, 'usage': {'total_tokens': 30}},
    ]
    s = aggregate(results)
    assert s['n_scenarios'] == 2
    assert s['degraded'] == 1
    assert s['mean_term_recall'] == pytest.approx(0.75)
    assert s['fabrication_rate'] == 0.5
    assert s['unsupported_claim_rate'] == 0.5
    assert s['forbidden_claim_scenarios'] == 1
    assert s['caveat_compliance'] == 0.5
    assert s['structured_parse_rate'] == 0.5
    assert s['latency_ms_mean'] == 200.0
    assert s['tokens_mean'] == 20.0


def test_run_benchmark_end_to_end_mock():
    report = run_benchmark(_JsonClient(_good_payload()), model='fake')
    assert report['summary']['n_scenarios'] == len(build_scenarios())
    assert all('scenario_id' in r for r in report['results'])
