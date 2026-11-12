"""Tests for Phase 1/2 AI reliability: retries, health, telemetry,
and schema-validated structured responses."""

import io
import json
import urllib.error

import pandas as pd
import pytest

from core.ai_analyst import (
    explain_event,
    parse_structured_response,
    validate_structured,
)
from core.ai_context import event_context
from core.ollama_client import (
    OllamaClient,
    OllamaModelNotFound,
    OllamaUnavailable,
)


class FakeResponse:
    def __init__(self, payload):
        self._data = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._data

    def __iter__(self):
        for line in self._data.splitlines(keepends=True):
            yield line


def _client(**kw):
    kw.setdefault('base_url', 'http://localhost:11434')
    kw.setdefault('model', 'test-model')
    kw.setdefault('timeout', 5)
    kw.setdefault('retry_backoff', 0)
    return OllamaClient(**kw)


@pytest.fixture
def scored_df():
    return pd.DataFrame([
        {'timestamp': '2024-01-01 10:00', 'user_id': 'u1',
         'session_id': 's1', 'source_ip': '10.0.0.1',
         'event_type': 'bulk_download', 'combined_score': 0.9,
         'risk_level': 'High', 'is_anomaly': True,
         'detection_id': 'DET-000001', 'event_id': 'evt-abc',
         'auth_result': ''},
    ])


# ----------------------------------------------------------------------
# Client: retries
# ----------------------------------------------------------------------

def test_generate_retries_transient_failure(monkeypatch):
    calls = {'n': 0}

    def fake(req, timeout=None):
        calls['n'] += 1
        if calls['n'] == 1:
            raise urllib.error.URLError('connection reset')
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        return FakeResponse({'response': 'ok'})

    monkeypatch.setattr('urllib.request.urlopen', fake)
    client = _client(max_retries=2)
    assert client.generate('hi') == 'ok'
    # 1 failed generate attempt + 1 successful (+1 tags call for resolve)
    assert calls['n'] >= 2


def test_generate_no_retry_on_model_not_found(monkeypatch):
    calls = {'n': 0}

    def fake(req, timeout=None):
        calls['n'] += 1
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        return FakeResponse({'error': 'model "x" not found'})

    monkeypatch.setattr('urllib.request.urlopen', fake)
    client = _client(max_retries=3)
    with pytest.raises(OllamaModelNotFound):
        client.generate('hi')
    assert calls['n'] == 2  # tags + single generate attempt


def test_generate_gives_up_after_max_retries(monkeypatch):
    calls = {'n': 0}

    def fake(req, timeout=None):
        calls['n'] += 1
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        raise urllib.error.URLError('down')

    monkeypatch.setattr('urllib.request.urlopen', fake)
    client = _client(max_retries=2)
    with pytest.raises(OllamaUnavailable):
        client.generate('hi')
    # tags(1) + generate attempts(3)
    assert calls['n'] == 4


# ----------------------------------------------------------------------
# Client: health + telemetry
# ----------------------------------------------------------------------

def test_health_reachable(monkeypatch):
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda req, timeout=None: FakeResponse(
            {'models': [{'name': 'test-model'}, {'name': 'other'}]}))
    info = _client().health()
    assert info['reachable'] is True
    assert info['resolved_model'] == 'test-model'
    assert info['models'] == ['test-model', 'other']
    assert info['latency_ms'] is not None
    assert info['error'] is None


def test_health_unreachable(monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError('connection refused')
    monkeypatch.setattr('urllib.request.urlopen', boom)
    info = _client().health()
    assert info['reachable'] is False
    assert info['resolved_model'] is None
    assert info['error']


def test_generate_records_token_stats(monkeypatch):
    def fake(req, timeout=None):
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        return FakeResponse({
            'response': 'hi',
            'prompt_eval_count': 10, 'eval_count': 5,
            'total_duration': 2_000_000_000,
            'load_duration': 100_000_000,
            'eval_duration': 1_000_000_000})

    monkeypatch.setattr('urllib.request.urlopen', fake)
    client = _client()
    assert client.generate('hi') == 'hi'
    stats = client.last_stats
    assert stats['prompt_tokens'] == 10
    assert stats['eval_tokens'] == 5
    assert stats['total_tokens'] == 15
    assert stats['eval_duration_ms'] == 1000.0


# ----------------------------------------------------------------------
# Analyst: telemetry + structured responses
# ----------------------------------------------------------------------

class _FakeClient:
    model = 'fake-model'

    def __init__(self, text):
        self._text = text
        self.last_stats = {'total_tokens': 42, 'model': 'fake-model'}

    def resolve_model(self):
        return 'fake-model'

    def generate(self, prompt, model=None, system=None, stream=False,
                 options=None):
        if stream:
            return iter([self._text])
        return self._text


def test_ask_records_elapsed_and_usage(scored_df):
    payload = json.dumps({
        'assessment': 'suspicious bulk download',
        'observations': ['score 0.9'],
        'evidence_refs': ['E1'],
        'hypotheses': [], 'limitations': [], 'confidence': 'high'})
    resp = explain_event(scored_df.iloc[0], scored_df,
                         client=_FakeClient(payload), structured=True)
    assert resp['elapsed_ms'] is not None
    assert resp['usage']['total_tokens'] == 42


def test_explain_event_structured_valid(scored_df):
    payload = json.dumps({
        'assessment': 'Bulk download anomaly',
        'observations': ['Combined score 0.9 with bulk_download rule'],
        'evidence_refs': ['E1', 'E2', 'E99', 'bogus'],
        'hypotheses': ['Possible data staging'],
        'limitations': ['Single event only'],
        'confidence': 'medium'})
    resp = explain_event(scored_df.iloc[0], scored_df,
                         client=_FakeClient(payload), structured=True)
    s = resp['structured']
    assert s is not None
    assert s['assessment'] == 'Bulk download anomaly'
    assert s['confidence'] == 'medium'
    # Only E1 exists in this fixture's evidence; E2/E99/'bogus' dropped
    assert s['evidence_refs'] == ['E1']
    assert s['dropped_evidence_refs'] == 3
    assert 'invalid evidence' in (resp['structured_error'] or '')


def test_explain_event_structured_invalid_json(scored_df):
    resp = explain_event(scored_df.iloc[0], scored_df,
                         client=_FakeClient('not json at all'),
                         structured=True)
    assert resp['structured'] is None
    assert resp['structured_error'] == 'response was not valid JSON'
    # raw text still available for display
    assert resp['text'] == 'not json at all'


def test_explain_event_structured_empty_payload(scored_df):
    resp = explain_event(scored_df.iloc[0], scored_df,
                         client=_FakeClient('{"foo": 1}'),
                         structured=True)
    assert resp['structured'] is None
    assert resp['structured_error'] == 'JSON failed schema validation'


def test_explain_event_structured_degraded(scored_df):
    class DeadClient:
        model = 'none'

        def resolve_model(self):
            return None

        def generate(self, *a, **k):
            raise OllamaUnavailable('down')

    resp = explain_event(scored_df.iloc[0], scored_df,
                         client=DeadClient(), structured=True)
    assert resp['degraded'] is True
    assert resp['structured'] is None
    assert 'deterministic evidence summary' in resp['text'].lower()


# ----------------------------------------------------------------------
# Structured parsing / validation units
# ----------------------------------------------------------------------

def test_parse_structured_fenced():
    text = 'Here you go:\n```json\n{"assessment": "a", "observations": ["o"]}\n```\nDone.'
    payload = parse_structured_response(text)
    assert payload['assessment'] == 'a'


def test_parse_structured_prose_wrapped():
    payload = parse_structured_response('prefix {"a": 1} suffix')
    assert payload == {'a': 1}


def test_parse_structured_garbage():
    assert parse_structured_response('no json here') is None
    assert parse_structured_response('') is None
    assert parse_structured_response('[1,2,3]') is None


def test_validate_structured_drops_bad_refs():
    out = validate_structured(
        {'assessment': 'x', 'evidence_refs': ['E1', 'E2', 'E0', 'E-1',
                                             'e3', 'E1']},
        n_evidence=3)
    assert out['evidence_refs'] == ['E1', 'E2', 'E3']
    # E0 out of range, E-1 malformed; duplicate E1 silently deduped
    assert out['dropped_evidence_refs'] == 2


def test_validate_structured_requires_content():
    assert validate_structured({'confidence': 'high'}, 5) is None
    assert validate_structured('nope', 5) is None
    assert validate_structured(None, 5) is None


def test_validate_structured_bad_confidence():
    out = validate_structured(
        {'assessment': 'a', 'confidence': 'certain'}, 0)
    assert out['confidence'] == 'unknown'


# ----------------------------------------------------------------------
# Context: event references
# ----------------------------------------------------------------------

def test_event_context_has_event_ref(scored_df):
    ctx = event_context(scored_df.iloc[0], scored_df)
    assert ctx['facts'].get('event_id') == 'evt-abc'
    assert ctx['facts'].get('detection_id') == 'DET-000001'
    assert ctx['facts'].get('event_ref') in ('DET-000001', 'evt-abc')


def test_event_context_event_ref_fallback(scored_df):
    row = scored_df.iloc[0].copy()
    row['detection_id'] = None
    row['event_id'] = None
    ctx = event_context(row, scored_df)
    assert ctx['facts'].get('event_ref', '').startswith('row_')
