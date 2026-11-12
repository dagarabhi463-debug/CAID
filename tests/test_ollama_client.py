"""Tests for core.ollama_client — Ollama is always mocked."""

import io
import json
import urllib.error

import pytest

from core.ollama_client import (
    OllamaClient,
    OllamaError,
    OllamaModelNotFound,
    OllamaTimeout,
    OllamaUnavailable,
)


class FakeResponse:
    def __init__(self, payload, json_lines=False):
        if json_lines:
            self._data = payload.encode('utf-8')
        else:
            self._data = json.dumps(payload).encode('utf-8')

    def read(self):
        return self._data

    def __iter__(self):
        for line in self._data.splitlines(keepends=True):
            yield line


def _http_error(code, body=''):
    return urllib.error.HTTPError(
        'http://localhost:11434/api/x', code, 'err', {},
        io.BytesIO(body.encode('utf-8')))


@pytest.fixture
def client():
    return OllamaClient(base_url='http://localhost:11434',
                        model='test-model', timeout=5)


def test_is_available_true(client, monkeypatch):
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda req, timeout=None: FakeResponse({'models': [{'name': 'm1'}]}))
    assert client.is_available() is True


def test_is_available_false(client, monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError('connection refused')
    monkeypatch.setattr('urllib.request.urlopen', boom)
    assert client.is_available() is False


def test_list_models(client, monkeypatch):
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda req, timeout=None: FakeResponse(
            {'models': [{'name': 'llama3.1:8b'}, {'name': 'qwen2.5:3b'}]}))
    assert client.list_models() == ['llama3.1:8b', 'qwen2.5:3b']


def test_generate_success(client, monkeypatch):
    captured = {}

    def fake(req, timeout=None):
        if req.data is None:  # GET /api/tags during model resolution
            return FakeResponse({'models': [{'name': 'test-model'}]})
        captured['body'] = json.loads(req.data.decode('utf-8'))
        return FakeResponse({'response': 'hello analyst'})

    monkeypatch.setattr('urllib.request.urlopen', fake)
    out = client.generate('explain this')
    assert out == 'hello analyst'
    assert captured['body']['model'] == 'test-model'
    assert captured['body']['stream'] is False


def test_generate_with_system(client, monkeypatch):
    captured = {}

    def fake(req, timeout=None):
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        captured['body'] = json.loads(req.data.decode('utf-8'))
        return FakeResponse({'response': 'ok'})

    monkeypatch.setattr('urllib.request.urlopen', fake)
    client.generate('p', system='be brief')
    assert captured['body']['system'] == 'be brief'


def test_model_not_found_http(client, monkeypatch):
    def fake(req, timeout=None):
        raise _http_error(404, '{"error":"model \'x\' not found"}')
    monkeypatch.setattr('urllib.request.urlopen', fake)
    with pytest.raises(OllamaModelNotFound):
        client._request('/api/tags')


def test_model_not_found_generate(client, monkeypatch):
    def fake(req, timeout=None):
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        return FakeResponse({'error': 'model "x" not found'})
    monkeypatch.setattr('urllib.request.urlopen', fake)
    with pytest.raises(OllamaModelNotFound):
        client.generate('hi')


def test_timeout(client, monkeypatch):
    def fake(req, timeout=None):
        raise TimeoutError('timed out')
    monkeypatch.setattr('urllib.request.urlopen', fake)
    with pytest.raises(OllamaTimeout):
        client.list_models()


def test_unavailable(client, monkeypatch):
    def fake(req, timeout=None):
        raise urllib.error.URLError('connection refused')
    monkeypatch.setattr('urllib.request.urlopen', fake)
    with pytest.raises(OllamaUnavailable):
        client.list_models()


def test_malformed_response(client, monkeypatch):
    class Bad(FakeResponse):
        def read(self):
            return b'not json'
    monkeypatch.setattr('urllib.request.urlopen',
                        lambda req, timeout=None: Bad({}))
    with pytest.raises(OllamaUnavailable):
        client.list_models()


def test_resolve_model_prefers_configured(client, monkeypatch):
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda req, timeout=None: FakeResponse(
            {'models': [{'name': 'test-model'}, {'name': 'other'}]}))
    assert client.resolve_model() == 'test-model'


def test_resolve_model_falls_back(client, monkeypatch):
    client.model = 'missing-model'
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda req, timeout=None: FakeResponse(
            {'models': [{'name': 'first'}, {'name': 'second'}]}))
    assert client.resolve_model() == 'first'


def test_resolve_model_none_when_unavailable(client, monkeypatch):
    def boom(req, timeout=None):
        raise urllib.error.URLError('down')
    monkeypatch.setattr('urllib.request.urlopen', boom)
    assert client.resolve_model() is None


def test_generate_stream(client, monkeypatch):
    payload = ('{"response": "Hel"}\n'
               '{"response": "lo"}\n'
               '{"done": true}\n').encode('utf-8')

    class StreamResp:
        def __iter__(self):
            for line in payload.splitlines(keepends=True):
                yield line

    def fake(req, timeout=None):
        if req.data is None:
            return FakeResponse({'models': [{'name': 'test-model'}]})
        return StreamResp()

    monkeypatch.setattr('urllib.request.urlopen', fake)
    chunks = list(client.generate('hi', stream=True))
    assert ''.join(chunks) == 'Hello'
