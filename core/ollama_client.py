"""
Local Ollama client for CAID's AI assistant.

Talks to a locally running Ollama server over its HTTP API using only
the standard library — no cloud AI dependencies, no API keys.

Configuration (environment variables, overridable per-call):
    OLLAMA_BASE_URL   default http://localhost:11434
    OLLAMA_MODEL      default llama3.1:8b
    OLLAMA_TIMEOUT    default 60 (seconds)

All failures raise typed OllamaError subclasses so the UI can degrade
gracefully when Ollama is not installed, not running, or missing the
configured model.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Iterator, List, Optional

DEFAULT_BASE_URL = 'http://localhost:11434'
DEFAULT_MODEL = 'llama3.1:8b'
DEFAULT_TIMEOUT = 60.0


class OllamaError(Exception):
    """Base error for Ollama client failures."""


class OllamaUnavailable(OllamaError):
    """Server unreachable or returned an unexpected error."""


class OllamaModelNotFound(OllamaError):
    """The requested model is not installed on the server."""


class OllamaTimeout(OllamaError):
    """Request exceeded the configured timeout."""


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


class OllamaClient:
    """Minimal Ollama HTTP client (generate + model listing)."""

    def __init__(self, base_url: Optional[str] = None,
                 model: Optional[str] = None,
                 timeout: Optional[float] = None):
        self.base_url = (base_url or _env('OLLAMA_BASE_URL',
                                          DEFAULT_BASE_URL)).rstrip('/')
        self.model = model or _env('OLLAMA_MODEL', DEFAULT_MODEL)
        try:
            self.timeout = float(timeout if timeout is not None
                                 else _env('OLLAMA_TIMEOUT',
                                           str(DEFAULT_TIMEOUT)))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _request(self, path: str, payload: Optional[dict] = None,
                 stream: bool = False, timeout: Optional[float] = None):
        url = f"{self.base_url}{path}"
        data = None
        headers = {'Content-Type': 'application/json'}
        if payload is not None:
            data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method='POST' if data else 'GET')
        effective_timeout = timeout if timeout is not None else self.timeout
        try:
            resp = urllib.request.urlopen(req, timeout=effective_timeout)
        except urllib.error.HTTPError as exc:
            body = ''
            try:
                body = exc.read().decode('utf-8', errors='replace')
            except Exception:
                pass
            if exc.code == 404 and 'model' in body.lower():
                raise OllamaModelNotFound(body or str(exc)) from exc
            raise OllamaUnavailable(f'HTTP {exc.code}: {body[:300]}') from exc
        except TimeoutError as exc:
            raise OllamaTimeout(f'request timed out after '
                                f'{effective_timeout}s') from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, 'reason', exc)
            if isinstance(reason, TimeoutError) or 'timed out' in str(reason):
                raise OllamaTimeout(str(reason)) from exc
            raise OllamaUnavailable(str(reason)) from exc
        except OSError as exc:
            raise OllamaUnavailable(str(exc)) from exc
        return resp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True when the Ollama server responds to /api/tags."""
        try:
            self.list_models()
            return True
        except OllamaError:
            return False

    def list_models(self) -> List[str]:
        """Return names of installed models ([] when unavailable)."""
        resp = self._request('/api/tags')
        try:
            data = json.loads(resp.read().decode('utf-8'))
        except Exception as exc:
            raise OllamaUnavailable(f'malformed /api/tags response: {exc}')
        models = data.get('models') or []
        names = []
        for m in models:
            name = m.get('name') if isinstance(m, dict) else None
            if name:
                names.append(str(name))
        return names

    def resolve_model(self) -> Optional[str]:
        """Return the configured model, falling back to the first installed.

        Returns None when Ollama is unavailable or has no models.
        """
        try:
            installed = self.list_models()
        except OllamaError:
            return None
        if not installed:
            return None
        if self.model in installed:
            return self.model
        return installed[0]

    def generate(self, prompt: str, model: Optional[str] = None,
                 system: Optional[str] = None,
                 stream: bool = False,
                 timeout: Optional[float] = None):
        """Generate a completion.

        Returns the full response text when ``stream=False``; otherwise
        yields text chunks. Raises typed OllamaError subclasses on
        failure.
        """
        use_model = model or self.resolve_model() or self.model
        payload = {
            'model': use_model,
            'prompt': prompt,
            'stream': bool(stream),
        }
        if system:
            payload['system'] = system
        resp = self._request('/api/generate', payload=payload,
                             stream=stream, timeout=timeout)
        if not stream:
            try:
                data = json.loads(resp.read().decode('utf-8'))
            except Exception as exc:
                raise OllamaUnavailable(
                    f'malformed generate response: {exc}') from exc
            if 'error' in data:
                err = str(data['error'])
                if 'model' in err.lower() and 'not found' in err.lower():
                    raise OllamaModelNotFound(err)
                raise OllamaError(err)
            return str(data.get('response', ''))
        return self._iter_stream(resp)

    @staticmethod
    def _iter_stream(resp) -> Iterator[str]:
        """Yield text chunks from a streaming /api/generate response."""
        for raw in resp:
            line = raw.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' in data:
                raise OllamaError(str(data['error']))
            chunk = data.get('response')
            if chunk:
                yield str(chunk)
