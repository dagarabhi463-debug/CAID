"""
Local Ollama client for CAID's AI assistant.

Talks to a locally running Ollama server over its HTTP API using only
the standard library — no cloud AI dependencies, no API keys.

Configuration (environment variables, overridable per-call):
    OLLAMA_BASE_URL      default http://localhost:11434
    OLLAMA_MODEL         default llama3.1:8b
    OLLAMA_TIMEOUT       default 120 (seconds)
    OLLAMA_MAX_RETRIES   default 2 (retries on transient failures)
    OLLAMA_RETRY_BACKOFF default 0.5 (base backoff seconds, exponential)
    OLLAMA_TEMPERATURE   optional float; when set, sent as the default
                         sampling temperature for every generate call
                         (0.0 = deterministic greedy decoding)
    OLLAMA_SEED          optional int; fixed RNG seed for reproducibility
    OLLAMA_CA_BUNDLE     optional path to a CA bundle for https:// bases
    OLLAMA_VERIFY_SSL    '0'/'false' disables TLS verification for
                         https:// bases (INSECURE — dev only)

All failures raise typed OllamaError subclasses so the UI can degrade
gracefully when Ollama is not installed, not running, or missing the
configured model. ``generate`` retries transient failures
(unavailable/timeout) with exponential backoff; model-not-found is
never retried. ``last_stats`` records token usage and stage durations
from the most recent generation for telemetry.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_BASE_URL = 'http://localhost:11434'
DEFAULT_MODEL = 'llama3.1:8b'
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 0.5


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
                 timeout: Optional[float] = None,
                 max_retries: Optional[int] = None,
                 retry_backoff: Optional[float] = None,
                 temperature: Optional[float] = None,
                 seed: Optional[int] = None,
                 ca_bundle: Optional[str] = None,
                 verify_ssl: Optional[bool] = None):
        self.base_url = (base_url or _env('OLLAMA_BASE_URL',
                                          DEFAULT_BASE_URL)).rstrip('/')
        self.model = model or _env('OLLAMA_MODEL', DEFAULT_MODEL)
        try:
            self.timeout = float(timeout if timeout is not None
                                 else _env('OLLAMA_TIMEOUT',
                                           str(DEFAULT_TIMEOUT)))
        except (TypeError, ValueError):
            self.timeout = DEFAULT_TIMEOUT
        try:
            self.max_retries = int(max_retries if max_retries is not None
                                   else _env('OLLAMA_MAX_RETRIES',
                                             str(DEFAULT_MAX_RETRIES)))
        except (TypeError, ValueError):
            self.max_retries = DEFAULT_MAX_RETRIES
        self.max_retries = max(0, self.max_retries)
        try:
            self.retry_backoff = float(
                retry_backoff if retry_backoff is not None
                else _env('OLLAMA_RETRY_BACKOFF',
                          str(DEFAULT_RETRY_BACKOFF)))
        except (TypeError, ValueError):
            self.retry_backoff = DEFAULT_RETRY_BACKOFF
        self.retry_backoff = max(0.0, self.retry_backoff)
        # Sampling options: explicit args win, then env vars. ``None``
        # leaves the option unset so Ollama uses its own defaults.
        self.options: Dict[str, Any] = {}
        temp = temperature
        if temp is None:
            raw = os.environ.get('OLLAMA_TEMPERATURE')
            if raw:
                try:
                    temp = float(raw)
                except ValueError:
                    temp = None
        if temp is not None:
            self.options['temperature'] = float(temp)
        sd = seed
        if sd is None:
            raw = os.environ.get('OLLAMA_SEED')
            if raw:
                try:
                    sd = int(raw)
                except ValueError:
                    sd = None
        if sd is not None:
            self.options['seed'] = int(sd)
        # TLS configuration for https:// base URLs.
        self.ca_bundle = ca_bundle or os.environ.get('OLLAMA_CA_BUNDLE')
        verify_env = os.environ.get('OLLAMA_VERIFY_SSL')
        self.verify_ssl = (verify_ssl if verify_ssl is not None
                           else (verify_env.lower() not in
                                 ('0', 'false', 'no')
                                 if verify_env else True))
        self._ssl_context: Optional[ssl.SSLContext] = None
        # Telemetry from the most recent generate() call.
        self.last_stats: Dict[str, Any] = {}

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
            if self.base_url.startswith('https://'):
                resp = urllib.request.urlopen(
                    req, timeout=effective_timeout,
                    context=self._get_ssl_context())
            else:
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

    def _request_with_retries(self, path: str,
                              payload: Optional[dict] = None,
                              stream: bool = False,
                              timeout: Optional[float] = None):
        """``_request`` with exponential backoff on transient failures.

        Retries ``OllamaUnavailable`` and ``OllamaTimeout`` up to
        ``self.max_retries`` times. ``OllamaModelNotFound`` and other
        ``OllamaError`` subclasses are never retried.
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._request(path, payload=payload, stream=stream,
                                     timeout=timeout)
            except (OllamaUnavailable, OllamaTimeout):
                if attempt >= self.max_retries:
                    raise
                delay = self.retry_backoff * (2 ** attempt)
                if delay > 0:
                    time.sleep(delay)
        raise OllamaUnavailable('unreachable')  # pragma: no cover

    @staticmethod
    def _extract_stats(data: dict, model: Optional[str] = None
                       ) -> Dict[str, Any]:
        """Pull token counts and stage durations (ns -> ms) from a
        generate response payload."""
        stats: Dict[str, Any] = {}
        if model:
            stats['model'] = model
        prompt_tokens = data.get('prompt_eval_count')
        eval_tokens = data.get('eval_count')
        if isinstance(prompt_tokens, (int, float)):
            stats['prompt_tokens'] = int(prompt_tokens)
        if isinstance(eval_tokens, (int, float)):
            stats['eval_tokens'] = int(eval_tokens)
        if 'prompt_tokens' in stats and 'eval_tokens' in stats:
            stats['total_tokens'] = (stats['prompt_tokens'] +
                                     stats['eval_tokens'])
        for src, dst in (('total_duration', 'total_duration_ms'),
                         ('load_duration', 'load_duration_ms'),
                         ('eval_duration', 'eval_duration_ms')):
            val = data.get(src)
            if isinstance(val, (int, float)):
                stats[dst] = round(val / 1e6, 1)
        return stats

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

    def health(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Connection health check with latency and model availability.

        Returns a dict; never raises. ``reachable`` is False with an
        ``error`` string when the server cannot be contacted.
        """
        started = time.monotonic()
        info: Dict[str, Any] = {
            'reachable': False,
            'base_url': self.base_url,
            'configured_model': self.model,
            'resolved_model': None,
            'models': [],
            'latency_ms': None,
            'error': None,
        }
        try:
            models = self.list_models()
        except OllamaError as exc:
            info['error'] = str(exc)
            info['latency_ms'] = round(
                (time.monotonic() - started) * 1000, 1)
            return info
        info['reachable'] = True
        info['models'] = models
        info['latency_ms'] = round((time.monotonic() - started) * 1000, 1)
        if models:
            info['resolved_model'] = (self.model if self.model in models
                                      else models[0])
        return info

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

    def _get_ssl_context(self) -> ssl.SSLContext:
        """SSL context for https:// Ollama endpoints (cached)."""
        if self._ssl_context is None:
            if self.verify_ssl:
                self._ssl_context = ssl.create_default_context(
                    cafile=self.ca_bundle or None)
            else:
                self._ssl_context = ssl._create_unverified_context()
        return self._ssl_context

    def generate(self, prompt: str, model: Optional[str] = None,
                 system: Optional[str] = None,
                 stream: bool = False,
                 timeout: Optional[float] = None,
                 options: Optional[Dict[str, Any]] = None):
        """Generate a completion.

        Returns the full response text when ``stream=False``; otherwise
        yields text chunks. Raises typed OllamaError subclasses on
        failure. ``options`` maps to Ollama sampling options
        (``temperature``, ``seed``, ``num_predict``, ...) and is merged
        over the client's configured defaults.
        """
        use_model = model or self.resolve_model() or self.model
        payload = {
            'model': use_model,
            'prompt': prompt,
            'stream': bool(stream),
        }
        if system:
            payload['system'] = system
        merged_options = dict(self.options)
        if options:
            merged_options.update(options)
        if merged_options:
            payload['options'] = merged_options
        self.last_stats = {}
        resp = self._request_with_retries('/api/generate', payload=payload,
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
            self.last_stats = self._extract_stats(data, model=use_model)
            return str(data.get('response', ''))
        self.last_stats = {'model': use_model}
        return self._iter_stream(resp, stats=self.last_stats)

    @staticmethod
    def _iter_stream(resp, stats: Optional[Dict[str, Any]] = None
                     ) -> Iterator[str]:
        """Yield text chunks from a streaming /api/generate response.

        When ``stats`` is given, token/duration fields from the final
        ``done`` chunk are merged into it.
        """
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
            if stats is not None:
                stats.update(OllamaClient._extract_stats(data))
            chunk = data.get('response')
            if chunk:
                yield str(chunk)
