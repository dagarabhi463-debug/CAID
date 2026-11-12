"""Prometheus /metrics HTTP endpoint for the streaming MetricsRegistry.

Stdlib-only implementation (``http.server`` on a daemon thread) — no
``prometheus_client`` dependency required. Point Prometheus at
``http://<host>:<port>/metrics``.

Usage::

    from core.streaming.stream_monitor import MetricsRegistry
    from core.streaming.metrics_server import MetricsHTTPServer

    registry = MetricsRegistry()
    server = MetricsHTTPServer(registry, host='0.0.0.0', port=9090)
    server.start()   # daemon thread; serves until stop()/process exit

Endpoints:
    GET /metrics  -> text exposition format (Prometheus scrape)
    GET /healthz  -> 200 'ok' liveness probe
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from core.streaming.stream_monitor import MetricsRegistry


class _MetricsHandler(BaseHTTPRequestHandler):
    registry: MetricsRegistry = None  # set by MetricsHTTPServer

    def log_message(self, *args):  # silence default stderr logging
        pass

    def _send(self, code: int, body: str, content_type: str) -> None:
        data = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        if self.path == '/metrics':
            try:
                body = self.registry.prometheus_exposition()
            except Exception as exc:  # never 500 without diagnostics
                body = (f'# metrics render error: {exc}\n')
            self._send(200, body, 'text/plain; version=0.0.4; '
                                   'charset=utf-8')
        elif self.path == '/healthz':
            self._send(200, 'ok', 'text/plain; charset=utf-8')
        elif self.path == '/snapshot':
            try:
                body = json.dumps(self.registry.snapshot(), default=str)
            except Exception as exc:
                body = json.dumps({'error': str(exc)})
            self._send(200, body, 'application/json')
        else:
            self._send(404, 'not found', 'text/plain; charset=utf-8')


class MetricsHTTPServer:
    """Daemon-thread HTTP server exposing a MetricsRegistry."""

    def __init__(self, registry: MetricsRegistry,
                 host: str = '127.0.0.1', port: int = 9090):
        self.registry = registry
        self.host = host
        self.port = int(port)
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f'http://{self.host}:{self.port}/metrics'

    def start(self) -> 'MetricsHTTPServer':
        if self._httpd is not None:
            return self
        handler = type('BoundMetricsHandler', (_MetricsHandler,),
                       {'registry': self.registry})
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name='caid-metrics', daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
