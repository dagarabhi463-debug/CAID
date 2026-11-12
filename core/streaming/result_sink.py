"""
Durable result sink — Milestone 2.5.

Scored streaming events are committed here BEFORE the queue entry is
acknowledged. The sink is an append-only JSONL file with flush+fsync
semantics — the "downstream commit" in the ack-after-commit contract.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Iterable, Optional


class JsonlResultSink:
    """Append-only JSONL sink; ``commit`` flushes and fsyncs."""

    def __init__(self, path: str, fsync: bool = True):
        self.path = path
        self.fsync = fsync
        self._lock = threading.Lock()
        self._fh = None
        self.records_written = 0
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Append mode: replay/restart never truncates prior results.
        self._fh = open(path, 'a', encoding='utf-8')

    def append(self, record: Dict) -> None:
        line = json.dumps(record, default=str, ensure_ascii=False)
        with self._lock:
            self._fh.write(line + '\n')
            self._fh.flush()
            self.records_written += 1

    def write_batch(self, records: Iterable[Dict]) -> None:
        lines = [
            json.dumps(record, default=str, ensure_ascii=False) + '\n'
            for record in records
        ]
        with self._lock:
            for line in lines:
                self._fh.write(line)
                self.records_written += 1
            self._fh.flush()

    def commit(self) -> None:
        """Durable commit point: data is on stable storage after this."""
        with self._lock:
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.flush()
                if self.fsync:
                    os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
