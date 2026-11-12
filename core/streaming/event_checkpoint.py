"""
Streaming checkpoints — Milestone 2.5.

Periodic snapshots of consumer position + incremental session/sequence
state + artifact fingerprints. Atomic write (tempfile + os.replace), same
pattern as the incident store. On restart the tracker state is restored
and the queue re-delivers pending entries — scores remain deterministic.
"""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Optional


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class StreamCheckpoint:
    """JSON metadata + pickled tracker state, atomically replaced."""

    VERSION = 1

    def __init__(self, path: str):
        self.path = path
        self.state_path = path + '.state.pkl'
        self.meta: dict = {}
        self.tracker_state: Optional[bytes] = None

    def save(self, *, queue_meta: Optional[dict] = None,
             tracker=None, artifact_fingerprints: Optional[dict] = None,
             monitor_snapshot: Optional[dict] = None,
             extra: Optional[dict] = None) -> str:
        meta = {
            'version': self.VERSION,
            'saved_utc': _utcnow(),
            'queue': queue_meta or {},
            'artifact_fingerprints': artifact_fingerprints or {},
            'monitor': monitor_snapshot or {},
            'extra': extra or {},
        }
        state_blob = None
        if tracker is not None:
            state_blob = pickle.dumps(self._tracker_snapshot(tracker),
                                      protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_json(self.path, meta)
        if state_blob is not None:
            self._atomic_write_bytes(self.state_path, state_blob)
        self.meta = meta
        self.tracker_state = state_blob
        return self.path

    def load(self):
        if not os.path.exists(self.path):
            return None
        with open(self.path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        if os.path.exists(self.state_path):
            with open(self.state_path, 'rb') as f:
                self.tracker_state = f.read()
        return self.meta

    @staticmethod
    def _tracker_snapshot(tracker) -> dict:
        return {
            'timed': tracker._timed,
            'explicit': tracker._explicit,
            'sessions': tracker._sessions,
            'user_assign': tracker._user_assign,
            'arrival_seq': tracker._arrival_seq,
            'late_events': tracker.late_events,
            'late_details': getattr(tracker, 'late_details', {}),
            'content_seen': tracker.content_seen,
        }

    @staticmethod
    def _tracker_restore(tracker, snap: dict) -> None:
        tracker._timed = snap.get('timed', {})
        tracker._explicit = snap.get('explicit', {})
        tracker._sessions = snap.get('sessions', {})
        tracker._user_assign = snap.get('user_assign', {})
        tracker._arrival_seq = snap.get('arrival_seq', 0)
        tracker.late_events = deque(snap.get('late_events', []), maxlen=10_000)
        tracker.late_details = OrderedDict(snap.get('late_details', {}))
        tracker.content_seen = snap.get('content_seen', OrderedDict())

    def restore_tracker(self, tracker) -> bool:
        """Restore pickled tracker state onto a fresh tracker instance."""
        self.load()
        if not self.tracker_state:
            return False
        try:
            snap = pickle.loads(self.tracker_state)
            self._tracker_restore(tracker, snap)
            return True
        except Exception:
            return False

    @staticmethod
    def _atomic_write_json(path: str, obj: dict) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                   suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(obj, f, default=str, indent=2)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    @staticmethod
    def _atomic_write_bytes(path: str, blob: bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)),
                                   suffix='.tmp')
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(blob)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
