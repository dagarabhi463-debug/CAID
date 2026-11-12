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
    """JSON metadata + pickled tracker state, atomically replaced.

    Schema versioning: ``meta['version']`` records the checkpoint schema
    generation. ``restore_tracker`` refuses checkpoints newer than
    ``VERSION`` (written by a newer binary — restoring them could
    silently corrupt state) and migrates older snapshots forward through
    ``_migrate_snapshot`` before applying them.
    """

    VERSION = 2
    SUPPORTED_VERSIONS = (1, 2)

    def __init__(self, path: str):
        self.path = path
        self.state_path = path + '.state.pkl'
        self.meta: dict = {}
        self.tracker_state: Optional[bytes] = None
        self.version: Optional[int] = None

    def save(self, *, queue_meta: Optional[dict] = None,
             tracker=None, artifact_fingerprints: Optional[dict] = None,
             monitor_snapshot: Optional[dict] = None,
             extra: Optional[dict] = None) -> str:
        meta = {
            'version': self.VERSION,
            'schema': 'caid-stream-checkpoint',
            'saved_utc': _utcnow(),
            'queue': queue_meta or {},
            'artifact_fingerprints': artifact_fingerprints or {},
            'monitor': monitor_snapshot or {},
            'extra': extra or {},
        }
        state_blob = None
        if tracker is not None:
            snap = self._tracker_snapshot(tracker)
            snap['schema_version'] = self.VERSION
            state_blob = pickle.dumps(snap,
                                      protocol=pickle.HIGHEST_PROTOCOL)
        self._atomic_write_json(self.path, meta)
        if state_blob is not None:
            self._atomic_write_bytes(self.state_path, state_blob)
        # Integrity sidecars: detect tampering/corruption of both the
        # JSON meta and the pickled tracker state before deserializing.
        try:
            from core.artifact_integrity import write_checksum
            write_checksum(self.path)
            if state_blob is not None:
                write_checksum(self.state_path)
        except Exception:
            pass
        self.meta = meta
        self.tracker_state = state_blob
        return self.path

    def load(self):
        if not os.path.exists(self.path):
            return None
        # Integrity check: a present .sha256 sidecar must match before
        # we trust the JSON meta (and any pickle it points at).
        try:
            from core.artifact_integrity import verify_checksum
            if not verify_checksum(self.path):
                return None
        except ImportError:
            pass
        with open(self.path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        version = self.meta.get('version', 1)
        try:
            self.version = int(version)
        except (TypeError, ValueError):
            self.version = None
        if os.path.exists(self.state_path):
            # Verify pickle integrity before reading bytes that will be
            # fed to pickle.loads (tampered pickle = arbitrary code).
            try:
                from core.artifact_integrity import verify_checksum
                if not verify_checksum(self.state_path):
                    self.tracker_state = None
                    return self.meta
            except ImportError:
                pass
            with open(self.state_path, 'rb') as f:
                self.tracker_state = f.read()
        return self.meta

    @staticmethod
    def _migrate_snapshot(snap: dict, from_version: int) -> dict:
        """Migrate a pickled tracker snapshot to the current schema.

        v1 snapshots predate the explicit ``schema_version`` marker and
        the bounded ``late_details`` traceability map; both are filled
        with safe defaults. Ordering buffers (``_timed``/``_explicit``)
        are intentionally NOT checkpointed — un-acked queue entries are
        redelivered by the queue itself.
        """
        snap = dict(snap or {})
        if from_version < 2:
            snap.setdefault('late_details', {})
            snap.setdefault('content_seen', OrderedDict())
        return snap

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
        """Restore pickled tracker state onto a fresh tracker instance.

        Returns False (caller falls back to a clean tracker) when the
        checkpoint is missing, unreadable, or written by a NEWER schema
        version than this binary supports.
        """
        self.load()
        if not self.tracker_state:
            return False
        if (self.version is not None
                and self.version not in self.SUPPORTED_VERSIONS):
            return False
        try:
            snap = pickle.loads(self.tracker_state)
            if not isinstance(snap, dict):
                return False
            snap_version = int(snap.get('schema_version')
                               or self.version or 1)
            if snap_version > self.VERSION:
                return False
            snap = self._migrate_snapshot(
                snap, min(snap_version, self.VERSION))
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
