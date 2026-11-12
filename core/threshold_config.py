"""Centralized threshold configuration for CAID — D-7.

One persisted, versioned config governs the tunable detection
thresholds that were previously scattered as per-engine defaults:

- ``risk_thresholds``     -> ``RiskEngine.thresholds``
- ``detection_thresholds``-> ``DetectionEngine.thresholds``
  (``critical``/``high``/``medium`` combined-score cutoffs)
- ``rule_overrides``      -> per-rule ``threshold``/``weight``/
  ``enabled`` via ``DetectionRulesEngine.configure_rule``

Guarantees:

- Every ``save()`` validates the FULL merged config before writing —
  invalid values are rejected, never partially applied.
- Each save bumps ``version`` and appends an audit record
  (``audit.jsonl``: timestamp, actor, reason, changed keys).
- Every version snapshot is retained under ``history/`` so
  ``rollback()`` can restore any prior configuration (itself audited).
- ``apply()`` pushes the active config into live engine instances;
  engines keep their own fingerprint-bound calibration artifacts —
  this layer manages the heuristic thresholds on top.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from typing import Dict, Optional

DEFAULT_DIR = os.path.join('data', 'threshold_config')
CONFIG_NAME = 'thresholds.json'
AUDIT_NAME = 'audit.jsonl'
HISTORY_DIR = 'history'
MAX_HISTORY = 50

# Bounds for every configurable key. Detection cutoffs are scores in
# [0, 1]; risk-engine keys are counts/volumes (positive, sane caps).
_RISK_BOUNDS = {
    'anomaly_score_high': (0.0, 1.0),
    'anomaly_score_medium': (0.0, 1.0),
    'api_threshold': (0.0, 1_000_000.0),
    'data_threshold_mb': (0.0, 1_000_000.0),
    'courses_threshold': (0.0, 1_000_000.0),
    'accounts_threshold': (0.0, 1_000_000.0),
}
_DETECTION_KEYS = ('critical', 'high', 'medium')
_RULE_OVERRIDE_FIELDS = ('threshold', 'weight', 'enabled')


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_config() -> Dict:
    return {
        'risk_thresholds': {
            'anomaly_score_high': 0.8,
            'anomaly_score_medium': 0.6,
            'api_threshold': 300,
            'data_threshold_mb': 10,
            'courses_threshold': 10,
            'accounts_threshold': 5,
        },
        'detection_thresholds': {
            'critical': 0.75,
            'high': 0.55,
            'medium': 0.30,
        },
        'rule_overrides': {},
    }


def _is_finite_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(float(v))


class ThresholdConfigError(ValueError):
    """Raised when a proposed threshold configuration is invalid."""


class ThresholdConfigStore:
    """Versioned, audited, persistent threshold configuration."""

    def __init__(self, directory: str = DEFAULT_DIR):
        self.directory = directory
        self.path = os.path.join(directory, CONFIG_NAME)
        self.audit_path = os.path.join(directory, AUDIT_NAME)
        self.history_dir = os.path.join(directory, HISTORY_DIR)
        self.config: Dict = _default_config()
        self.version: int = 0
        self.updated_utc: Optional[str] = None
        os.makedirs(self.history_dir, exist_ok=True)
        self.load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> Dict:
        """Load the active config; falls back to defaults when absent."""
        if not os.path.exists(self.path):
            return self.config
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return self.config
        if not isinstance(data, dict):
            return self.config
        cfg = _default_config()
        for section in ('risk_thresholds', 'detection_thresholds',
                        'rule_overrides'):
            if isinstance(data.get(section), dict):
                cfg[section].update(data[section])
        # Tolerate unknown keys already on disk (forward compat) by
        # keeping them out of the validated surface.
        self.config = cfg
        try:
            self.version = int(data.get('version', 0))
        except (TypeError, ValueError):
            self.version = 0
        self.updated_utc = data.get('updated_utc')
        return self.config

    def _atomic_write(self, path: str, payload: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)),
                    exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=os.path.dirname(os.path.abspath(path)), suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(payload)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _snapshot(self) -> Dict:
        return {
            'version': self.version,
            'updated_utc': self.updated_utc,
            'risk_thresholds': dict(self.config['risk_thresholds']),
            'detection_thresholds':
                dict(self.config['detection_thresholds']),
            'rule_overrides': json.loads(
                json.dumps(self.config['rule_overrides'], default=str)),
        }

    def _write_snapshot(self, snapshot: Dict) -> None:
        name = f"v{snapshot['version']:05d}.json"
        self._atomic_write(
            os.path.join(self.history_dir, name),
            json.dumps(snapshot, indent=2, default=str))
        # Bound history: drop oldest snapshots beyond MAX_HISTORY.
        snaps = sorted(
            (n for n in os.listdir(self.history_dir)
             if n.startswith('v') and n.endswith('.json')))
        for old in snaps[:-MAX_HISTORY]:
            try:
                os.unlink(os.path.join(self.history_dir, old))
            except OSError:
                pass

    def _audit(self, action: str, actor: str, reason: str,
               changes: Dict) -> None:
        entry = {
            'ts_utc': _utcnow(),
            'version': self.version,
            'action': action,
            'actor': actor or 'unknown',
            'reason': reason or '',
            'changes': changes,
        }
        os.makedirs(self.directory, exist_ok=True)
        with open(self.audit_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, default=str) + '\n')
        try:
            from core.artifact_integrity import restrict_permissions
            restrict_permissions(self.audit_path)
        except Exception:
            pass

    def audit_log(self, limit: int = 100) -> list:
        if not os.path.exists(self.audit_path):
            return []
        out = []
        with open(self.audit_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out[-limit:]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate(config: Dict) -> Dict:
        """Return a normalized copy or raise ThresholdConfigError."""
        if not isinstance(config, dict):
            raise ThresholdConfigError('config must be a dict')
        out = _default_config()

        risk = config.get('risk_thresholds') or {}
        if not isinstance(risk, dict):
            raise ThresholdConfigError('risk_thresholds must be a dict')
        for key, bounds in _RISK_BOUNDS.items():
            if key in risk:
                v = risk[key]
                if not _is_finite_number(v):
                    raise ThresholdConfigError(
                        f'risk_thresholds.{key} must be a finite number')
                if not (bounds[0] <= float(v) <= bounds[1]):
                    raise ThresholdConfigError(
                        f'risk_thresholds.{key}={v} outside {bounds}')
                out['risk_thresholds'][key] = float(v)

        det = config.get('detection_thresholds') or {}
        if not isinstance(det, dict):
            raise ThresholdConfigError('detection_thresholds must be a dict')
        for key in _DETECTION_KEYS:
            if key in det:
                v = det[key]
                if not _is_finite_number(v):
                    raise ThresholdConfigError(
                        f'detection_thresholds.{key} must be a finite number')
                if not (0.0 <= float(v) <= 1.0):
                    raise ThresholdConfigError(
                        f'detection_thresholds.{key}={v} outside [0, 1]')
                out['detection_thresholds'][key] = float(v)
        # Monotonicity: critical >= high >= medium.
        d = out['detection_thresholds']
        if not (d['critical'] >= d['high'] >= d['medium']):
            raise ThresholdConfigError(
                'detection_thresholds must satisfy '
                'critical >= high >= medium')

        rules = config.get('rule_overrides') or {}
        if not isinstance(rules, dict):
            raise ThresholdConfigError('rule_overrides must be a dict')
        for rid, ov in rules.items():
            if not isinstance(rid, str) or not rid.strip():
                raise ThresholdConfigError('rule id must be a non-empty str')
            if not isinstance(ov, dict):
                raise ThresholdConfigError(
                    f'rule_overrides.{rid} must be a dict')
            bad = set(ov) - set(_RULE_OVERRIDE_FIELDS)
            if bad:
                raise ThresholdConfigError(
                    f'rule_overrides.{rid}: unsupported fields {sorted(bad)}')
            norm = {}
            for field in _RULE_OVERRIDE_FIELDS:
                if field not in ov:
                    continue
                v = ov[field]
                if field == 'enabled':
                    if not isinstance(v, bool):
                        raise ThresholdConfigError(
                            f'rule_overrides.{rid}.enabled must be bool')
                    norm[field] = v
                elif field == 'weight':
                    if not _is_finite_number(v) or not (0 <= float(v) <= 1):
                        raise ThresholdConfigError(
                            f'rule_overrides.{rid}.weight must be in [0,1]')
                    norm[field] = float(v)
                else:
                    if not _is_finite_number(v):
                        raise ThresholdConfigError(
                            f'rule_overrides.{rid}.threshold must be finite')
                    norm[field] = float(v)
            out['rule_overrides'][rid] = norm
        return out

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def save(self, changes: Dict, *, actor: str = 'system',
             reason: str = '') -> Dict:
        """Merge ``changes`` into the active config, validate, persist.

        ``changes`` may contain any of the three sections; each is
        deep-merged over the current values. Returns the new snapshot.
        """
        merged = {
            'risk_thresholds': dict(self.config['risk_thresholds']),
            'detection_thresholds':
                dict(self.config['detection_thresholds']),
            'rule_overrides': json.loads(json.dumps(
                self.config['rule_overrides'], default=str)),
        }
        for section in ('risk_thresholds', 'detection_thresholds'):
            upd = (changes or {}).get(section)
            if upd is not None:
                if not isinstance(upd, dict):
                    raise ThresholdConfigError(f'{section} must be a dict')
                merged[section].update(upd)
        upd_rules = (changes or {}).get('rule_overrides')
        if upd_rules is not None:
            if not isinstance(upd_rules, dict):
                raise ThresholdConfigError('rule_overrides must be a dict')
            for rid, ov in upd_rules.items():
                if ov is None:
                    merged['rule_overrides'].pop(rid, None)
                else:
                    merged['rule_overrides'].setdefault(rid, {}).update(ov)

        validated = self.validate(merged)
        # Diff for the audit record.
        diff = {}
        for section in ('risk_thresholds', 'detection_thresholds'):
            for k, v in validated[section].items():
                if self.config[section].get(k) != v:
                    diff[f'{section}.{k}'] = {
                        'from': self.config[section].get(k), 'to': v}
        old_rules = self.config['rule_overrides']
        for rid in set(old_rules) | set(validated['rule_overrides']):
            if old_rules.get(rid) != validated['rule_overrides'].get(rid):
                diff[f'rule_overrides.{rid}'] = {
                    'from': old_rules.get(rid),
                    'to': validated['rule_overrides'].get(rid)}

        self.version += 1
        self.updated_utc = _utcnow()
        self.config = validated
        snapshot = self._snapshot()
        self._atomic_write(self.path, json.dumps(
            {**snapshot, 'updated_utc': self.updated_utc},
            indent=2, default=str))
        self._write_snapshot(snapshot)
        self._audit('save', actor, reason, diff)
        return snapshot

    def list_versions(self) -> list:
        out = []
        if os.path.isdir(self.history_dir):
            for n in sorted(os.listdir(self.history_dir)):
                if n.startswith('v') and n.endswith('.json'):
                    try:
                        out.append(int(n[1:-5]))
                    except ValueError:
                        continue
        return out

    def get_version(self, version: int) -> Optional[Dict]:
        path = os.path.join(self.history_dir, f'v{int(version):05d}.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def rollback(self, version: int, *, actor: str = 'system',
                 reason: str = 'rollback') -> Dict:
        """Restore a prior snapshot as a NEW audited version."""
        snap = self.get_version(version)
        if snap is None:
            raise ThresholdConfigError(
                f'no snapshot for version {version}')
        changes = {
            'risk_thresholds': snap.get('risk_thresholds') or {},
            'detection_thresholds':
                snap.get('detection_thresholds') or {},
            'rule_overrides': snap.get('rule_overrides') or {},
        }
        return self.save(changes, actor=actor,
                         reason=f'rollback to v{version}: {reason}')

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def apply(self, *, risk_engine=None, detection_engine=None,
              rules_engine=None) -> Dict[str, str]:
        """Push the active config into live engine instances.

        Returns a mapping of target -> 'applied'|'skipped'|'error:<msg>'.
        """
        results = {}
        if risk_engine is not None:
            try:
                risk_engine.thresholds.update(
                    dict(self.config['risk_thresholds']))
                results['risk_engine'] = 'applied'
            except Exception as exc:
                results['risk_engine'] = f'error:{exc}'
        if detection_engine is not None:
            try:
                detection_engine.thresholds.update(
                    dict(self.config['detection_thresholds']))
                results['detection_engine'] = 'applied'
            except Exception as exc:
                results['detection_engine'] = f'error:{exc}'
        if rules_engine is not None:
            ok, errs = 0, []
            for rid, ov in (self.config.get('rule_overrides') or {}).items():
                try:
                    changes = {k: v for k, v in ov.items()}
                    if rules_engine.configure_rule(rid, **changes):
                        ok += 1
                    else:
                        errs.append(f'{rid}:unknown')
                except Exception as exc:
                    errs.append(f'{rid}:{exc}')
            results['rules_engine'] = (
                f'applied:{ok}' if not errs else f'partial:{ok}:{errs[:3]}')
        return results


def default_store(directory: str = DEFAULT_DIR) -> ThresholdConfigStore:
    return ThresholdConfigStore(directory)
