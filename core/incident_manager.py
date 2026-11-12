"""
Threat Investigation & Incident Management for CAID — Milestone 2.2

Builds on the Milestone 2.1 cross-session correlation layer. Correlation
findings are grouped into investigation incidents that analysts can
review, annotate, merge, split, and export as PDF/CSV reports.

Design contract:
- RETROSPECTIVE ONLY. Incidents consume correlation findings and scored
  events; they never feed back into per-event causal scores,
  DetectionEngine channel weights, risk levels, or correlation
  thresholds.
- EVIDENCE-FIRST. Incidents store the full finding dicts (including
  evidence lists) so every displayed element traces back to the
  correlation layer.
- DETERMINISTIC. Incident IDs are ``inc_<sha1[:12]>`` over sorted
  finding IDs; grouping is stable for identical input.
- SHARED-IP POLICY PRESERVED. ``shared_ip_only`` findings keep their
  capped score and operating note; incident severity is derived from
  finding scores but the caveat travels with the data.
- JSON PERSISTENCE. Incidents persist to a JSON file so status, notes,
  and groupings survive application restarts.
- SYNTHETIC DATA ONLY. All inputs are synthetic/defensive research
  logs; no real Canvas connectivity.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class IncidentConfig:
    """Grouping, severity, and lifecycle settings for incidents."""

    auto_group_enabled: bool = True
    # Findings are unioned into one incident when they share any of
    # these entity kinds: 'user' (entities), 'session' (sessions),
    # 'ip' (IPs extracted from finding evidence).
    group_entity_kinds: tuple = ('user', 'session', 'ip')
    min_findings_per_incident: int = 1
    # severity_thresholds maps an upper bound on the max finding score
    # to a severity label; evaluated low→high, last match wins.
    severity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'Low': 0.4,
        'Medium': 0.6,
        'High': 0.8,
        'Critical': 1.01,
    })
    statuses: tuple = ('Open', 'Investigating', 'Contained',
                       'Closed', 'FalsePositive')
    default_status: str = 'Open'
    store_path: str = os.path.join('data', 'incidents', 'incidents.json')

    def to_dict(self) -> Dict:
        return {
            'auto_group_enabled': bool(self.auto_group_enabled),
            'group_entity_kinds': list(self.group_entity_kinds),
            'min_findings_per_incident': int(self.min_findings_per_incident),
            'severity_thresholds': dict(self.severity_thresholds),
            'statuses': list(self.statuses),
            'default_status': self.default_status,
            'store_path': self.store_path,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'IncidentConfig':
        if not data:
            return cls()
        base = cls().to_dict()
        base.update({k: v for k, v in data.items() if k in base})
        base['group_entity_kinds'] = tuple(base['group_entity_kinds'])
        base['statuses'] = tuple(base['statuses'])
        return cls(**base)


def _inc_id(finding_ids: List[str]) -> str:
    blob = json.dumps({'findings': sorted(finding_ids)},
                      sort_keys=True).encode('utf-8')
    return 'inc_' + hashlib.sha1(blob).hexdigest()[:12]


def _finding_ips(finding: Dict) -> set:
    """Extract IPs referenced in a finding's evidence."""
    ips = set()
    for ev in finding.get('evidence') or []:
        if not isinstance(ev, dict):
            continue
        if ev.get('signal') in ('shared_source_ip', 'source_ip'):
            val = ev.get('observed_value')
            if isinstance(val, str) and val:
                ips.add(val)
    return ips


def _finding_entities(finding: Dict, kinds: tuple) -> set:
    """Entity keys used for grouping: u:<user>, s:<session>, ip:<ip>.

    ``similar_account_behavior`` findings get type-scoped user keys
    (``sim:u:<user>``) so pairwise similarity links form their own
    cluster instead of merging every suspicious account into one
    incident. ``shared_ip_activity`` keeps unscoped user keys: users on
    one IP are intended to group with that IP's other findings.
    """
    keys = set()
    if 'user' in kinds:
        scope = 'sim:' if finding.get('correlation_type') == \
            'similar_account_behavior' else ''
        for e in finding.get('entities') or []:
            if e:
                keys.add(f'{scope}u:{e}')
    if 'session' in kinds:
        for s in finding.get('sessions') or []:
            if s:
                keys.add(f's:{s}')
    if 'ip' in kinds:
        for ip in _finding_ips(finding):
            keys.add(f'ip:{ip}')
    return keys


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class IncidentStore:
    """JSON-file persistence for incident records."""

    def __init__(self, path: str):
        self.path = path
        self.incidents: List[Dict] = []
        self.load_error: Optional[str] = None
        self.load()

    def load(self) -> None:
        self.incidents = []
        self.load_error = None
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data = data.get('incidents', [])
            if not isinstance(data, list):
                raise ValueError('incidents payload is not a list')
            self.incidents = [d for d in data if isinstance(d, dict)]
        except Exception as exc:  # corrupt/missing → empty store
            self.incidents = []
            self.load_error = f'{type(exc).__name__}: {exc}'

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {'incidents': self.incidents}
        fd, tmp = tempfile.mkstemp(dir=directory or '.', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2, default=str)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def get(self, incident_id: str) -> Optional[Dict]:
        for inc in self.incidents:
            if inc.get('incident_id') == incident_id:
                return inc
        return None

    def upsert(self, incident: Dict) -> Dict:
        """Insert or refresh an incident. Preserves analyst state."""
        existing = self.get(incident['incident_id'])
        if existing is not None:
            incident['status'] = existing.get('status', incident.get('status'))
            incident['notes'] = existing.get('notes', incident.get('notes', []))
            incident['created_at'] = existing.get(
                'created_at', incident.get('created_at'))
            incident['source'] = existing.get('source', incident.get('source'))
            idx = self.incidents.index(existing)
            self.incidents[idx] = incident
        else:
            self.incidents.append(incident)
        self.save()
        return incident

    def remove(self, incident_id: str) -> bool:
        before = len(self.incidents)
        self.incidents = [i for i in self.incidents
                          if i.get('incident_id') != incident_id]
        if len(self.incidents) != before:
            self.save()
            return True
        return False

    def clear(self) -> None:
        self.incidents = []
        self.save()


class IncidentManager:
    """Groups correlation findings into incidents and manages lifecycle."""

    def __init__(self, config: Optional[IncidentConfig] = None,
                 store: Optional[IncidentStore] = None,
                 clock: Optional[Callable[[], datetime]] = None):
        self.config = config or IncidentConfig()
        self.store = store or IncidentStore(self.config.store_path)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._findings: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Entity / severity helpers
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return self.clock().isoformat()

    def _entities_of(self, findings: List[Dict]) -> Dict[str, List[str]]:
        users, sessions, ips = set(), set(), set()
        for f in findings:
            users.update(str(e) for e in (f.get('entities') or []) if e)
            sessions.update(str(s) for s in (f.get('sessions') or []) if s)
            ips.update(_finding_ips(f))
        return {
            'entity_users': sorted(users),
            'entity_sessions': sorted(sessions),
            'entity_ips': sorted(ips),
        }

    def severity_for(self, findings: List[Dict]) -> str:
        if not findings:
            return 'Low'
        max_score = max(float(f.get('correlation_score') or 0.0)
                        for f in findings)
        severity = 'Low'
        for label in sorted(self.config.severity_thresholds,
                            key=lambda k: self.config.severity_thresholds[k]):
            if max_score < self.config.severity_thresholds[label]:
                severity = label
                break
        else:
            severity = sorted(self.config.severity_thresholds,
                              key=lambda k: self.config.severity_thresholds[k])[-1]
        return severity

    def _make_incident(self, findings: List[Dict], source: str,
                       status: Optional[str] = None,
                       notes: Optional[List[Dict]] = None,
                       created_at: Optional[str] = None) -> Dict:
        finding_ids = sorted(str(f.get('correlation_id')) for f in findings
                             if f.get('correlation_id'))
        users = sorted({str(e) for f in findings
                        for e in (f.get('entities') or []) if e})
        n = len(findings)
        if len(users) == 1:
            title = f'{n} correlated finding{"s" if n != 1 else ""} — user {users[0]}'
        elif users:
            title = (f'{n} correlated findings across '
                     f'{len(users)} accounts ({", ".join(users[:3])}'
                     f'{"…" if len(users) > 3 else ""})')
        else:
            title = f'{n} correlated finding{"s" if n != 1 else ""}'
        now = self._now()
        return {
            'incident_id': _inc_id(finding_ids),
            'title': title,
            'status': status or self.config.default_status,
            'severity': self.severity_for(findings),
            'finding_ids': finding_ids,
            'findings': findings,
            **self._entities_of(findings),
            'notes': list(notes or []),
            'created_at': created_at or now,
            'updated_at': now,
            'source': source,
        }

    def _refresh(self, incident: Dict) -> Dict:
        """Recompute derived fields after findings/status changes."""
        findings = incident.get('findings') or []
        incident['severity'] = self.severity_for(findings)
        incident.update(self._entities_of(findings))
        incident['finding_ids'] = sorted(
            str(f.get('correlation_id')) for f in findings
            if f.get('correlation_id'))
        incident['incident_id'] = _inc_id(incident['finding_ids'])
        users = incident['entity_users']
        n = len(findings)
        if len(users) == 1:
            incident['title'] = (
                f'{n} correlated finding{"s" if n != 1 else ""} — '
                f'user {users[0]}')
        elif users:
            incident['title'] = (
                f'{n} correlated findings across {len(users)} accounts '
                f'({", ".join(users[:3])}{"…" if len(users) > 3 else ""})')
        else:
            incident['title'] = (
                f'{n} correlated finding{"s" if n != 1 else ""}')
        incident['updated_at'] = self._now()
        return incident

    # ------------------------------------------------------------------
    # Auto-grouping
    # ------------------------------------------------------------------

    def build_incidents(self, correlations: pd.DataFrame,
                        auto_group: Optional[bool] = None) -> List[Dict]:
        """Group correlation findings into incidents (union-find).

        Findings sharing any configured entity kind (user/session/IP)
        are merged into a single candidate incident. Existing incidents
        with the same deterministic ID are refreshed while preserving
        analyst status/notes.
        """
        if correlations is None or len(correlations) == 0:
            return list(self.store.incidents)

        records = correlations.to_dict('records')
        self._findings = {str(r.get('correlation_id')): r
                          for r in records if r.get('correlation_id')}

        do_group = self.config.auto_group_enabled if auto_group is None \
            else auto_group
        groups: List[List[Dict]] = []

        if do_group and len(records) > 1:
            keys = [_finding_entities(r, self.config.group_entity_kinds)
                    for r in records]
            uf = _UnionFind(len(records))
            seen: Dict[str, int] = {}
            for i, ks in enumerate(keys):
                for k in ks:
                    if k in seen:
                        uf.union(i, seen[k])
                    else:
                        seen[k] = i
            buckets: Dict[int, List[Dict]] = {}
            for i, r in enumerate(records):
                buckets.setdefault(uf.find(i), []).append(r)
            groups = [g for g in buckets.values()
                      if len(g) >= self.config.min_findings_per_incident]
        else:
            groups = [[r] for r in records]
            groups = [g for g in groups
                      if len(g) >= self.config.min_findings_per_incident]

        out = []
        for g in groups:
            inc = self._make_incident(g, source='auto')
            self.store.upsert(inc)
            out.append(self.store.get(inc['incident_id']))
        return [i for i in out if i is not None]

    # ------------------------------------------------------------------
    # Manual lifecycle operations
    # ------------------------------------------------------------------

    def _resolve_findings(self, finding_ids: List[str]) -> List[Dict]:
        resolved = []
        for fid in finding_ids:
            f = self._findings.get(str(fid))
            if f is None:
                # Fall back to findings embedded in existing incidents.
                for inc in self.store.incidents:
                    for cand in inc.get('findings') or []:
                        if str(cand.get('correlation_id')) == str(fid):
                            f = cand
                            break
                    if f is not None:
                        break
            if f is not None:
                resolved.append(f)
        return resolved

    def create_from_findings(self, finding_ids: List[str],
                             notes: str = '') -> Optional[Dict]:
        findings = self._resolve_findings(finding_ids)
        if not findings:
            return None
        note_list = []
        if notes:
            note_list.append({'timestamp': self._now(),
                              'author': 'analyst', 'text': notes})
        inc = self._make_incident(findings, source='manual', notes=note_list)
        return self.store.upsert(inc)

    def merge(self, incident_ids: List[str]) -> Optional[Dict]:
        sources = [self.store.get(i) for i in incident_ids]
        sources = [s for s in sources if s is not None]
        if len(sources) < 2:
            return None
        findings, notes = [], []
        src_ids = []
        for s in sources:
            findings.extend(s.get('findings') or [])
            notes.extend(s.get('notes') or [])
            src_ids.append(s['incident_id'])
        notes.append({'timestamp': self._now(), 'author': 'system',
                      'text': f'Merged incidents: {", ".join(sorted(src_ids))}'})
        inc = self._make_incident(findings, source='manual', notes=notes)
        for s in sources:
            self.store.remove(s['incident_id'])
        return self.store.upsert(inc)

    def split(self, incident_id: str,
              finding_ids: List[str]) -> Optional[Dict]:
        inc = self.store.get(incident_id)
        if inc is None:
            return None
        findings = inc.get('findings') or []
        move_ids = {str(f) for f in finding_ids}
        moving = [f for f in findings
                  if str(f.get('correlation_id')) in move_ids]
        remaining = [f for f in findings
                     if str(f.get('correlation_id')) not in move_ids]
        if not moving or not remaining:
            return None
        inc['findings'] = remaining
        self._refresh(inc)
        new_inc = self._make_incident(moving, source=inc.get('source', 'manual'),
                                      notes=[{'timestamp': self._now(),
                                              'author': 'system',
                                              'text': f'Split from {incident_id}'}])
        self.store.upsert(inc)
        return self.store.upsert(new_inc)

    def add_note(self, incident_id: str, text: str,
                 author: str = 'analyst') -> Optional[Dict]:
        inc = self.store.get(incident_id)
        if inc is None or not text:
            return inc
        inc.setdefault('notes', []).append(
            {'timestamp': self._now(), 'author': author, 'text': text})
        inc['updated_at'] = self._now()
        return self.store.upsert(inc)

    def set_status(self, incident_id: str, status: str) -> Optional[Dict]:
        inc = self.store.get(incident_id)
        if inc is None or status not in self.config.statuses:
            return inc
        inc['status'] = status
        inc['updated_at'] = self._now()
        return self.store.upsert(inc)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_incidents(self, status: Optional[str] = None,
                       severity: Optional[str] = None) -> List[Dict]:
        sev_rank = {label: i for i, label in
                    enumerate(sorted(self.config.severity_thresholds,
                                     key=lambda k: self.config.severity_thresholds[k]))}
        rows = [i for i in self.store.incidents
                if (status is None or i.get('status') == status)
                and (severity is None or i.get('severity') == severity)]
        # Stable two-pass sort: updated_at desc first, then severity desc
        # so severity is the primary key and recency breaks ties.
        rows.sort(key=lambda i: str(i.get('updated_at', '')), reverse=True)
        rows.sort(key=lambda i: -sev_rank.get(i.get('severity'), -1))
        return rows

    def evidence_timeline(self, incident_id: str,
                          scored_events: Optional[pd.DataFrame] = None,
                          limit: Optional[int] = None) -> pd.DataFrame:
        """Merged, time-ordered events for all users/sessions in incident."""
        inc = self.store.get(incident_id)
        if inc is None or scored_events is None or len(scored_events) == 0:
            return pd.DataFrame()
        users = set(inc.get('entity_users') or [])
        sessions = set(inc.get('entity_sessions') or [])
        fids = set(inc.get('finding_ids') or [])
        f_sessions = {s for f in (inc.get('findings') or [])
                      for s in (f.get('sessions') or [])}
        sessions |= f_sessions

        ev = scored_events.copy()
        mask = pd.Series(False, index=ev.index)
        if 'user_id' in ev.columns and users:
            mask |= ev['user_id'].astype(str).isin(users)
        if 'session_id' in ev.columns and sessions:
            mask |= ev['session_id'].astype(str).isin(sessions)
        sub = ev[mask]
        if len(sub) == 0:
            return pd.DataFrame()

        cols = [c for c in ['timestamp', 'user_id', 'session_id', 'event_type',
                            'source_ip', 'combined_score', 'risk_level',
                            'sequence_triggered_patterns'] if c in sub.columns]
        out = sub[cols].copy()
        if 'timestamp' in out.columns:
            out['_ts'] = pd.to_datetime(out['timestamp'], errors='coerce')
            out = out.sort_values('_ts', kind='stable').drop(columns='_ts')
        # Tag each row with the finding IDs whose sessions/users cover it.
        finding_map = []
        for _, f in pd.DataFrame(inc.get('findings') or []).iterrows() \
                if inc.get('findings') else []:
            finding_map.append((str(f.get('correlation_id')),
                                set(map(str, f.get('sessions') or [])),
                                set(map(str, f.get('entities') or []))))
        if 'session_id' in out.columns or 'user_id' in out.columns:
            tags = []
            for _, r in out.iterrows():
                sid = str(r.get('session_id', ''))
                uid = str(r.get('user_id', ''))
                hit = [fid for fid, fs, fe in finding_map
                       if sid in fs or uid in fe]
                tags.append(sorted(set(hit)))
            out['finding_ids'] = tags
        if limit:
            out = out.head(limit)
        return out.reset_index(drop=True)
