"""
MITRE ATT&CK integration for CAID.

Loads the official MITRE ATT&CK Enterprise STIX 2.1 bundle (downloaded
once and cached locally under ``data/mitre_attack/``), indexes tactics
and techniques/sub-techniques, and maps CAID detection rules,
correlation findings, and event evidence to techniques.

Mapping contract:
- Every mapping records technique ID, tactic, CAID source, evidence,
  rationale, confidence, mapping source, and framework version.
- Confidence states: ``confirmed`` (explicit evidence per the curated
  mapping rules), ``potential`` (resembles a technique; needs analyst
  review). Unmapped sources are reported, never guessed.
- A potential ATT&CK mapping is never presented as confirmed attacker
  behavior. Shared-IP-only findings produce no technique mapping.
- Synthetic/defensive research data only — no real Canvas connectivity.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

STIX_URL = (
    'https://raw.githubusercontent.com/mitre/cti/master/'
    'enterprise-attack/enterprise-attack.json'
)
DEFAULT_CACHE_DIR = os.path.join('data', 'mitre_attack')
BUNDLE_FILENAME = 'enterprise-attack.json'
META_FILENAME = 'mitre_attack_meta.json'
MAPPING_SOURCE = 'caid_curated_v1'

TACTIC_NAMES = {
    'reconnaissance': 'Reconnaissance',
    'resource-development': 'Resource Development',
    'initial-access': 'Initial Access',
    'execution': 'Execution',
    'persistence': 'Persistence',
    'privilege-escalation': 'Privilege Escalation',
    'defense-evasion': 'Defense Evasion',
    'stealth': 'Defense Evasion',  # legacy phase name (pre-v10 bundles)
    'defense-impairment': 'Defense Impairment',
    'credential-access': 'Credential Access',
    'discovery': 'Discovery',
    'lateral-movement': 'Lateral Movement',
    'collection': 'Collection',
    'command-and-control': 'Command and Control',
    'exfiltration': 'Exfiltration',
    'impact': 'Impact',
}

# Curated CAID -> ATT&CK mappings. ``state`` is 'confirmed' or
# 'potential'; 'unmapped' sources are listed in UNMAPPED_SOURCES.
SENSITIVE_EVENT_TYPES = (
    'grade_download', 'student_export', 'course_export',
    'bulk_download', 'bulk_export',
)

CAID_MITRE_MAP: List[Dict] = [
    # --- Detection rules -------------------------------------------------
    {
        'caid_source': 'rule:bulk_download_pattern',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': ('Bulk download event types collect repository '
                      'content at volume; matches Data from Information '
                      'Repositories.'),
    },
    {
        'caid_source': 'rule:bulk_download_pattern',
        'technique_id': 'T1567',
        'state': 'potential',
        'rationale': ('Large downloads may precede exfiltration but CAID '
                      'cannot confirm data left the environment.'),
    },
    {
        'caid_source': 'rule:high_data_transfer',
        'technique_id': 'T1567',
        'state': 'potential',
        'rationale': ('High data volume is consistent with exfiltration '
                      'over web channels; destination unverified.'),
    },
    {
        'caid_source': 'rule:critical_access_pattern',
        'technique_id': 'T1213',
        'state': 'potential',
        'rationale': ('Broad account access plus high transfer suggests '
                      'collection; requires analyst confirmation.'),
    },
    {
        'caid_source': 'rule:multiple_accounts_accessed',
        'technique_id': 'T1110',
        'state': 'potential',
        'rationale': ('Access spanning many accounts resembles brute '
                      'force or credential stuffing.'),
    },
    {
        'caid_source': 'rule:old_token_activity',
        'technique_id': 'T1550',
        'state': 'potential',
        'rationale': ('Activity with stale tokens may indicate use of '
                      'alternate/stolen authentication material.'),
    },
    {
        'caid_source': 'rule:after_hours_activity',
        'technique_id': 'T1078',
        'state': 'potential',
        'rationale': ('Off-hours activity is weakly consistent with '
                      'valid-account misuse; low confidence.'),
    },
    # --- Event types -----------------------------------------------------
    {
        'caid_source': 'event_type:bulk_download',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': 'Bulk download event directly evidences collection.',
    },
    {
        'caid_source': 'event_type:bulk_export',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': 'Bulk export event directly evidences collection.',
    },
    {
        'caid_source': 'event_type:student_export',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': 'Student data export evidences collection.',
    },
    {
        'caid_source': 'event_type:course_export',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': 'Course data export evidences collection.',
    },
    {
        'caid_source': 'event_type:grade_download',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': 'Grade download evidences collection of sensitive data.',
    },
    {
        'caid_source': 'event_type:any_sensitive',
        'technique_id': 'T1567',
        'state': 'potential',
        'rationale': ('Sensitive operations may be exfiltration staging; '
                      'no egress evidence in CAID logs.'),
    },
    {
        'caid_source': 'event_type:login_failed',
        'technique_id': 'T1110',
        'state': 'potential',
        'rationale': 'Failed authentications resemble brute force attempts.',
    },
    # --- Correlation finding types ----------------------------------------
    {
        'caid_source': 'correlation:repeated_sensitive_operations',
        'technique_id': 'T1213',
        'state': 'confirmed',
        'rationale': ('Repeated sensitive operations across sessions '
                      'corroborate systematic collection.'),
    },
    {
        'caid_source': 'correlation:repeated_sensitive_operations',
        'technique_id': 'T1567',
        'state': 'potential',
        'rationale': 'Repeated sensitive ops may stage exfiltration.',
    },
    {
        'caid_source': 'correlation:similar_account_behavior',
        'technique_id': 'T1078',
        'state': 'potential',
        'rationale': ('Similar behavior across accounts may indicate '
                      'shared valid credentials.'),
    },
    {
        'caid_source': 'correlation:long_term_behavior_change',
        'technique_id': 'T1078',
        'state': 'potential',
        'rationale': ('Behavior drift may reflect compromised valid '
                      'accounts.'),
    },
    {
        'caid_source': 'correlation:shared_ip_activity',
        'technique_id': 'T1090',
        'state': 'potential',
        'rationale': ('Corroborated shared-IP activity may indicate '
                      'proxy/VPN infrastructure; shared IP alone is '
                      'never sufficient.'),
    },
]

# Sources explicitly considered and left unmapped (reported, not guessed).
UNMAPPED_SOURCES = [
    'rule:high_api_requests',
    'rule:high_data_transfer_volume_only',
    'correlation:repeated_pattern_across_sessions',
    'correlation:shared_ip_activity_un corroborated',
]

_STATE_RANK = {'unmapped': 0, 'potential': 1, 'confirmed': 2}


@dataclass
class Technique:
    technique_id: str
    name: str
    description: str = ''
    tactics: List[str] = field(default_factory=list)
    is_subtechnique: bool = False
    url: str = ''
    deprecated: bool = False


class MitreAttackStore:
    """Loads, caches, and queries the MITRE ATT&CK Enterprise bundle."""

    def __init__(self, cache_dir: Optional[str] = None,
                 fetcher: Optional[Callable[[], bytes]] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.bundle_path = os.path.join(self.cache_dir, BUNDLE_FILENAME)
        self.meta_path = os.path.join(self.cache_dir, META_FILENAME)
        self._fetcher = fetcher or self._http_fetch
        self.available = False
        self.load_error: Optional[str] = None
        self.framework_version: Optional[str] = None
        self.retrieved_at: Optional[str] = None
        self.techniques: Dict[str, Technique] = {}
        self.tactics: Dict[str, str] = dict(TACTIC_NAMES)
        self.matrix: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # Loading / caching
    # ------------------------------------------------------------------

    @staticmethod
    def _http_fetch() -> bytes:
        req = urllib.request.Request(
            STIX_URL, headers={'User-Agent': 'CAID-mitre-loader'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()

    def ensure_data(self, allow_download: bool = True) -> bool:
        """Ensure bundle+meta exist locally, then parse. Returns success."""
        if os.path.exists(self.bundle_path) and self._load():
            return True
        if allow_download:
            try:
                self._download()
            except Exception as exc:
                self.load_error = f'download failed: {exc}'
                # Fall back to any stale cache.
                if os.path.exists(self.bundle_path) and self._load():
                    return True
                return False
            if self._load():
                return True
        elif os.path.exists(self.bundle_path):
            return self._load()
        self.load_error = self.load_error or 'no cached MITRE data'
        return False

    def _download(self) -> None:
        raw = self._fetcher()
        sha256 = hashlib.sha256(raw).hexdigest()
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.bundle_path, 'wb') as fh:
            fh.write(raw)
        meta = {
            'source_url': STIX_URL,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'bytes': len(raw),
            'sha256': sha256,
        }
        try:
            bundle = json.loads(raw.decode('utf-8'))
            meta['stix_version'] = bundle.get('spec_version')
            meta['id'] = bundle.get('id')
        except Exception:
            pass
        with open(self.meta_path, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2)
        self._read_meta(meta)

    def _read_meta(self, meta: Optional[Dict] = None) -> None:
        if meta is None and os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as fh:
                    meta = json.load(fh)
            except Exception:
                meta = None
        if meta:
            self.retrieved_at = meta.get('retrieved_at')
            ver = meta.get('x_mitre_version') or meta.get('version')
            if ver and not self.framework_version:
                self.framework_version = ver
        # ATT&CK bundles carry version in spec_version/objects; the
        # enterprise-attack bundle id embeds the version, e.g.
        # attack-pattern--... ; use 'spec_version' fallback below.
        if not self.framework_version:
            self.framework_version = 'enterprise-attack (STIX 2.1)'

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _load(self) -> bool:
        # Integrity check: if the meta records a SHA-256, verify the
        # cached bundle matches it. A mismatch indicates tampering or
        # corruption → refuse to load and surface the error.
        expected = None
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, 'r', encoding='utf-8') as fh:
                    expected = json.load(fh).get('sha256')
            except Exception:
                expected = None
        try:
            with open(self.bundle_path, 'rb') as fh:
                raw = fh.read()
        except Exception as exc:
            self.load_error = f'cache unreadable: {exc}'
            return False
        if expected:
            actual = hashlib.sha256(raw).hexdigest()
            if actual != expected:
                self.load_error = (
                    'bundle integrity check failed: sha256 mismatch '
                    f'(expected {expected[:12]}…, got {actual[:12]}…). '
                    'Re-download the MITRE dataset from the Settings page.')
                return False
        try:
            bundle = json.loads(raw.decode('utf-8'))
        except Exception as exc:
            self.load_error = f'cache unreadable: {exc}'
            return False
        objects = bundle.get('objects') if isinstance(bundle, dict) else None
        if not isinstance(objects, list):
            self.load_error = 'bundle has no objects list'
            return False
        self.techniques = {}
        self.matrix = {name: [] for name in TACTIC_NAMES.values()}
        versions = []
        for obj in objects:
            if not isinstance(obj, dict) or obj.get('type') != 'attack-pattern':
                continue
            ver = obj.get('x_mitre_attack_spec_version')
            if isinstance(ver, str) and ver:
                versions.append(ver)
            if obj.get('x_mitre_revoked') or obj.get('revoked') \
                    or obj.get('x_mitre_deprecated'):
                continue
            tid = self._technique_id(obj)
            if not tid:
                continue
            tactics = []
            for phase in obj.get('kill_chain_phases') or []:
                if not isinstance(phase, dict):
                    continue
                if phase.get('kill_chain_name') != 'mitre-attack':
                    continue
                slug = phase.get('phase_name')
                name = TACTIC_NAMES.get(slug, slug.replace('-', ' ').title())
                if name not in tactics:
                    tactics.append(name)
                    self.matrix.setdefault(name, []).append(tid)
            tech = Technique(
                technique_id=tid,
                name=str(obj.get('name') or tid),
                description=str(obj.get('description') or ''),
                tactics=tactics,
                is_subtechnique=bool(obj.get('x_mitre_is_subtechnique')),
                url=self._technique_url(obj, tid),
            )
            self.techniques[tid] = tech
        if versions:
            try:
                self.framework_version = (
                    'ATT&CK Enterprise (STIX spec '
                    f'{max(versions, key=lambda v: [int(p) for p in v.split(".")])})')
            except Exception:
                self.framework_version = f'ATT&CK Enterprise ({versions[0]})'
        self._read_meta()
        self.available = bool(self.techniques)
        if not self.available:
            self.load_error = 'bundle contained no usable techniques'
        return self.available

    @staticmethod
    def _external_references(obj: Dict) -> List[Dict]:
        refs = obj.get('external-references') or obj.get('external_references')
        return [r for r in (refs or []) if isinstance(r, dict)]

    @classmethod
    def _technique_id(cls, obj: Dict) -> Optional[str]:
        for ref in cls._external_references(obj):
            if ref.get('source_name') == 'mitre-attack':
                ext = ref.get('external_id')
                if isinstance(ext, str) and ext.upper().startswith('T'):
                    return ext.upper()
        return None

    @classmethod
    def _technique_url(cls, obj: Dict, technique_id: str = '') -> str:
        for ref in cls._external_references(obj):
            if ref.get('url'):
                return str(ref['url'])
        if technique_id:
            return (f'https://attack.mitre.org/techniques/'
                    f'{technique_id.replace(".", "/")}/')
        return ''

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_technique(self, technique_id: str) -> Optional[Technique]:
        return self.techniques.get(str(technique_id).upper())

    def list_techniques(self, tactic: Optional[str] = None) -> List[Technique]:
        techs = sorted(self.techniques.values(),
                       key=lambda t: t.technique_id)
        if tactic:
            techs = [t for t in techs if tactic in t.tactics]
        return techs

    def technique_url(self, technique_id: str) -> str:
        tech = self.get_technique(technique_id)
        if tech and tech.url:
            return tech.url
        return (f'https://attack.mitre.org/techniques/'
                f'{technique_id.replace(".", "/")}/')

    # ------------------------------------------------------------------
    # CAID mapping
    # ------------------------------------------------------------------

    def map_event(self, row: Dict) -> List[Dict]:
        """Map a scored event row to ATT&CK techniques."""
        mappings: List[Dict] = []
        event_type = str(row.get('event_type') or '')
        sources = []
        if event_type == 'login' and str(row.get('auth_result') or '') \
                == 'failure':
            sources.append('event_type:login_failed')
        if event_type in SENSITIVE_EVENT_TYPES:
            sources.append(f'event_type:{event_type}')
            sources.append('event_type:any_sensitive')
        # Detection rules that fired on this event.
        for rule in row.get('triggered_rules') or []:
            rid = rule.get('rule_id') if isinstance(rule, dict) else None
            if rid:
                sources.append(f'rule:{rid}')
            name = rule.get('name') if isinstance(rule, dict) else None
            if name:
                sources.append('rule:' + str(name))
        # Fall back to rule name strings if triggered_rules absent.
        if not sources and row.get('triggered_rules'):
            pass
        return self._resolve_sources(sources, evidence={
            'event_type': event_type,
            'user_id': row.get('user_id'),
            'timestamp': str(row.get('timestamp') or ''),
            'combined_score': row.get('combined_score'),
            'risk_level': row.get('risk_level'),
        }, caid_kind='event')

    def map_finding(self, finding: Dict) -> List[Dict]:
        """Map a correlation finding to ATT&CK techniques."""
        ctype = str(finding.get('correlation_type') or '')
        shared_ip_only = bool(finding.get('shared_ip_only'))
        sources = []
        if ctype == 'shared_ip_activity' and shared_ip_only:
            return []  # shared IP alone is never a technique mapping
        if ctype == 'shared_ip_activity':
            sources.append('correlation:shared_ip_activity')
        elif ctype:
            sources.append(f'correlation:{ctype}')
        return self._resolve_sources(sources, evidence={
            'correlation_id': finding.get('correlation_id'),
            'correlation_type': ctype,
            'entities': finding.get('entities') or [],
            'sessions': finding.get('sessions') or [],
            'correlation_score': finding.get('correlation_score'),
        }, caid_kind='correlation_finding')

    def map_incident(self, incident: Dict) -> List[Dict]:
        """Aggregate technique mappings across an incident's findings."""
        merged: Dict[str, Dict] = {}
        for finding in incident.get('findings') or []:
            for m in self.map_finding(finding):
                tid = m['technique_id']
                prev = merged.get(tid)
                if prev is None or _STATE_RANK[m['confidence']] > \
                        _STATE_RANK[prev['confidence']]:
                    merged[tid] = m
        return sorted(merged.values(),
                      key=lambda m: (-_STATE_RANK[m['confidence']],
                                     m['technique_id']))

    def _resolve_sources(self, sources: List[str], evidence: Dict,
                         caid_kind: str) -> List[Dict]:
        """Resolve curated sources to technique mappings (deduped)."""
        out: Dict[str, Dict] = {}
        for src in sources:
            for rule in CAID_MITRE_MAP:
                if rule['caid_source'] != src:
                    continue
                tid = rule['technique_id']
                tech = self.get_technique(tid)
                entry = {
                    'technique_id': tid,
                    'technique_name': tech.name if tech else tid,
                    'tactic': (tech.tactics[0] if tech and tech.tactics
                               else ''),
                    'caid_source': src,
                    'caid_kind': caid_kind,
                    'evidence': evidence,
                    'rationale': rule['rationale'],
                    'confidence': rule['state'],
                    'mapping_source': MAPPING_SOURCE,
                    'framework_version': self.framework_version or 'unknown',
                }
                prev = out.get(tid)
                if prev is None or _STATE_RANK[entry['confidence']] > \
                        _STATE_RANK[prev['confidence']]:
                    out[tid] = entry
        return sorted(out.values(),
                      key=lambda m: (-_STATE_RANK[m['confidence']],
                                     m['technique_id']))

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------

    def coverage(self) -> Dict:
        """Mapping coverage stats for the MITRE dashboard."""
        mapped_sources = {r['caid_source'] for r in CAID_MITRE_MAP}
        rule_ids = {r['caid_source'].split(':', 1)[1]
                    for r in CAID_MITRE_MAP if r['caid_source'].startswith('rule:')}
        known_rules = {
            'high_api_requests', 'after_hours_activity',
            'bulk_download_pattern', 'high_data_transfer',
            'multiple_accounts_accessed', 'old_token_activity',
            'critical_access_pattern',
        }
        mapped_techniques = {r['technique_id'] for r in CAID_MITRE_MAP}
        per_tactic: Dict[str, int] = {}
        for tid in mapped_techniques:
            tech = self.get_technique(tid)
            for tac in (tech.tactics if tech else []):
                per_tactic[tac] = per_tactic.get(tac, 0) + 1
        return {
            'available': self.available,
            'framework_version': self.framework_version,
            'retrieved_at': self.retrieved_at,
            'total_techniques': len(self.techniques),
            'mapped_rules': sorted(rule_ids & known_rules),
            'unmapped_rules': sorted(known_rules - rule_ids),
            'mapped_techniques': sorted(mapped_techniques),
            'potential_mappings': sum(
                1 for r in CAID_MITRE_MAP if r['state'] == 'potential'),
            'confirmed_mappings': sum(
                1 for r in CAID_MITRE_MAP if r['state'] == 'confirmed'),
            'techniques_by_tactic': per_tactic,
            'unmapped_sources': UNMAPPED_SOURCES,
        }
