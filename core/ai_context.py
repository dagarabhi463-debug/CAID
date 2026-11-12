"""
Grounded context builder for CAID's AI analyst.

Every prompt sent to Ollama is assembled from *actual* CAID detection
data: scored event fields, structured evidence, correlation findings,
and incident records. This module also sanitizes log-derived strings
against prompt injection and provides a registry of safe,
predefined threat-hunting queries — the AI never executes generated
code, SQL, or shell commands.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional

import pandas as pd

# Event fields that are safe to quote to the model (no free-text blobs).
_EVENT_FACT_FIELDS = [
    'timestamp', 'user_id', 'session_id', 'source_ip', 'event_type',
    'role', 'auth_result', 'is_anomaly', 'anomaly_score_normalized',
    'anomaly_score', 'behavioral_deviation_score', 'sequence_deviation',
    'combined_score', 'risk_level', 'operating_mode', 'detection_id',
    'event_id',
]

_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier|'
    r'preceding|foregoing)\s+\w*\s*(instructions?|rules?|prompts?|'
    r'directives?|context)',
    r'\b(system|assistant|user)\s*:',
    r'<\|?(system|im_start|im_end|endoftext|assistant)\|?>',
    r'\[/?INST\]',
    r'disregard\s+(your|all|the|any)\s+(rules?|instructions?|'
    r'directives?|guidelines?)',
    r'\byou\s+are\s+now\b',
    r'\bfrom\s+now\s+on\b.{0,40}\b(you\s+are|act|behave|respond)',
    r'\bnew\s+(system\s+)?instructions?\b',
    r'\boverride\s+(your|all|the|any|previous)\b',
    r'\bjailbreak\b',
    r'\bdo\s+anything\s+now\b',
    r'\bDAN\s+mode\b',
    r'\breveal\s+(the\s+|your\s+)?(system\s+|hidden\s+|initial\s+)?'
    r'(prompt|instructions?|directives?)\b',
    r'\b(show|print|output|repeat|display)\s+(me\s+)?(the\s+|your\s+)?'
    r'(system\s+|hidden\s+|initial\s+)?(prompt|instructions?)\b',
    r'\bpretend\s+(you\s+are|to\s+be|that)\b',
    r'\bact\s+as\s+(a|an|if|though|the)\b',
    r'\bforget\s+(everything|all|your|previous)\b',
    r'\bno\s+longer\s+(bound|restricted|limited)\b',
]
_INJECTION_RE = re.compile('|'.join(_INJECTION_PATTERNS), re.IGNORECASE)


def sanitize_for_prompt(value, max_len: int = 500) -> str:
    """Neutralize prompt-injection patterns in log-derived text.

    Removes control characters, truncates, and masks common injection
    phrases. The result is wrapped in ``<<DATA>>`` delimiters by
    ``wrap_untrusted`` so the model treats it as data, not instructions.
    """
    if value is None:
        return ''
    if isinstance(value, (list, tuple, set)):
        value = ', '.join(str(v) for v in value)
    if isinstance(value, dict):
        import json
        try:
            value = json.dumps(value, default=str)
        except Exception:
            value = str(value)
    text = str(value)
    text = ''.join(ch for ch in text if ch == '\n' or ch == '\t'
                   or ord(ch) >= 32)
    text = _INJECTION_RE.sub('[filtered]', text)
    # Delimiter-breakout hardening: untrusted content must never be able
    # to forge the <<UNTRUSTED_DATA>>/...<<END_UNTRUSTED_DATA>> markers
    # (or the legacy <<DATA>> markers) that wrap_untrusted relies on.
    text = re.sub(r'<<.*?>>', '[delimiter]', text)
    text = re.sub(r'UNTRUSTED_DATA|END_UNTRUSTED_DATA', '[tag]', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]


def detect_injection_attempt(text: str) -> bool:
    """True when ``text`` contains a known prompt-injection pattern.

    Used to flag (not block) analyst-supplied questions so callers can
    surface a warning alongside the sanitized, still-answered request.
    """
    return bool(_INJECTION_RE.search(str(text or '')))


def wrap_untrusted(label: str, content: str) -> str:
    """Wrap untrusted content in explicit data delimiters."""
    return (f'<<UNTRUSTED_DATA source="{sanitize_for_prompt(label, 80)}">\n'
            f'{content}\n<<END_UNTRUSTED_DATA>>')


def _row_facts(row: pd.Series) -> Dict:
    facts = {}
    for col in _EVENT_FACT_FIELDS:
        if col not in row.index:
            continue
        val = row[col]
        if not isinstance(val, (list, tuple, set, dict)):
            try:
                if pd.isna(val):
                    continue
            except (TypeError, ValueError):
                pass
        if hasattr(val, 'item'):
            try:
                val = val.item()
            except Exception:
                pass
        facts[col] = sanitize_for_prompt(val, 200)
    # Guaranteed traceable reference even when detection_id is absent.
    ref = row.get('detection_id') or row.get('event_id')
    if ref is None or (isinstance(ref, float) and pd.isna(ref)):
        ref = f"row_{row.name}" if row.name is not None else ''
    if ref not in (None, ''):
        facts['event_ref'] = sanitize_for_prompt(ref, 120)
    return facts


def event_context(row: pd.Series,
                  scored_df: Optional[pd.DataFrame] = None) -> Dict:
    """Build a grounded evidence bundle for a single event."""
    facts = _row_facts(row)
    evidence: List[str] = []
    limitations: List[str] = []

    expl = row.get('detection_explanation') if 'detection_explanation' in row.index else None
    if expl and not pd.isna(expl):
        evidence.append(f"DetectionEngine explanation: "
                        f"{sanitize_for_prompt(expl, 600)}")
    rules = row.get('triggered_rules') if 'triggered_rules' in row.index else None
    if rules:
        names = [r.get('name') or r.get('rule_id')
                 for r in rules if isinstance(r, dict)]
        if names:
            evidence.append("Triggered rules: " + ", ".join(
                sanitize_for_prompt(n, 80) for n in names))
    sev = row.get('structured_evidence') if 'structured_evidence' in row.index else None
    if sev:
        evidence.append("Structured evidence: " +
                        sanitize_for_prompt(sev, 800))

    # Related session activity (same session, up to 10 events).
    session_events = 0
    if scored_df is not None and 'session_id' in row.index \
            and 'session_id' in scored_df.columns:
        sid = row['session_id']
        sub = scored_df[scored_df['session_id'].astype(str)
                        == str(sid)]
        session_events = int(len(sub))
        if len(sub) > 1:
            evidence.append(
                f"Session {sanitize_for_prompt(sid, 80)} contains "
                f"{session_events} events in the analyzed dataset.")
    else:
        limitations.append("No session-level context available.")

    score = row.get('combined_score') if 'combined_score' in row.index else None
    if score is not None and not pd.isna(score):
        evidence.append(f"Combined score {float(score):.3f}, "
                        f"risk level {sanitize_for_prompt(row.get('risk_level'), 40)}.")
    else:
        limitations.append("No combined score recorded for this event.")

    return {'facts': facts, 'evidence': evidence,
            'limitations': limitations or ['None identified.'],
            'session_event_count': session_events}


def finding_context(finding: Dict) -> Dict:
    """Build a grounded evidence bundle for a correlation finding."""
    import json
    evidence = []
    ev_list = finding.get('evidence') or []
    if ev_list:
        evidence.append("Finding evidence: " + sanitize_for_prompt(
            json.dumps(ev_list, default=str), 1200))
    facts = {
        'correlation_id': sanitize_for_prompt(finding.get('correlation_id'), 80),
        'correlation_type': sanitize_for_prompt(finding.get('correlation_type'), 80),
        'entities': sanitize_for_prompt(finding.get('entities'), 200),
        'sessions': sanitize_for_prompt(
            (finding.get('sessions') or [])[:10], 300),
        'correlation_score': finding.get('correlation_score'),
        'shared_ip_only': bool(finding.get('shared_ip_only')),
    }
    limitations = []
    if facts['shared_ip_only']:
        limitations.append(
            "Shared-IP-only finding: insufficient evidence of "
            "coordination; treat as a low-confidence lead.")
    if finding.get('operating_note'):
        evidence.append("Operating note: " + sanitize_for_prompt(
            finding['operating_note'], 300))
    return {'facts': facts, 'evidence': evidence,
            'limitations': limitations or ['None identified.']}


def incident_context(incident: Dict) -> Dict:
    """Build a grounded evidence bundle for an incident."""
    import json
    facts = {
        'incident_id': sanitize_for_prompt(incident.get('incident_id'), 80),
        'title': sanitize_for_prompt(incident.get('title'), 200),
        'severity': sanitize_for_prompt(incident.get('severity'), 40),
        'status': sanitize_for_prompt(incident.get('status'), 40),
        'entity_users': sanitize_for_prompt(
            incident.get('entity_users'), 300),
        'entity_sessions_count': len(incident.get('entity_sessions') or []),
        'entity_ips': sanitize_for_prompt(incident.get('entity_ips'), 300),
        'finding_count': len(incident.get('findings') or []),
        'finding_types': sanitize_for_prompt(
            sorted({str(f.get('correlation_type'))
                    for f in (incident.get('findings') or [])
                    if f.get('correlation_type')}), 300),
    }
    evidence = []
    for f in (incident.get('findings') or [])[:10]:
        evidence.append(
            f"Finding {sanitize_for_prompt(f.get('correlation_id'), 60)}: "
            f"{sanitize_for_prompt(f.get('correlation_type'), 60)} "
            f"score={f.get('correlation_score')}")
    notes = incident.get('notes') or []
    if notes:
        evidence.append(f"{len(notes)} analyst note(s) on record.")
    limitations = []
    if facts['finding_count'] == 0:
        limitations.append("Incident has no linked correlation findings.")
    if any(f.get('shared_ip_only')
           for f in (incident.get('findings') or [])):
        limitations.append(
            "Contains shared-IP-only findings — low-confidence leads.")
    return {'facts': facts, 'evidence': evidence,
            'limitations': limitations or ['None identified.']}


def attack_chain_context(incident: Dict,
                         scored_df: Optional[pd.DataFrame] = None,
                         mitre_mappings: Optional[List[Dict]] = None) -> Dict:
    """Grounded evidence bundle for retrospective attack-chain analysis."""
    import json
    facts = {
        'incident_id': sanitize_for_prompt(incident.get('incident_id'), 80),
        'title': sanitize_for_prompt(incident.get('title'), 200),
        'severity': sanitize_for_prompt(incident.get('severity'), 40),
        'status': sanitize_for_prompt(incident.get('status'), 40),
        'entity_users': sanitize_for_prompt(
            incident.get('entity_users'), 300),
        'entity_sessions_count': len(incident.get('entity_sessions') or []),
        'entity_ips': sanitize_for_prompt(incident.get('entity_ips'), 300),
        'finding_count': len(incident.get('findings') or []),
        'finding_types': sanitize_for_prompt(
            sorted({str(f.get('correlation_type'))
                    for f in (incident.get('findings') or [])
                    if f.get('correlation_type')}), 300),
        'mitre_mapping_count': len(mitre_mappings or []),
    }
    evidence = []
    findings = sorted(
        incident.get('findings') or [],
        key=lambda f: float(f.get('correlation_score') or 0.0),
        reverse=True)
    for f in findings[:12]:
        evidence.append(
            f"Finding {sanitize_for_prompt(f.get('correlation_id'), 60)}: "
            f"{sanitize_for_prompt(f.get('correlation_type'), 60)} "
            f"score={f.get('correlation_score')} "
            f"shared_ip_only={bool(f.get('shared_ip_only'))}")
        ev = f.get('evidence')
        if ev:
            evidence.append("  evidence: " + sanitize_for_prompt(
                json.dumps(ev, default=str), 700))
    for m in (mitre_mappings or [])[:12]:
        evidence.append(
            f"ATT&CK {sanitize_for_prompt(m.get('technique_id'), 30)} "
            f"{sanitize_for_prompt(m.get('technique_name'), 120)} "
            f"tactic={sanitize_for_prompt(m.get('tactic'), 80)} "
            f"confidence={sanitize_for_prompt(m.get('confidence'), 40)}")

    event_count = 0
    if scored_df is not None and len(scored_df):
        users = set(map(str, incident.get('entity_users') or []))
        sessions = set(map(str, incident.get('entity_sessions') or []))
        ips = set(map(str, incident.get('entity_ips') or []))
        mask = pd.Series(False, index=scored_df.index)
        if users and 'user_id' in scored_df.columns:
            mask |= scored_df['user_id'].astype(str).isin(users)
        if sessions and 'session_id' in scored_df.columns:
            mask |= scored_df['session_id'].astype(str).isin(sessions)
        if ips and 'source_ip' in scored_df.columns:
            mask |= scored_df['source_ip'].astype(str).isin(ips)
        sub = scored_df[mask]
        event_count = int(len(sub))
        if event_count:
            evidence.append(
                f"Matched {event_count} scored events for incident entities.")
            if 'combined_score' in sub.columns:
                sub = sub.assign(combined_score=pd.to_numeric(
                    sub['combined_score'], errors='coerce'))
            top = sub.sort_values(
                'combined_score' if 'combined_score' in sub.columns
                else sub.columns[0], ascending=False).head(8)
            for _, r in top.iterrows():
                evidence.append(
                    "top_event: " + sanitize_for_prompt({
                        'timestamp': r.get('timestamp'),
                        'user_id': r.get('user_id'),
                        'session_id': r.get('session_id'),
                        'source_ip': r.get('source_ip'),
                        'event_type': r.get('event_type'),
                        'combined_score': r.get('combined_score'),
                        'risk_level': r.get('risk_level'),
                    }, 500))
        else:
            evidence.append("No scored events matched incident entities.")
    else:
        event_count = 0

    limitations = []
    if facts['finding_count'] == 0:
        limitations.append("Incident has no linked correlation findings.")
    if any(f.get('shared_ip_only') for f in findings):
        limitations.append(
            "Contains shared-IP-only findings — low-confidence leads, "
            "not proof of coordination.")
    if event_count == 0:
        limitations.append("No event-level evidence matched incident entities.")
    if not mitre_mappings:
        limitations.append("No MITRE ATT&CK mappings available for this incident.")
    else:
        limitations.append(
            "ATT&CK mappings are potential/defensive research mappings, "
            "not confirmed attacker behavior.")
    limitations.append(
        "Attack-chain reconstruction is retrospective and does not alter "
        "detection scores, thresholds, or incident state.")
    return {'facts': facts, 'evidence': evidence,
            'limitations': limitations or ['None identified.'],
            'matched_event_count': event_count}


# ----------------------------------------------------------------------
# Safe threat-hunting queries
# ----------------------------------------------------------------------

def _q_repeated_suspicious_sessions(df: pd.DataFrame) -> pd.DataFrame:
    if not {'user_id', 'session_id', 'combined_score'}.issubset(df.columns):
        return pd.DataFrame()
    susp = df[pd.to_numeric(df['combined_score'], errors='coerce') >= 0.5]
    counts = susp.groupby(['user_id', 'session_id']).size().reset_index(
        name='suspicious_events')
    return counts[counts['suspicious_events'] >= 2].sort_values(
        'suspicious_events', ascending=False)


def _q_failed_auth(df: pd.DataFrame) -> pd.DataFrame:
    if 'auth_result' not in df.columns:
        return pd.DataFrame()
    fails = df[df['auth_result'].astype(str).str.lower() == 'failure']
    if 'user_id' not in fails.columns:
        return fails
    return fails.groupby('user_id').size().reset_index(
        name='failed_logins').sort_values('failed_logins', ascending=False)


def _q_sensitive_ops(df: pd.DataFrame) -> pd.DataFrame:
    sensitive = {'grade_download', 'student_export', 'course_export',
                 'bulk_download', 'bulk_export'}
    if 'event_type' not in df.columns:
        return pd.DataFrame()
    sub = df[df['event_type'].isin(sensitive)]
    cols = [c for c in ['user_id', 'event_type', 'timestamp',
                        'combined_score', 'risk_level'] if c in sub.columns]
    return sub[cols].sort_values(
        'combined_score' if 'combined_score' in sub.columns else cols[0],
        ascending=False) if cols else pd.DataFrame()


def _q_high_risk_users(df: pd.DataFrame) -> pd.DataFrame:
    if not {'user_id', 'combined_score'}.issubset(df.columns):
        return pd.DataFrame()
    g = df.groupby('user_id').agg(
        events=('combined_score', 'size'),
        max_score=('combined_score', 'max'),
        mean_score=('combined_score', 'mean'),
    ).reset_index()
    return g[g['max_score'] >= 0.7].sort_values('max_score',
                                               ascending=False)


def _q_ip_findings(df: pd.DataFrame) -> pd.DataFrame:
    if 'source_ip' not in df.columns or 'combined_score' not in df.columns:
        return pd.DataFrame()
    susp = df[pd.to_numeric(df['combined_score'], errors='coerce') >= 0.5]
    if 'user_id' in susp.columns:
        return susp.groupby('source_ip').agg(
            users=('user_id', 'nunique'), events=('user_id', 'size'),
            max_score=('combined_score', 'max')).reset_index().sort_values(
            'max_score', ascending=False)
    return susp.groupby('source_ip').size().reset_index(name='events')


def _q_after_hours_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    sensitive = {'grade_download', 'student_export', 'course_export',
                 'bulk_download', 'bulk_export'}
    if not {'event_type', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    mask = df['event_type'].isin(sensitive)
    if 'after_hours' in df.columns:
        mask &= pd.to_numeric(df['after_hours'], errors='coerce').fillna(0) != 0
    sub = df[mask]
    if 'combined_score' in sub.columns:
        return sub.groupby('user_id').agg(
            events=('event_type', 'size'),
            max_score=('combined_score', 'max')).reset_index().sort_values(
            'max_score', ascending=False)
    return sub.groupby('user_id').size().reset_index(name='events')


def _q_auth_failure_bursts(df: pd.DataFrame) -> pd.DataFrame:
    if not {'auth_result', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    fails = df[df['auth_result'].astype(str).str.lower() == 'failure']
    if fails.empty:
        return fails
    out = fails.groupby(['user_id', 'session_id']).size().reset_index(
        name='failed_events') if 'session_id' in fails.columns else \
        fails.groupby('user_id').size().reset_index(name='failed_events')
    return out[out['failed_events'] >= 3].sort_values(
        'failed_events', ascending=False)


def _q_shared_ip_multiuser(df: pd.DataFrame) -> pd.DataFrame:
    if not {'source_ip', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    if 'combined_score' in df.columns:
        susp = df[pd.to_numeric(df['combined_score'], errors='coerce') >= 0.4]
    else:
        susp = df
    agg = {'users': ('user_id', 'nunique'), 'events': ('user_id', 'size')}
    if 'combined_score' in susp.columns:
        agg['max_score'] = ('combined_score', 'max')
    g = susp.groupby('source_ip').agg(**agg).reset_index()
    return g[g['users'] >= 2].sort_values('users', ascending=False)


def _q_sensitive_breadth(df: pd.DataFrame) -> pd.DataFrame:
    sensitive = {'grade_download', 'student_export', 'course_export',
                 'bulk_download', 'bulk_export'}
    if not {'event_type', 'user_id'}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[df['event_type'].isin(sensitive)]
    if sub.empty:
        return sub
    g = sub.groupby('user_id').agg(
        sensitive_events=('event_type', 'size'),
        unique_sensitive=('event_type', 'nunique')).reset_index()
    return g[(g['sensitive_events'] >= 3) &
             (g['unique_sensitive'] >= 2)].sort_values(
        'sensitive_events', ascending=False)


THREAT_HUNT_QUERIES: List[Dict] = [
    {
        'id': 'repeated_suspicious_sessions',
        'label': 'Users with repeated suspicious sessions',
        'keywords': ['repeated', 'suspicious', 'session'],
        'fn': _q_repeated_suspicious_sessions,
        'describe': lambda r: (
            f"{len(r)} user/session pairs with ≥2 suspicious events"),
    },
    {
        'id': 'failed_auth',
        'label': 'Accounts with failed authentication events',
        'keywords': ['failed', 'auth', 'login', 'credential'],
        'fn': _q_failed_auth,
        'describe': lambda r: f"{len(r)} accounts with failed logins",
    },
    {
        'id': 'sensitive_ops',
        'label': 'Sensitive operations summary',
        'keywords': ['sensitive', 'export', 'download', 'grade', 'bulk'],
        'fn': _q_sensitive_ops,
        'describe': lambda r: f"{len(r)} sensitive-operation events",
    },
    {
        'id': 'high_risk_users',
        'label': 'High-risk users (max combined score ≥ 0.7)',
        'keywords': ['high', 'risk', 'user', 'account'],
        'fn': _q_high_risk_users,
        'describe': lambda r: f"{len(r)} high-risk accounts",
    },
    {
        'id': 'ip_activity',
        'label': 'Suspicious activity by source IP',
        'keywords': ['ip', 'shared', 'address'],
        'fn': _q_ip_findings,
        'describe': lambda r: f"{len(r)} source IPs with suspicious events",
    },
    {
        'id': 'after_hours_sensitive',
        'label': 'After-hours sensitive operations',
        'keywords': ['after', 'hours', 'sensitive', 'night', 'late'],
        'fn': _q_after_hours_sensitive,
        'describe': lambda r: f"{len(r)} users with after-hours sensitive ops",
    },
    {
        'id': 'auth_failure_bursts',
        'label': 'Authentication failure bursts',
        'keywords': ['auth', 'failure', 'burst', 'brute', 'login'],
        'fn': _q_auth_failure_bursts,
        'describe': lambda r: f"{len(r)} user/session auth-failure bursts",
    },
    {
        'id': 'shared_ip_multiuser',
        'label': 'Shared IPs used by multiple users',
        'keywords': ['shared', 'ip', 'multi', 'users', 'address'],
        'fn': _q_shared_ip_multiuser,
        'describe': lambda r: f"{len(r)} shared IPs with multiple users",
    },
    {
        'id': 'sensitive_breadth',
        'label': 'Users with broad sensitive-operation activity',
        'keywords': ['sensitive', 'breadth', 'export', 'download', 'bulk'],
        'fn': _q_sensitive_breadth,
        'describe': lambda r: f"{len(r)} users with broad sensitive activity",
    },
]


def match_threat_hunt(question: str) -> Optional[Dict]:
    """Match a natural-language question to a safe predefined query.

    Returns the query spec or None. Never executes generated code.
    """
    q = (question or '').lower()
    if not q.strip():
        return None

    def _kw_hit(kw: str) -> bool:
        # Short keywords (e.g. 'ip') need word boundaries so they don't
        # match inside unrelated words; longer ones use substring match.
        if len(kw) <= 3:
            return re.search(r'\b' + re.escape(kw) + r'\b', q) is not None
        return kw in q

    best, best_score = None, 0
    for spec in THREAT_HUNT_QUERIES:
        score = sum(1 for kw in spec['keywords'] if _kw_hit(kw))
        if score > best_score:
            best, best_score = spec, score
    return best if best_score > 0 else None
