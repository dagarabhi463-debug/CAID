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
]

_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?(previous|prior|above)\s+instructions',
    r'\b(system|assistant|user)\s*:',
    r'<\|?(system|im_start|im_end|endoftext)\|?>',
    r'\[/?INST\]',
    r'disregard\s+(your|all)\s+(rules|instructions)',
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
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:max_len]


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
        if pd.isna(val):
            continue
        if hasattr(val, 'item'):
            try:
                val = val.item()
            except Exception:
                pass
        facts[col] = sanitize_for_prompt(val, 200)
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
]


def match_threat_hunt(question: str) -> Optional[Dict]:
    """Match a natural-language question to a safe predefined query.

    Returns the query spec or None. Never executes generated code.
    """
    q = (question or '').lower()
    if not q.strip():
        return None
    best, best_score = None, 0
    for spec in THREAT_HUNT_QUERIES:
        score = sum(1 for kw in spec['keywords'] if kw in q)
        if score > best_score:
            best, best_score = spec, score
    return best if best_score > 0 else None
