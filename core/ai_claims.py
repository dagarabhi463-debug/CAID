"""
Claim-level grounding checks for CAID's AI analyst.

Shared by ``core.ai_analyst`` (runtime advisory warnings on responses)
and ``core.ai_evaluation`` (benchmark scoring) so both apply identical
fabrication, unsupported-number, negation, and ATT&CK-technique logic.

All checks are deterministic heuristics — they bound known failure
modes but do not prove a response is accurate.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

# Tokens that look like concrete identifiers. Anything matching these in
# a response must also appear in the evidence the model was allowed to
# see, otherwise it is a fabricated entity.
_IDENTIFIER_RES = [
    re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b'),               # IPv4
    re.compile(r'\b(?:user|usr|account)[-_][A-Za-z0-9_-]+', re.I),
    re.compile(r'\b(?:sess|session|inc|incident|corr|correlation|det|evt)[-_][A-Za-z0-9_-]+', re.I),
    re.compile(r'\bT\d{4}(?:\.\d{3})?\b'),                    # ATT&CK technique
    re.compile(r'\bCVE-\d{4}-\d{4,7}\b', re.I),               # CVE id
    re.compile(r'\b[a-f0-9]{12,}\b'),                         # long hex ids
    re.compile(r'\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b'),   # MAC address
    re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b'),               # email
    re.compile(r'\b(?:https?://|www\.)[^\s\'"<>]+', re.I),    # URL
    re.compile(r'\b(?:[A-Za-z]:\\|/)[\w./\\-]{3,}'),          # absolute path
]

_GENERIC_TOKENS = {f'e{i}' for i in range(1, 21)}

_TECHNIQUE_RE = re.compile(r'\bT\d{4}(?:\.\d{3})?\b')

_ATTRIBUTION_RE = re.compile(
    r'\bconfirmed\s+(?:a\s+|an\s+|the\s+)?'
    r'(attacker|attack|compromise|breach|intrusion|threat actor|'
    r'malicious activity|malware|att&ck|technique)\b', re.IGNORECASE)

# Numbers worth checking: decimals (scores) or integers >= 100. Small
# ints are allowed because the model legitimately derives counts.
_NUMBER_RE = re.compile(
    r'(?<![\w.])-?\d+\.\d+|(?<![\w.])-?\d{3,}(?![\d.])')

_NEGATION_CUES = ('not ', 'no ', 'never ', 'without', 'cannot',
                  "can't", "isn't", "aren't", 'dont', "don't",
                  'unconfirmed', 'no evidence', 'not be', 'rather than',
                  'as opposed', 'no indication', 'insufficient evidence',
                  'not enough evidence', 'unproven', 'speculative',
                  'cannot be confirmed', 'cannot confirm')


def extract_identifiers(text: str) -> List[str]:
    """Return identifier-like tokens found in ``text`` (lowercased)."""
    found = set()
    for rx in _IDENTIFIER_RES:
        for m in rx.finditer(text or ''):
            tok = m.group(0).lower()
            if tok not in _GENERIC_TOKENS:
                found.add(tok)
    return sorted(found)


def extract_technique_ids(text: str) -> List[str]:
    """Return ATT&CK technique IDs (uppercased) found in ``text``."""
    return sorted(set(_TECHNIQUE_RE.findall(text or '')))


def extract_checkable_numbers(text: str) -> List[str]:
    """Return decimals and large ints found in ``text``."""
    return [m.group(0) for m in _NUMBER_RE.finditer(text or '')]


def _norm_num(n: str) -> str:
    try:
        return f'{float(n):g}'
    except ValueError:
        return n


def numbers_supported(numbers: List[str], evidence_text: str) -> List[str]:
    """Return numbers from ``numbers`` whose normalized value does not
    appear among the numbers in ``evidence_text``."""
    ev_raw = set(re.findall(r'-?\d+\.?\d*', evidence_text or ''))
    supported = {_norm_num(v) for v in ev_raw}
    return [n for n in numbers if _norm_num(n) not in supported]


def fabricated_entities(response_text: str,
                        evidence_text: str) -> List[str]:
    """Identifiers in the response absent from the evidence text."""
    ev_ids = set(extract_identifiers(evidence_text or ''))
    ev_lower = (evidence_text or '').lower()
    out = []
    for tok in extract_identifiers(response_text):
        if tok in ev_ids or tok in ev_lower:
            continue
        out.append(tok)
    return out


def unsupported_technique_ids(response_text: str,
                              evidence_text: str) -> List[str]:
    """ATT&CK technique IDs cited in the response but absent from the
    evidence (i.e. techniques the model invented)."""
    ev_upper = (evidence_text or '').upper()
    return [t for t in extract_technique_ids(response_text)
            if t.upper() not in ev_upper]


def is_negated(text: str, start: int, window: int = 48) -> bool:
    """True when a match at ``start`` sits inside a negated phrase
    (e.g. 'not confirmed attacker behavior')."""
    ctx = (text or '')[max(0, start - window):start].lower()
    return any(cue in ctx for cue in _NEGATION_CUES)


def attribution_claims(text: str) -> List[str]:
    """Non-negated 'confirmed attacker/attack/compromise/...' claims."""
    hits = []
    for m in _ATTRIBUTION_RE.finditer(text or ''):
        if not is_negated(text, m.start()):
            hits.append(m.group(0))
    return hits


def context_to_text(context: Dict) -> str:
    """Flatten an ai_context bundle (facts/evidence/limitations) to the
    plain text the model was allowed to see."""
    if not isinstance(context, dict):
        return ''
    parts = [json.dumps(context.get('facts') or {}, default=str)]
    parts += [str(e) for e in (context.get('evidence') or [])]
    parts += [str(l) for l in (context.get('limitations') or [])]
    return '\n'.join(parts)


def response_claim_text(resp: Dict) -> str:
    """All analyst-visible claim text from a response dict: raw text
    plus structured observations/hypotheses/assessment."""
    parts = [resp.get('text') or '']
    s = resp.get('structured')
    if isinstance(s, dict):
        parts.append(s.get('assessment') or '')
        parts += [str(o) for o in (s.get('observations') or [])]
        parts += [str(o) for o in (s.get('derived_facts') or [])]
        parts += [str(o) for o in (s.get('hypotheses') or [])]
    return '\n'.join(parts)
