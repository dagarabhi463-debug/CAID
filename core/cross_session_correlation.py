"""
Cross-Session Threat Correlation for CAID — Milestone 2.1

Retrospective correlation layer over already-scored security events.
Consumes the output of the v2.0 pipeline (session reconstruction,
session/sequence deviation scores, DetectionEngine risk levels and
structured evidence) and links sessions, accounts, and shared-IP
activity into findings with explicit, inspectable evidence.

Design contract:
- RETROSPECTIVE ONLY. This module never feeds back into per-event
  causal scores, DetectionEngine channel weights, or risk levels.
- EVIDENCE-FIRST. Every finding carries a list of evidence dicts
  describing the observed signals, the sessions/events involved, and
  why the correlation was made.
- SHARED IP IS NEVER PROOF. Findings of type ``shared_ip_activity``
  are always emitted with ``shared_ip_only=True`` and a capped,
  low-confidence score; they escalate only when corroborated by at
  least one other signal type.
- DETERMINISTIC. Identical input produces identical finding IDs and
  ordering; no randomness, no wall-clock dependence.
- SYNTHETIC DATA ONLY. All inputs are synthetic/defensive research
  logs; no real Canvas connectivity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.session_reconstruction import (
    SessionConfig,
    SessionReconstructor,
    _is_failure,
    _norm_id,
    _norm_token,
)


@dataclass
class CorrelationConfig:
    """Thresholds and windows for cross-session correlation."""

    suspicious_score_threshold: float = 0.5   # combined_score considered elevated
    high_risk_levels: tuple = ('High', 'Critical')
    min_corroborating_signals: int = 1        # signals needed to escalate shared-IP

    pattern_min_sessions: int = 2
    deviation_repeat_threshold: float = 0.5   # deviation counted as suspicious

    similarity_threshold: float = 0.80
    similarity_min_shared_patterns: int = 1
    top_event_types: int = 8                  # event-type dims in user vectors

    sensitive_min_distinct_days: int = 2
    sensitive_min_events_per_day: int = 1

    change_min_events_per_window: int = 3
    change_threshold: float = 0.6             # normalized shift required

    shared_ip_min_users: int = 2
    shared_ip_max_score: float = 0.4          # hard cap unless corroborated

    session_config: Optional[Dict] = None
    sensitive_event_types: tuple = (
        'grade_download', 'student_export', 'course_export',
        'bulk_download', 'bulk_export',
    )

    def to_dict(self) -> Dict:
        return {
            'suspicious_score_threshold': float(self.suspicious_score_threshold),
            'high_risk_levels': list(self.high_risk_levels),
            'min_corroborating_signals': int(self.min_corroborating_signals),
            'pattern_min_sessions': int(self.pattern_min_sessions),
            'deviation_repeat_threshold': float(self.deviation_repeat_threshold),
            'similarity_threshold': float(self.similarity_threshold),
            'similarity_min_shared_patterns': int(self.similarity_min_shared_patterns),
            'top_event_types': int(self.top_event_types),
            'sensitive_min_distinct_days': int(self.sensitive_min_distinct_days),
            'sensitive_min_events_per_day': int(self.sensitive_min_events_per_day),
            'change_min_events_per_window': int(self.change_min_events_per_window),
            'change_threshold': float(self.change_threshold),
            'shared_ip_min_users': int(self.shared_ip_min_users),
            'shared_ip_max_score': float(self.shared_ip_max_score),
            'session_config': self.session_config,
            'sensitive_event_types': list(self.sensitive_event_types),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'CorrelationConfig':
        if not data:
            return cls()
        base = cls().to_dict()
        base.update({k: v for k, v in data.items() if k in base})
        base['high_risk_levels'] = tuple(base.get('high_risk_levels') or ('High', 'Critical'))
        base['sensitive_event_types'] = tuple(
            base.get('sensitive_event_types') or cls().sensitive_event_types)
        return cls(**base)


CORRELATION_TYPES = (
    'repeated_pattern_across_sessions',
    'similar_account_behavior',
    'repeated_sensitive_operations',
    'long_term_behavior_change',
    'shared_ip_activity',
)

_FINDING_COLUMNS = [
    'correlation_id', 'correlation_type', 'entity_a', 'entity_b', 'entities',
    'sessions', 'correlation_score', 'shared_ip_only', 'evidence',
    'operating_note',
]


def _corr_id(correlation_type: str, entities: List[str]) -> str:
    blob = json.dumps({'type': correlation_type,
                       'entities': sorted(entities)}, sort_keys=True).encode('utf-8')
    return 'corr_' + hashlib.sha1(blob).hexdigest()[:12]


def _ev(signal: str, observed, expected=None, detail: str = '') -> Dict:
    return {
        'signal': signal,
        'observed_value': observed,
        'expected_baseline_or_threshold': expected,
        'explanation': detail,
    }


def _patterns_of(series) -> set:
    """Union of sequence_triggered_patterns values into a set of names."""
    out = set()
    for val in series:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, (list, tuple, set)):
            out.update(str(p) for p in val if p)
        else:
            txt = str(val).strip()
            if txt and txt.lower() not in {'nan', 'none', '[]'}:
                out.add(txt)
    return out


def _col(df: pd.DataFrame, name: str, default=0.0) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index)


class CrossSessionCorrelator:
    """Retrospective cross-session correlation over scored events."""

    def __init__(self, config: Optional[CorrelationConfig] = None):
        self.config = config or CorrelationConfig()

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure session IDs exist; return a working copy with normalized cols."""
        if df is None:
            raise ValueError('Cannot correlate None DataFrame')
        out = df.copy()
        has_sessions = 'session_id' in out.columns and \
            out['session_id'].map(lambda x: _norm_token(x) != '').sum() > 0
        if not has_sessions:
            session_cfg = SessionConfig.from_dict(self.config.session_config)
            out = SessionReconstructor(session_cfg).reconstruct(out)

        out['_user'] = _col(out, 'user_id', 'unknown').map(_norm_id)
        out['_session'] = _col(out, 'session_id', '').map(
            lambda x: _norm_token(x) or 'unknown')
        out['_ip'] = _col(out, 'source_ip', 'unknown').map(_norm_id)
        out['_event_type'] = _col(out, 'event_type', '').map(
            lambda x: _norm_token(x) or 'unknown')
        out['_ts'] = pd.to_datetime(_col(out, 'timestamp'), errors='coerce') \
            if 'timestamp' in out.columns else pd.NaT
        out['_sensitive'] = out['_event_type'].isin(
            set(self.config.sensitive_event_types)).astype(int)
        out['_after_hours'] = (pd.to_numeric(_col(out, 'after_hours'),
                                              errors='coerce').fillna(0) > 0).astype(int)
        out['_auth_fail'] = _col(out, 'auth_result', '').map(_is_failure).astype(int)
        out['_combined'] = pd.to_numeric(_col(out, 'combined_score', np.nan),
                                         errors='coerce')
        out['_seq_dev'] = pd.to_numeric(_col(out, 'sequence_deviation', np.nan),
                                        errors='coerce')
        out['_sess_dev'] = pd.to_numeric(_col(out, 'session_deviation', np.nan),
                                         errors='coerce')
        return out

    def _session_table(self, ev: pd.DataFrame) -> pd.DataFrame:
        """Aggregate events into completed-session rows (retrospective)."""
        grp = ev.groupby('_session', sort=True)
        sess = grp.agg(
            user_id=('_user', 'first'),
            event_count=('_session', 'size'),
            sensitive_count=('_sensitive', 'sum'),
            after_hours_count=('_after_hours', 'sum'),
            auth_failures=('_auth_fail', 'sum'),
            first_ts=('_ts', 'min'),
            last_ts=('_ts', 'max'),
        ).reset_index().rename(columns={'_session': 'session_id'})

        reindexed = lambda s: s.reindex(sess['session_id']).to_numpy()
        sess['unique_ips'] = reindexed(grp['_ip'].nunique())
        sess['unique_event_types'] = reindexed(grp['_event_type'].nunique())
        sess['max_combined'] = reindexed(grp['_combined'].max())
        sess['max_seq_dev'] = reindexed(grp['_seq_dev'].max())
        sess['max_sess_dev'] = reindexed(grp['_sess_dev'].max())

        if 'sequence_triggered_patterns' in ev.columns:
            pat_map = {sid: _patterns_of(sub['sequence_triggered_patterns'])
                       for sid, sub in ev.groupby('_session', sort=True)}
        else:
            pat_map = {}
        sess['patterns'] = sess['session_id'].map(pat_map).map(
            lambda x: x if isinstance(x, set) else set())

        if 'risk_level' in ev.columns:
            risk_map = ev.groupby('_session')['risk_level'].agg(
                lambda s: sorted({str(x) for x in s if x is not None}))
            sess['risk_levels'] = sess['session_id'].map(risk_map).map(
                lambda x: x if isinstance(x, list) else [])
        else:
            sess['risk_levels'] = [[] for _ in range(len(sess))]
        return sess

    def _user_vectors(self, sess: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
        """Per-user numeric + pattern vectors for similarity comparison."""
        cfg = self.config
        users = sorted(sess['user_id'].unique())
        vocab = list(ev['_event_type'].value_counts().head(cfg.top_event_types).index)

        rows = []
        for uid in users:
            us = sess[sess['user_id'] == uid]
            ue = ev[ev['_user'] == uid]
            total_events = max(int(us['event_count'].sum()), 1)
            total_sess = max(len(us), 1)
            high_risk = us['risk_levels'].apply(
                lambda r: any(x in cfg.high_risk_levels for x in r))
            has_pattern = us['patterns'].apply(lambda p: len(p) > 0)
            elevated = (us['max_combined'] >= cfg.suspicious_score_threshold) | \
                (us['max_seq_dev'] >= cfg.deviation_repeat_threshold) | \
                (us['max_sess_dev'] >= cfg.deviation_repeat_threshold)
            vec = {
                'user_id': uid,
                'sessions': int(total_sess),
                'events': int(total_events),
                'sensitive_rate': float(us['sensitive_count'].sum()) / total_events,
                'auth_fail_rate': float(us['auth_failures'].sum()) / total_events,
                'after_hours_rate': float(us['after_hours_count'].sum()) / total_events,
                'ip_switch_rate': float((us['unique_ips'] > 1).sum()) / total_sess,
                'elevated_sessions': int((elevated | high_risk | has_pattern).sum()),
                'suspicious_sessions': int((elevated | high_risk).sum()),
            }
            vc = ue['_event_type'].value_counts()
            for et in vocab:
                vec[f'et_{et}'] = float(vc.get(et, 0)) / total_events
            pat_counts: Dict[str, int] = {}
            for pats in us['patterns']:
                for p in pats:
                    pat_counts[p] = pat_counts.get(p, 0) + 1
            vec['_patterns'] = pat_counts
            vec['_pattern_set'] = set(pat_counts)
            rows.append(vec)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Correlation detectors
    # ------------------------------------------------------------------

    def _detect_repeated_patterns(self, sess: pd.DataFrame) -> List[Dict]:
        cfg = self.config
        findings = []
        for uid, us in sess.groupby('user_id', sort=True):
            pat_sessions: Dict[str, set] = {}
            for _, row in us.iterrows():
                for p in row['patterns']:
                    pat_sessions.setdefault(p, set()).add(row['session_id'])
            for pat, sids in sorted(pat_sessions.items()):
                sids = sorted(sids)
                if len(sids) >= cfg.pattern_min_sessions:
                    findings.append(self._finding(
                        'repeated_pattern_across_sessions', [uid],
                        score=min(1.0, 0.5 + 0.15 * (len(sids) - cfg.pattern_min_sessions)),
                        evidence=[_ev(
                            'repeated_sequence_pattern', pat,
                            f'>= {cfg.pattern_min_sessions} sessions',
                            f'Pattern "{pat}" triggered in {len(sids)} sessions '
                            f'for user {uid}: {", ".join(sids[:8])}'
                            + ('…' if len(sids) > 8 else ''))],
                        sessions=sids))

            # Elevated deviation repeated across sessions without a named pattern.
            susp = us[(us['max_combined'] >= cfg.suspicious_score_threshold) |
                      (us['max_seq_dev'] >= cfg.deviation_repeat_threshold) |
                      (us['max_sess_dev'] >= cfg.deviation_repeat_threshold) |
                      us['risk_levels'].apply(
                          lambda r: any(x in cfg.high_risk_levels for x in r))]
            named_covered = set().union(*us['patterns']) if len(us) else set()
            if len(susp) >= cfg.pattern_min_sessions and not named_covered:
                findings.append(self._finding(
                    'repeated_pattern_across_sessions', [uid],
                    score=min(1.0, 0.45 + 0.1 * (len(susp) - cfg.pattern_min_sessions)),
                    evidence=[_ev(
                        'repeated_elevated_deviation', len(susp),
                        f'>= {cfg.pattern_min_sessions} sessions',
                        f'User {uid} has {len(susp)} sessions with elevated '
                        f'combined/deviation scores (sessions: '
                        f'{", ".join(sorted(susp["session_id"])[:8])})')],
                    sessions=sorted(susp['session_id'].tolist())))
        return findings

    def _detect_similar_accounts(self, users_df: pd.DataFrame) -> List[Dict]:
        cfg = self.config
        findings = []
        candidates = users_df[users_df['suspicious_sessions'] > 0]
        if len(candidates) < 2:
            return findings
        feature_cols = [c for c in candidates.columns
                        if c.startswith('et_') or c in (
                            'sensitive_rate', 'auth_fail_rate', 'after_hours_rate',
                            'ip_switch_rate', 'elevated_sessions')]
        mat = candidates[feature_cols].to_numpy(dtype=float)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_n = mat / norms
        ids = candidates['user_id'].tolist()
        pat_sets = {r['user_id']: r['_pattern_set'] for _, r in candidates.iterrows()}

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                sim = float(np.dot(mat_n[i], mat_n[j]))
                shared_pats = pat_sets[ids[i]] & pat_sets[ids[j]]
                if sim >= cfg.similarity_threshold and \
                        len(shared_pats) >= cfg.similarity_min_shared_patterns:
                    score = min(1.0, sim * 0.9 + min(0.1, 0.03 * len(shared_pats)))
                    findings.append(self._finding(
                        'similar_account_behavior', [ids[i], ids[j]],
                        score=score,
                        evidence=[
                            _ev('behavioral_profile_similarity', round(sim, 4),
                               f'>= {cfg.similarity_threshold}',
                               f'Cosine similarity of aggregated behavioral '
                               f'profiles for {ids[i]} and {ids[j]}'),
                            _ev('shared_triggered_patterns', sorted(shared_pats),
                               f'>= {cfg.similarity_min_shared_patterns}',
                               'Both accounts triggered the same sequence '
                               'patterns'),
                        ]))
        return findings

    def _sensitive_tracks(self, ev: pd.DataFrame) -> pd.DataFrame:
        """Per-user per-day sensitive event counts."""
        sub = ev[ev['_sensitive'] > 0].copy()
        if len(sub) == 0:
            return pd.DataFrame(columns=['user_id', 'day', 'sensitive_events'])
        sub['_day'] = sub['_ts'].dt.strftime('%Y-%m-%d')
        sub.loc[sub['_ts'].isna(), '_day'] = 'unknown'
        tracks = sub.groupby(['_user', '_day'], sort=True).size().reset_index(
            name='sensitive_events')
        return tracks.rename(columns={'_user': 'user_id', '_day': 'day'})

    def _detect_repeated_sensitive(self, tracks: pd.DataFrame) -> List[Dict]:
        cfg = self.config
        findings = []
        if len(tracks) == 0:
            return findings
        for uid, g in tracks.groupby('user_id', sort=True):
            active = g[g['sensitive_events'] >= cfg.sensitive_min_events_per_day]
            if len(active) >= cfg.sensitive_min_distinct_days:
                total = int(active['sensitive_events'].sum())
                score = min(1.0, 0.4 + 0.1 * len(active) +
                            min(0.2, 0.02 * total))
                day_list = ', '.join(
                    f"{r['day']}:{int(r['sensitive_events'])}"
                    for _, r in active.head(10).iterrows())
                findings.append(self._finding(
                    'repeated_sensitive_operations', [uid],
                    score=score,
                    evidence=[_ev(
                        'sensitive_ops_per_day', day_list,
                        f'>= {cfg.sensitive_min_distinct_days} distinct days',
                        f'User {uid} performed {total} sensitive operations '
                        f'across {len(active)} distinct days ({day_list})')]))
        return findings

    def _detect_long_term_change(self, ev: pd.DataFrame) -> List[Dict]:
        """Detect users whose aggregate behavior shifts early vs late."""
        cfg = self.config
        findings = []
        timed = ev[ev['_ts'].notna()].copy()
        if len(timed) == 0:
            return findings

        def _profile(sub: pd.DataFrame) -> Dict[str, float]:
            n = max(len(sub), 1)
            span_h = max((sub['_ts'].max() - sub['_ts'].min()).total_seconds() / 3600.0,
                         1.0 / 3600.0)
            return {
                'sensitive_rate': float(sub['_sensitive'].sum()) / n,
                'auth_fail_rate': float(sub['_auth_fail'].sum()) / n,
                'after_hours_rate': float(sub['_after_hours'].sum()) / n,
                'event_rate': float(n / span_h),
                'unique_event_types': float(sub['_event_type'].nunique()),
            }

        for uid, ue in timed.groupby('_user', sort=True):
            if len(ue) < 2 * cfg.change_min_events_per_window:
                continue
            mid = ue['_ts'].quantile(0.5)
            early, late = ue[ue['_ts'] <= mid], ue[ue['_ts'] > mid]
            if len(early) < cfg.change_min_events_per_window or \
                    len(late) < cfg.change_min_events_per_window:
                continue
            pe, pl = _profile(early), _profile(late)
            shifts = {k: abs(pl[k] - pe[k]) / max(abs(pe[k]), abs(pl[k]), 1e-6)
                      for k in pe}
            change_score = float(np.clip(
                np.mean(list(shifts.values())) / max(cfg.change_threshold, 1e-6)
                * 0.5, 0.0, 1.0))
            material = any(
                abs(pl[k] - pe[k]) >= 0.05 * max(abs(pe[k]), 1e-6) or
                (k == 'unique_event_types' and abs(pl[k] - pe[k]) >= 2)
                for k in pe)
            if change_score >= cfg.change_threshold and material:
                top = sorted(shifts.items(), key=lambda kv: -kv[1])[:3]
                detail = '; '.join(f'{k}: {pe[k]:.3g} -> {pl[k]:.3g}' for k, _ in top)
                findings.append(self._finding(
                    'long_term_behavior_change', [uid],
                    score=min(1.0, change_score),
                    evidence=[_ev(
                        'early_vs_late_profile_shift',
                        {k: [round(pe[k], 4), round(pl[k], 4)] for k in sorted(pe)},
                        f'>= {cfg.change_threshold} normalized shift',
                        f'User {uid} behavior shifted between early and late '
                        f'windows ({detail})')]))
        return findings

    def _detect_shared_ip(self, ev: pd.DataFrame,
                          existing: List[Dict],
                          users_df: pd.DataFrame) -> List[Dict]:
        """Shared-IP findings: always low-confidence unless corroborated.

        Only IPs where at least one involved account shows suspicious
        activity are reported; routine shared IPs (NAT, labs) are skipped.
        """
        cfg = self.config
        findings = []
        suspicious_users = set(users_df.loc[
            users_df['suspicious_sessions'] > 0, 'user_id']) \
            if len(users_df) else set()

        ip_users: Dict[str, set] = {}
        ip_sessions: Dict[str, set] = {}
        ip_suspicious: Dict[str, bool] = {}
        ip_max_score: Dict[str, float] = {}
        for _, row in ev.iterrows():
            ip = row['_ip']
            ip_users.setdefault(ip, set()).add(row['_user'])
            ip_sessions.setdefault(ip, set()).add(row['_session'])
            if row['_user'] in suspicious_users:
                ip_suspicious[ip] = True
            score = row['_combined']
            if pd.notna(score):
                ip_max_score[ip] = max(ip_max_score.get(ip, 0.0), float(score))

        # Corroboration: user pairs already linked by another correlation type.
        corroborated_pairs = set()
        for f in existing:
            ents = f.get('entities', [])
            if len(ents) >= 2 and f['correlation_type'] != 'shared_ip_activity':
                corroborated_pairs.add(tuple(sorted(ents[:2])))

        for ip in sorted(ip_users):
            if not ip_suspicious.get(ip, False):
                continue
            users = sorted(u for u in ip_users[ip] if u != 'unknown')
            if len(users) < cfg.shared_ip_min_users:
                continue
            sessions = sorted(ip_sessions.get(ip, set()))
            pairs = [(users[i], users[j])
                     for i in range(len(users)) for j in range(i + 1, len(users))]
            n_corr = sum(1 for p in pairs if p in corroborated_pairs)
            corroborated = n_corr > 0
            ip_elev = ip_max_score.get(ip, 0.0) >= cfg.suspicious_score_threshold
            score = 0.55 if corroborated else min(
                cfg.shared_ip_max_score,
                0.15 + 0.05 * min(len(users) - 2, 4) + (0.1 if ip_elev else 0.0))
            findings.append(self._finding(
                'shared_ip_activity', users,
                score=min(1.0, score),
                shared_ip_only=not corroborated,
                sessions=sessions,
                evidence=[
                    _ev('shared_source_ip', ip,
                       f'>= {cfg.shared_ip_min_users} distinct users',
                       f'{len(users)} users share source IP {ip} across '
                       f'{len(sessions)} sessions: {", ".join(users[:10])}'
                       + ('…' if len(users) > 10 else '') +
                       f'; max combined_score on this IP: '
                       f'{ip_max_score.get(ip, 0.0):.3f}'),
                    _ev('corroborating_signal', n_corr,
                       f'>= {cfg.min_corroborating_signals} user pair(s)',
                       f'{n_corr} of {len(pairs)} user pairs on this IP are '
                       'linked by an independent correlation type'
                       if corroborated else
                       'No independent correlation links these accounts; '
                       'shared IP may reflect NAT, labs, or coincidental '
                       'access'),
                ],
                operating_note=(
                    f'Corroborated by {n_corr} independent pair correlation(s)'
                    if corroborated else
                    'Shared IP only — insufficient evidence of coordination')))
        return findings

    # ------------------------------------------------------------------
    # Finding assembly & public API
    # ------------------------------------------------------------------

    def _finding(self, correlation_type: str, entities: List[str],
                 score: float, evidence: List[Dict],
                 sessions: Optional[List[str]] = None,
                 shared_ip_only: bool = False,
                 operating_note: str = '') -> Dict:
        return {
            'correlation_id': _corr_id(correlation_type, entities),
            'correlation_type': correlation_type,
            'entity_a': entities[0] if entities else '',
            'entity_b': entities[1] if len(entities) > 1 else '',
            'entities': sorted(entities),
            'sessions': sorted(sessions or []),
            'correlation_score': round(float(np.clip(score, 0.0, 1.0)), 4),
            'shared_ip_only': bool(shared_ip_only),
            'evidence': evidence,
            'operating_note': operating_note,
        }

    def correlate(self, df: pd.DataFrame) -> Dict:
        """
        Correlate scored events across sessions/accounts/IPs.

        Returns dict with keys:
          - 'correlations': DataFrame of findings (typed-empty if none)
          - 'session_table': completed-session aggregates
          - 'account_profiles': per-user behavioral profiles
          - 'sensitive_tracks': per-user per-day sensitive event counts
          - 'skipped_channels': correlation types skipped on missing input
        """
        ev = self._prepare_events(df)
        if len(ev) == 0:
            return self._empty_result()

        sess = self._session_table(ev)
        users_df = self._user_vectors(sess, ev)
        findings: List[Dict] = []

        findings.extend(self._detect_repeated_patterns(sess))
        findings.extend(self._detect_similar_accounts(users_df))
        tracks = self._sensitive_tracks(ev)
        findings.extend(self._detect_repeated_sensitive(tracks))
        findings.extend(self._detect_long_term_change(ev))
        findings.extend(self._detect_shared_ip(ev, findings, users_df))

        corr_df = pd.DataFrame(findings, columns=_FINDING_COLUMNS) \
            if findings else pd.DataFrame(columns=_FINDING_COLUMNS)
        if len(corr_df):
            corr_df = corr_df.sort_values(
                ['correlation_score', 'correlation_id'],
                ascending=[False, True]).reset_index(drop=True)

        return {
            'correlations': corr_df,
            'session_table': sess,
            'account_profiles': users_df,
            'sensitive_tracks': tracks,
            'skipped_channels': [],
        }

    def _empty_result(self) -> Dict:
        return {
            'correlations': pd.DataFrame(columns=_FINDING_COLUMNS),
            'session_table': pd.DataFrame(),
            'account_profiles': pd.DataFrame(),
            'sensitive_tracks': pd.DataFrame(
                columns=['user_id', 'day', 'sensitive_events']),
            'skipped_channels': [],
        }
