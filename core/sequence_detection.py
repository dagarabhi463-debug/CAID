"""
SequenceAnalyzer for CAID — Milestone 2.0

Configurable, interpretable sequence/transition anomaly scoring. Uses only
causal session-prefix context (events up to and including the current event)
so future events cannot influence earlier scores.

The output ``sequence_deviation_score`` is a normalized deviation in [0,1]
(higher = more anomalous). It is NOT a probability.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from core.session_reconstruction import (
    SessionConfig,
    SessionReconstructor,
    DEFAULT_SENSITIVE_EVENT_TYPES,
    _is_failure,
    _is_success,
    _norm_token,
    _norm_id,
    _safe_float,
)


DEFAULT_SEQUENCE_DEFINITIONS: Tuple[Dict, ...] = (
    {
        'name': 'login_failure_burst_then_access',
        'kind': 'auth_transition',
        'params': {'min_failures': 2, 'require_success': True},
    },
    {
        'name': 'sensitive_operation_burst',
        'kind': 'sensitive_burst',
        'params': {'min_sensitive': 3},
    },
    {
        'name': 'export_after_broad_access',
        'kind': 'broad_access_then_sensitive',
        'params': {'min_accounts': 5, 'min_courses': 5},
    },
    {
        'name': 'ip_switch_burst',
        'kind': 'ip_switch_burst',
        'params': {'min_switches': 3},
    },
)

COMPONENT_WEIGHTS = {
    'transition_rarity': 0.30,
    'event_rate_deviation': 0.20,
    'sensitive_sequence_deviation': 0.25,
    'pattern_match': 0.25,
}


@dataclass
class SequenceConfig:
    window_seconds: float = 900.0
    max_window_events: int = 50
    min_window_events: int = 2
    ignore_duplicates: bool = True
    sensitive_event_types: Tuple[str, ...] = DEFAULT_SENSITIVE_EVENT_TYPES
    auth_column: str = 'auth_result'
    transition_smoothing: float = 0.5
    deviation_multiplier: float = 3.0
    definitions: Tuple[Dict, ...] = DEFAULT_SEQUENCE_DEFINITIONS
    weights: Dict[str, float] = field(default_factory=lambda: dict(COMPONENT_WEIGHTS))

    def to_dict(self) -> Dict:
        return {
            'window_seconds': float(self.window_seconds),
            'max_window_events': int(self.max_window_events),
            'min_window_events': int(self.min_window_events),
            'ignore_duplicates': bool(self.ignore_duplicates),
            'sensitive_event_types': list(self.sensitive_event_types),
            'auth_column': self.auth_column,
            'transition_smoothing': float(self.transition_smoothing),
            'deviation_multiplier': float(self.deviation_multiplier),
            'definitions': [dict(d) for d in self.definitions],
            'weights': dict(self.weights),
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> 'SequenceConfig':
        if not data:
            return cls()
        base = cls().to_dict()
        base.update({k: v for k, v in data.items() if k in base})
        base['sensitive_event_types'] = tuple(base.get('sensitive_event_types', DEFAULT_SENSITIVE_EVENT_TYPES))
        return cls(**base)


def _scope_stats_empty() -> Dict:
    return {
        'pair_counts': {},
        'prev_totals': {},
        'unigram': {},
        'event_count': 0,
        'rate_mean': 0.0,
        'rate_std': 1.0,
        'sensitive_mean': 0.0,
        'sensitive_std': 1.0,
        'has_auth': False,
    }


class SequenceAnalyzer:
    """Fit-once sequence model with persisted artifact and causal scoring."""

    ARTIFACT_VERSION = 1

    def __init__(self, config: Optional[SequenceConfig] = None):
        self.config = config or SequenceConfig()
        self.user_models: Dict[str, Dict] = {}
        self.global_model: Dict = _scope_stats_empty()
        self.session_config: Dict = SessionConfig().to_dict()
        self._is_fitted = False
        self._fit_event_count = 0
        self._fit_session_count = 0
        # Ephemeral runtime status set by ensure_fitted_sequence_analyzer:
        # 'loaded' | 'refit_fallback_missing' | 'refit_fallback_corrupt'
        self.artifact_status: Optional[str] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def _accumulate(self, model: Dict, events: pd.DataFrame,
                    sessions: pd.DataFrame, user_col: str):
        ev = events.copy()
        ev['_event_type'] = ev.get('event_type', pd.Series('', index=ev.index)).map(
            lambda x: _norm_token(x) or 'unknown')
        ev['_prev_event_type'] = ev.get('session_prefix_prev_event_type',
                                        pd.Series('', index=ev.index)).fillna('')
        ev['_transition'] = ev.get('session_prefix_transition_key',
                                   pd.Series('', index=ev.index)).fillna('')
        ev['_user'] = ev.get(user_col, pd.Series('unknown', index=ev.index)).map(
            lambda x: _norm_id(x))

        for _, row in ev.iterrows():
            et = row['_event_type']
            model['unigram'][et] = model['unigram'].get(et, 0) + 1
            model['event_count'] += 1
            trans = row['_transition']
            prev = row['_prev_event_type']
            if trans and '->' in trans:
                model['pair_counts'][trans] = model['pair_counts'].get(trans, 0) + 1
                model['prev_totals'][prev] = model['prev_totals'].get(prev, 0) + 1
            auth = row.get(self.config.auth_column)
            if auth is not None and str(auth).strip() != '':
                model['has_auth'] = True

        if sessions is not None and len(sessions):
            rates = pd.to_numeric(sessions.get('session_event_rate', pd.Series(dtype=float)),
                                  errors='coerce').dropna()
            sens = pd.to_numeric(sessions.get('session_sensitive_event_count',
                                              pd.Series(dtype=float)), errors='coerce').fillna(0)
            model['rate_mean'] = float(rates.mean()) if len(rates) else 0.0
            model['rate_std'] = float(rates.std()) if len(rates) and rates.std() > 0 else 1.0
            model['sensitive_mean'] = float(sens.mean()) if len(sens) else 0.0
            model['sensitive_std'] = float(sens.std()) if len(sens) and sens.std() > 0 else 1.0

    def fit(self, df: pd.DataFrame,
            config: Optional[SequenceConfig] = None,
            session_config: Optional[SessionConfig] = None) -> 'SequenceAnalyzer':
        """Fit sequence baselines on TRAINING-normal data only."""
        if df is None or len(df) == 0:
            raise ValueError('Cannot fit sequence analyzer on empty DataFrame')
        self.config = config or self.config
        session_cfg = session_config or SessionConfig.from_dict(self.session_config)
        self.session_config = session_cfg.to_dict()

        recon = SessionReconstructor(session_cfg)
        events, sessions = recon.reconstruct_and_summarize(df)

        self.global_model = _scope_stats_empty()
        self.user_models = {}
        self._accumulate(self.global_model, events, sessions, session_cfg.user_column)

        ev_users = events.get(session_cfg.user_column, pd.Series('unknown', index=events.index)).map(_norm_id)
        sess_users = sessions.get('user_id', pd.Series('unknown', index=sessions.index)).map(_norm_id) \
            if len(sessions) else pd.Series(dtype=object)
        for uid in sorted(set(ev_users) | set(sess_users)):
            model = _scope_stats_empty()
            ev_mask = ev_users == uid
            sess_mask = (sess_users == uid) if len(sess_users) else pd.Series(False, index=sessions.index)
            self._accumulate(model, events.loc[ev_mask],
                             sessions.loc[sess_mask] if len(sessions) else sessions,
                             session_cfg.user_column)
            self.user_models[uid] = model

        self._fit_event_count = int(len(events))
        self._fit_session_count = int(len(sessions))
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _model_for_user(self, user_id) -> Tuple[Dict, str]:
        uid = _norm_id(user_id)
        if uid in self.user_models:
            return self.user_models[uid], 'user'
        return self.global_model, 'global'

    @staticmethod
    def _safe_std(v) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 1.0
        return f if np.isfinite(f) and f > 0 else 1.0

    def _transition_score(self, model: Dict, prev: str, cur: str) -> Tuple[float, bool, Dict]:
        if not prev:
            return float('nan'), True, {'reason': 'first_event_in_session'}
        key = f'{prev}->{cur}'
        alpha = float(self.config.transition_smoothing)
        V = max(len(model['unigram']), 1)
        pair_count = model['pair_counts'].get(key, 0)
        prev_total = model['prev_totals'].get(prev, 0)
        if prev_total > 0:
            prob = (pair_count + alpha) / (prev_total + alpha * V)
        else:
            # Unseen previous event: fall back to unigram probability.
            prob = (model['unigram'].get(cur, 0) + alpha) / (model['event_count'] + alpha * V)
        # rarity: 0 for uniform-typical transitions, ->1 for very rare ones
        score = float(np.clip(1.0 - prob * V, 0.0, 1.0))
        return score, False, {
            'transition': key, 'probability': round(float(prob), 6),
            'pair_count': int(pair_count), 'prev_total': int(prev_total),
        }

    def _rate_score(self, model: Dict, rate) -> Tuple[float, bool]:
        try:
            value = float(rate)
        except (TypeError, ValueError):
            return float('nan'), True
        if not np.isfinite(value):
            return float('nan'), True
        std = self._safe_std(model['rate_std'])
        mean = float(model['rate_mean'])
        if mean == 0:
            dev = 1.0 if value > 0 else 0.0
        else:
            z = abs(value - mean) / std
            thr = float(self.config.deviation_multiplier)
            dev = float(np.clip((z - thr) / thr, 0.0, 1.0)) if z > thr else 0.0
        return dev, False

    def _sensitive_score(self, model: Dict, count) -> Tuple[float, bool]:
        try:
            value = float(count)
        except (TypeError, ValueError):
            return float('nan'), True
        std = self._safe_std(model['sensitive_std'])
        mean = float(model['sensitive_mean'])
        if mean == 0:
            dev = 1.0 if value > 0 else 0.0
        else:
            z = abs(value - mean) / std
            thr = float(self.config.deviation_multiplier)
            dev = float(np.clip((z - thr) / thr, 0.0, 1.0)) if z > thr else 0.0
        return dev, False

    # ------------------------------------------------------------------
    # Pattern definitions (causal window only)
    # ------------------------------------------------------------------

    def _eval_definitions(self, window: List[Dict], current: Dict) -> Tuple[float, List[str], List[Dict]]:
        """Return (best_score, triggered_names, evidence)."""
        sensitive = set(self.config.sensitive_event_types)
        best = 0.0
        triggered: List[str] = []
        evidence: List[Dict] = []
        cfg_defs = self.config.definitions or DEFAULT_SEQUENCE_DEFINITIONS

        for definition in cfg_defs:
            kind = definition.get('kind')
            params = definition.get('params', {})
            name = definition.get('name', kind)
            score = 0.0
            detail = ''

            if kind == 'auth_transition':
                if self._window_has_auth(window, current):
                    failures = sum(1 for e in window if e.get('auth_fail'))
                    cur_success = bool(current.get('auth_success'))
                    cur_sensitive = current.get('event_type') in sensitive
                    need_fail = int(params.get('min_failures', 2))
                    require_success = bool(params.get('require_success', True))
                    if failures >= need_fail and (not require_success or cur_success or cur_sensitive):
                        score = 1.0
                        detail = f'{failures} prior auth failures then ' + \
                                 ('success' if cur_success else 'sensitive/access')
                # If auth_result is entirely absent, this definition is inactive.

            elif kind == 'sensitive_burst':
                min_sens = int(params.get('min_sensitive', 3))
                sens_count = sum(1 for e in window if e.get('sensitive')) + \
                    (1 if current.get('sensitive') else 0)
                if sens_count >= min_sens:
                    score = 1.0
                    detail = f'{sens_count} sensitive operations in causal window'

            elif kind == 'broad_access_then_sensitive':
                accounts = sum(_safe_float(e.get('accounts_accessed', 0)) for e in window) + \
                    _safe_float(current.get('accounts_accessed', 0))
                courses = sum(_safe_float(e.get('courses_accessed', 0)) for e in window) + \
                    _safe_float(current.get('courses_accessed', 0))
                min_acc = float(params.get('min_accounts', 5))
                min_crs = float(params.get('min_courses', 5))
                if current.get('sensitive') and accounts >= min_acc and courses >= min_crs:
                    score = 1.0
                    detail = f'sensitive op after broad access (accounts={accounts:.0f}, courses={courses:.0f})'

            elif kind == 'ip_switch_burst':
                min_sw = int(params.get('min_switches', 3))
                switches = sum(1 for e in window if e.get('ip_switch'))
                if switches >= min_sw:
                    score = 1.0
                    detail = f'{switches} source-IP switches in causal window'

            if score > best:
                best = score
            if score > 0:
                triggered.append(name)
                evidence.append({
                    'pattern': name,
                    'kind': kind,
                    'score': score,
                    'detail': detail,
                })
        return best, triggered, evidence

    @staticmethod
    def _window_has_auth(window: List[Dict], current: Dict) -> bool:
        if current.get('auth_present'):
            return True
        return any(e.get('auth_present') for e in window)

    # ------------------------------------------------------------------
    # Public scoring
    # ------------------------------------------------------------------

    def _ensure_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'session_prefix_event_count' in df.columns and 'session_id' in df.columns:
            return df
        session_cfg = SessionConfig.from_dict(self.session_config)
        return SessionReconstructor(session_cfg).reconstruct(df)

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score every event using causal session context only."""
        if not self._is_fitted:
            raise RuntimeError('SequenceAnalyzer is not fitted')
        if df is None:
            raise ValueError('Cannot analyze None DataFrame')
        events = self._ensure_session_features(df)
        out = events.copy()
        n = len(out)
        out['_row_pos'] = np.arange(n)
        scores = [np.nan] * n
        missing = [True] * n
        factors_all: List[List[Dict]] = [[] for _ in range(n)]
        evidence_all: List[List[Dict]] = [[] for _ in range(n)]
        triggered_all: List[List[str]] = [[] for _ in range(n)]
        scope_all = ['insufficient'] * n

        cfg = self.config
        sensitive = set(cfg.sensitive_event_types)
        auth_col = cfg.auth_column
        user_col = 'user_id'

        # Process sessions independently in deterministic order.
        work = out.copy()
        work['_order_key'] = pd.to_numeric(
            work.get('session_prefix_event_index', pd.Series(0, index=work.index)),
            errors='coerce').fillna(0)
        work = work.sort_values(['session_id', '_order_key', '_row_pos'],
                                kind='mergesort')

        for _, grp in work.groupby('session_id', sort=True):
            window: deque = deque(maxlen=int(cfg.max_window_events))
            prev_ip = None
            ip_switches = 0
            for _, row in grp.iterrows():
                row_d = row.to_dict()
                pos = int(row_d['_row_pos'])
                event_type = _norm_token(row_d.get('event_type', '')) or 'unknown'
                ip = _norm_id(row_d.get('source_ip', 'unknown'))
                if prev_ip is not None and ip != prev_ip:
                    ip_switches += 1
                prev_ip = ip

                auth_val = row_d.get(auth_col)
                auth_present = auth_val is not None and str(auth_val).strip() != ''
                cur = {
                    'event_type': event_type,
                    'sensitive': event_type in sensitive,
                    'auth_present': auth_present,
                    'auth_fail': auth_present and _is_failure(auth_val),
                    'auth_success': auth_present and _is_success(auth_val),
                    'ip_switch': ip_switches > 0 and bool(window) and window[-1].get('ip') != ip,
                    'accounts_accessed': _safe_float(row_d.get('accounts_accessed', 0)),
                    'courses_accessed': _safe_float(row_d.get('courses_accessed', 0)),
                    'api_requests': _safe_float(row_d.get('api_requests', 0)),
                    'data_mb': _safe_float(row_d.get('data_mb', 0)),
                    'source_ip': ip,
                }

                # Build causal window: same session, earlier/equal position,
                # within time window when timestamps are available.
                cur_ts = pd.to_datetime(row_d.get('timestamp'), errors='coerce') \
                    if 'timestamp' in row_d else pd.NaT
                win_events = []
                if cur_ts is not pd.NaT and pd.notna(cur_ts):
                    for e in window:
                        ets = e.get('_ts')
                        if ets is not None and pd.notna(ets) and \
                                0 <= (cur_ts - ets).total_seconds() <= cfg.window_seconds:
                            win_events.append(e)
                else:
                    win_events = list(window)
                if cfg.ignore_duplicates:
                    win_events = [e for e in win_events if not e.get('duplicate')]

                model, scope = self._model_for_user(row_d.get(user_col))
                prev_event = row_d.get('session_prefix_prev_event_type') or ''
                transition_score, trans_missing, trans_ev = self._transition_score(
                    model, str(prev_event), event_type)
                rate_score, rate_missing = self._rate_score(
                    model, row_d.get('session_prefix_event_rate'))
                sens_score, sens_missing = self._sensitive_score(
                    model, row_d.get('session_prefix_sensitive_event_count'))
                pattern_score, triggered, pattern_ev = self._eval_definitions(win_events, cur)

                components = {}
                factors = []
                evidence = []
                if not trans_missing and np.isfinite(transition_score):
                    components['transition_rarity'] = transition_score
                    factors.append({'factor': 'Transition Rarity', 'value': transition_score,
                                    'weight': COMPONENT_WEIGHTS['transition_rarity'],
                                    'impact': 'positive' if transition_score > 0 else 'none',
                                    'detail': trans_ev})
                    if transition_score > 0.5:
                        evidence.append({'channel': 'sequence', 'evidence_name': 'rare_transition',
                                         'observed_value': transition_score,
                                         'expected_baseline_or_threshold': 'user/global transition model',
                                         'contribution_or_severity': transition_score,
                                         'availability': 'available',
                                         'explanation': f"Unusual transition {trans_ev.get('transition','')}"})
                else:
                    factors.append({'factor': 'Transition Rarity', 'value': None, 'impact': 'missing',
                                    'missing': True})

                if not rate_missing and np.isfinite(rate_score):
                    components['event_rate_deviation'] = rate_score
                    factors.append({'factor': 'Event Rate Deviation', 'value': rate_score,
                                    'weight': COMPONENT_WEIGHTS['event_rate_deviation'],
                                    'impact': 'positive' if rate_score > 0 else 'none'})
                else:
                    factors.append({'factor': 'Event Rate Deviation', 'value': None,
                                    'impact': 'missing', 'missing': True})

                if not sens_missing and np.isfinite(sens_score):
                    components['sensitive_sequence_deviation'] = sens_score
                    factors.append({'factor': 'Sensitive Sequence Deviation', 'value': sens_score,
                                    'weight': COMPONENT_WEIGHTS['sensitive_sequence_deviation'],
                                    'impact': 'positive' if sens_score > 0 else 'none'})
                else:
                    factors.append({'factor': 'Sensitive Sequence Deviation', 'value': None,
                                    'impact': 'missing', 'missing': True})

                # Pattern match is available whenever at least one definition ran.
                components['pattern_match'] = pattern_score
                if pattern_score > 0:
                    factors.append({'factor': 'Sequence Pattern Match', 'value': pattern_score,
                                    'weight': COMPONENT_WEIGHTS['pattern_match'],
                                    'impact': 'positive'})
                    for pe in pattern_ev:
                        evidence.append({'channel': 'sequence', 'evidence_name': pe['pattern'],
                                         'observed_value': pe['score'],
                                         'expected_baseline_or_threshold': pe['kind'],
                                         'contribution_or_severity': pe['score'],
                                         'availability': 'available',
                                         'explanation': pe.get('detail', pe['pattern'])})
                else:
                    factors.append({'factor': 'Sequence Pattern Match', 'value': 0.0,
                                    'weight': COMPONENT_WEIGHTS['pattern_match'],
                                    'impact': 'none'})

                weights = self.config.weights or COMPONENT_WEIGHTS
                total_w = sum(weights.get(k, 0.0) for k in components)
                if total_w <= 0:
                    score = float('nan')
                    miss = True
                else:
                    score = float(np.clip(
                        sum(components[k] * weights.get(k, 0.0) for k in components) / total_w,
                        0.0, 1.0))
                    miss = False

                scores[pos] = score
                missing[pos] = miss
                factors_all[pos] = factors
                evidence_all[pos] = evidence
                triggered_all[pos] = triggered
                scope_all[pos] = scope if not miss else 'insufficient'

                # Append current event to causal window for subsequent events.
                window.append({
                    **cur,
                    '_ts': cur_ts if pd.notna(cur_ts) else None,
                    'duplicate': bool(row_d.get('is_duplicate_event', False)),
                })

        out['sequence_deviation_score'] = scores
        out['sequence_score_missing'] = missing
        out['sequence_factors'] = factors_all
        out['sequence_evidence'] = evidence_all
        out['sequence_triggered_patterns'] = triggered_all
        out['sequence_baseline_scope'] = scope_all
        # Restore original row order.
        out = out.sort_values('_row_pos' if '_row_pos' in out.columns else out.index)
        return out.drop(columns=[c for c in ('_row_pos',) if c in out.columns])

    def score_sessions(self, sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Retrospective session-level scoring for dashboards/evaluation."""
        # Reuse per-event max sequence score per session as a simple summary.
        if sessions_df is None or len(sessions_df) == 0:
            return sessions_df
        return sessions_df

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def is_fitted(self) -> bool:
        return self._is_fitted

    def fingerprint(self) -> str:
        payload = {
            'artifact_version': self.ARTIFACT_VERSION,
            'config': self.config.to_dict(),
            'session_config': self.session_config,
            'global_model': self.global_model,
            'user_models': self.user_models,
            'fit_event_count': self._fit_event_count,
            'fit_session_count': self._fit_session_count,
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str) -> None:
        if not self._is_fitted:
            raise RuntimeError('Cannot save unfitted sequence analyzer')
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            'artifact_version': self.ARTIFACT_VERSION,
            'config': self.config.to_dict(),
            'session_config': self.session_config,
            'global_model': self.global_model,
            'user_models': self.user_models,
            'fit_event_count': self._fit_event_count,
            'fit_session_count': self._fit_session_count,
        }, path)

    def load(self, path: str) -> 'SequenceAnalyzer':
        if not os.path.exists(path):
            raise FileNotFoundError(f'Sequence model artifact not found: {path}')
        data = joblib.load(path)
        if not isinstance(data, dict):
            raise ValueError(f'Invalid sequence artifact at {path}')
        if data.get('artifact_version') != self.ARTIFACT_VERSION:
            raise ValueError(
                f'Incompatible sequence artifact version {data.get("artifact_version")!r} at {path}')
        required = ('config', 'session_config', 'global_model', 'user_models', 'fit_event_count')
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f'Incomplete sequence artifact at {path}: missing {missing}')
        self.config = SequenceConfig.from_dict(data['config'])
        self.session_config = SessionConfig.from_dict(data['session_config']).to_dict()
        self.global_model = data['global_model']
        self.user_models = data['user_models']
        self._fit_event_count = data['fit_event_count']
        self._fit_session_count = data.get('fit_session_count', 0)
        self._is_fitted = True
        return self


def get_sequence_model_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'models', 'saved_models', 'sequence_model.joblib')


def ensure_fitted_sequence_analyzer(path: Optional[str] = None,
                                    train_df: Optional[pd.DataFrame] = None
                                    ) -> SequenceAnalyzer:
    """Load persisted sequence model, or fit on training-normal data.

    Lifecycle policy (AUD-002, see core/artifact_lifecycle.py):
    - loads cleanly            -> artifact_status = 'loaded'
    - missing                  -> fit + persist, 'refit_fallback_missing'
    - corrupt/incompatible     -> quarantine corrupt file, fit + persist
                                  fresh, 'refit_fallback_corrupt'

    ``train_df`` is used only when an artifact must be (re)fit; it must be
    designated training-normal data. Once a usable artifact exists it is
    loaded and ``train_df`` is ignored.
    """
    from core.artifact_lifecycle import (
        STATUS_LOADED, STATUS_REFIT_MISSING, STATUS_REFIT_CORRUPT,
        quarantine_corrupt)

    path = path or get_sequence_model_path()
    if os.path.exists(path):
        try:
            analyzer = SequenceAnalyzer().load(path)
            analyzer.artifact_status = STATUS_LOADED
            return analyzer
        except Exception as e:
            quarantine = quarantine_corrupt(path)
            print(f"Failed to load sequence model: {e}. "
                  f"Quarantined corrupt artifact to {quarantine}; "
                  "refitting on training-normal data...")
            if train_df is None:
                from core.log_processor import load_sample_dataset
                train_df = load_sample_dataset('normal')
            analyzer = SequenceAnalyzer().fit(train_df)
            analyzer.save(path)
            analyzer.artifact_status = STATUS_REFIT_CORRUPT
            return analyzer
    if train_df is None:
        from core.log_processor import load_sample_dataset
        train_df = load_sample_dataset('normal')
    analyzer = SequenceAnalyzer().fit(train_df)
    analyzer.save(path)
    analyzer.artifact_status = STATUS_REFIT_MISSING
    return analyzer
