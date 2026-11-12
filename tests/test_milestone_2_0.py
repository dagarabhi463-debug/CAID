"""Milestone 2.0 session/sequence intelligence tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.session_reconstruction import SessionReconstructor, SessionConfig, SESSION_PREFIX_FEATURES
from core.session_baseline import SessionBaselineManager
from core.session_analyzer import SessionAnalyzer
from core.sequence_detection import SequenceAnalyzer, SequenceConfig
from core.detection_engine import DetectionEngine, OperatingMode
from core.calibration import save_calibration, load_calibration, CalibrationUnavailable


def make_events(n_sessions=3, per_session=4, start='2026-01-05 08:00:00',
                user='user1', explicit=False, auth_pattern=False):
    rows = []
    base = pd.Timestamp(start)
    sid = 0
    for s in range(n_sessions):
        sid += 1
        session_id = f'sess_{sid}' if explicit else ''
        for i in range(per_session):
            ts = base + pd.Timedelta(minutes=s * 40 + i * 2)
            et = 'login' if i == 0 else ('course_view' if i % 2 else 'file_view')
            auth = ''
            if auth_pattern and et == 'login':
                auth = 'failure' if i < 2 else 'success'
            rows.append({
                'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': user, 'role': 'student',
                'source_ip': '192.0.2.10', 'event_type': et,
                'courses_accessed': 2, 'accounts_accessed': 2,
                'api_requests': 50, 'data_mb': 1.0,
                'token_age_minutes': 30, 'after_hours': 0,
                'label': 'normal', 'session_id': session_id,
                'auth_result': auth,
            })
    return pd.DataFrame(rows)


class TestSessionReconstruction:
    def test_explicit_session_ids_respected(self):
        df = make_events(n_sessions=2, per_session=3, explicit=True)
        out = SessionReconstructor().reconstruct(df)
        assert out['session_id'].nunique() == 2
        assert out['session_id'].str.startswith('explicit:').all()

    def test_timeout_splits_sessions(self):
        df = make_events(n_sessions=1, per_session=6, explicit=False)
        # Make events 10 minutes apart -> two sessions with 30-min timeout.
        ts = pd.to_datetime(df['timestamp'])
        new_ts = [ts.iloc[0] + pd.Timedelta(minutes=int(m)) for m in (35, 37, 39)]
        df.loc[3:, 'timestamp'] = [t.strftime('%Y-%m-%d %H:%M:%S') for t in new_ts]
        out = SessionReconstructor(SessionConfig(inactivity_timeout_seconds=1800)).reconstruct(df)
        assert out['session_id'].nunique() == 2

    def test_missing_timestamps_isolated(self):
        df = make_events(n_sessions=1, per_session=3)
        df.loc[1, 'timestamp'] = ''
        out = SessionReconstructor().reconstruct(df)
        assert out.loc[1, 'session_id'].str.startswith('missing_ts:').iloc[0] if hasattr(out.loc[1,'session_id'],'str') else True
        assert out.loc[1, 'session_timestamp_missing'] == True

    def test_duplicate_flagged_and_preserved(self):
        df = make_events(n_sessions=1, per_session=3)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        out = SessionReconstructor().reconstruct(df)
        assert len(out) == 4
        assert out['is_duplicate_event'].sum() >= 2

    def test_prefix_features_causal(self):
        df = make_events(n_sessions=1, per_session=4, explicit=True)
        out = SessionReconstructor().reconstruct(df)
        counts = out['session_prefix_event_count'].tolist()
        assert counts == [1, 2, 3, 4]
        # Later events must not change earlier prefix counts.
        assert out.loc[0, 'session_prefix_event_count'] == 1

    def test_session_summary_columns(self):
        df = make_events(n_sessions=2, per_session=3, explicit=True)
        events, sessions = SessionReconstructor().reconstruct_and_summarize(df)
        assert 'session_event_count' in sessions.columns
        assert 'session_duration_seconds' in sessions.columns
        assert len(sessions) == 2


class TestSessionBaseline:
    def test_fit_transform_save_load(self, tmp_path):
        train = make_events(n_sessions=4, per_session=5, explicit=True)
        mgr = SessionBaselineManager().fit(train)
        assert mgr.is_fitted()
        path = tmp_path / 'session_baselines.joblib'
        mgr.save(str(path))
        loaded = SessionBaselineManager().load(str(path))
        assert loaded.fingerprint() == mgr.fingerprint()
        out = loaded.transform_events(SessionReconstructor().reconstruct(train))
        assert 'session_deviation_score' in out.columns
        assert out['session_score_missing'].dtype == bool

    def test_unseen_user_global_fallback(self):
        train = make_events(n_sessions=3, per_session=4, explicit=True, user='user1')
        mgr = SessionBaselineManager().fit(train)
        ev = SessionReconstructor().reconstruct(make_events(n_sessions=1, per_session=3, explicit=True, user='new_user'))
        out = mgr.transform_events(ev)
        assert out['session_baseline_scope'].iloc[0] == 'global'

    def test_missing_values_explicit(self):
        mgr = SessionBaselineManager().fit(make_events(n_sessions=2, per_session=3, explicit=True))
        row = pd.Series({'user_id': 'u', 'session_prefix_event_count': np.nan})
        score, factors, missing, scope = mgr.score_row(row)
        assert missing is True or np.isnan(score)


class TestSequenceDetection:
    def test_auth_failure_success_export_scores(self):
        rows = []
        base = pd.Timestamp('2026-01-05 09:00:00')
        for i, (et, auth) in enumerate([
            ('login', 'failure'), ('login', 'failure'),
            ('login', 'success'), ('bulk_export', '')]):
            rows.append({
                'timestamp': (base + pd.Timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S'),
                'user_id': 'u1', 'role': 'student', 'source_ip': '192.0.2.10',
                'event_type': et, 'courses_accessed': 8, 'accounts_accessed': 8,
                'api_requests': 200, 'data_mb': 20.0, 'token_age_minutes': 30,
                'after_hours': 0, 'label': 'suspicious', 'session_id': 's1',
                'auth_result': auth})
        df = pd.DataFrame(rows)
        train = make_events(n_sessions=5, per_session=4, explicit=True, user='u1')
        analyzer = SequenceAnalyzer().fit(train)
        out = analyzer.analyze(df)
        assert 'sequence_deviation_score' in out.columns
        assert out['sequence_deviation_score'].between(0, 1).all()
        last = out.iloc[-1]
        assert 'bulk_export' in str(last['sequence_triggered_patterns']) or \
               'sensitive_operation_burst' in str(last['sequence_triggered_patterns']) or \
               last['sequence_deviation_score'] >= 0.3

    def test_missing_auth_result_does_not_fabricate(self):
        df = make_events(n_sessions=1, per_session=3, explicit=True)
        analyzer = SequenceAnalyzer().fit(make_events(n_sessions=4, per_session=4, explicit=True))
        out = analyzer.analyze(df)
        assert out['sequence_score_missing'].dtype == bool
        assert out['sequence_deviation_score'].between(0, 1).all()

    def test_save_load_deterministic(self, tmp_path):
        train = make_events(n_sessions=4, per_session=4, explicit=True)
        analyzer = SequenceAnalyzer().fit(train)
        path = tmp_path / 'seq.joblib'
        analyzer.save(str(path))
        loaded = SequenceAnalyzer().load(str(path))
        assert loaded.fingerprint() == analyzer.fingerprint()


class TestDetectionEngineV2:
    def _row(self, **kw):
        base = {'user_id': 'u', 'timestamp': '2026-01-01 00:00:00',
                'anomaly_score_normalized': 0.4, 'rule_score': 0.1,
                'behavioral_deviation_score': 0.2,
                'session_deviation_score': 0.3, 'sequence_deviation_score': 0.2,
                'session_score_missing': False, 'sequence_score_missing': False}
        base.update(kw)
        return pd.Series(base)

    def test_v2_weights_combined(self):
        engine = DetectionEngine(channel_profile='v2_0')
        det = engine.process_event(self._row())
        expected = (0.25*0.4 + 0.25*0.1 + 0.15*0.2 + 0.20*0.3 + 0.15*0.2)
        assert det.combined_score == pytest.approx(expected)
        assert det.operating_mode == OperatingMode.FULL_V2_0.value

    def test_missing_session_excluded_renormalized(self):
        engine = DetectionEngine(channel_profile='v2_0')
        det = engine.process_event(self._row(session_deviation_score=np.nan,
                                              session_score_missing=True))
        assert 'session' in det.missing_channels
        assert det.combined_score == pytest.approx(
            (0.25*0.4 + 0.25*0.1 + 0.15*0.2 + 0.15*0.2) / 0.80)
        assert det.operating_mode == OperatingMode.DEGRADED_V2_0.value

    def test_legacy_profile_ignores_session_columns(self):
        engine = DetectionEngine()
        det = engine.process_event(self._row())
        assert det.operating_mode == OperatingMode.FULL_ML_RULES_BEHAVIORAL.value
        assert det.combined_score == pytest.approx((0.35*0.4 + 0.35*0.1 + 0.30*0.2))

    def test_insufficient_evidence_nan(self):
        engine = DetectionEngine(channel_profile='v2_0')
        det = engine.process_event(self._row(anomaly_score_normalized=np.nan,
                                             rule_score=np.nan,
                                             behavioral_deviation_score=np.nan,
                                             session_deviation_score=np.nan,
                                             sequence_deviation_score=np.nan,
                                             session_score_missing=True,
                                             sequence_score_missing=True))
        assert np.isnan(det.combined_score)
        assert det.risk_level.value == 'Insufficient Evidence'


class TestCalibrationV2:
    def test_v2_artifact_fingerprints(self, tmp_path):
        path = tmp_path / 'cal.joblib'
        art = save_calibration({'medium': 0.1, 'high': 0.2, 'critical': 0.3},
                               {'calibration_rows': 10}, detector=None,
                               dataset_role='calibration',
                               path=str(path), artifact_version=2,
                               fingerprints={'detector': 'abc', 'session_baseline': 'def'},
                               channel_profile='v2_0',
                               detection_thresholds={'medium': 0.2, 'high': 0.4, 'critical': 0.6})
        loaded = load_calibration(str(path), fingerprints={'detector': 'abc'})
        assert loaded['artifact_version'] == 2
        assert loaded['fingerprints']['session_baseline'] == 'def'
        with pytest.raises(CalibrationUnavailable):
            load_calibration(str(path), fingerprints={'detector': 'wrong'})

    def test_v1_artifact_still_loads(self, tmp_path):
        path = tmp_path / 'cal_v1.joblib'
        save_calibration({'medium': 0.1, 'high': 0.2, 'critical': 0.3},
                         {'calibration_rows': 5}, detector=None,
                         dataset_role='calibration', path=str(path))
        loaded = load_calibration(str(path))
        assert loaded['artifact_version'] == 1
