"""
Milestone 1.7 tests: performance vectorization equivalence, risk
calibration, strict data-quality cleaning, and score-contract cleanup.
"""

import numpy as np
import pandas as pd
import pytest

from core.log_processor import load_sample_dataset
from models.anomaly_detector import AnomalyDetector
from core.risk_engine import RiskEngine, RiskLevel
from core.detection_engine import DetectionEngine
from core.behavioural_analyzer import ensure_fitted_analyzer
from utils.validators import clean_and_summarize, clean_security_log
from modules.export import (get_download_button_csv, export_csv,
                            export_analysis_summary, export_filtered_data,
                            export_ip_investigation, export_user_investigation,
                            export_timeline_events)


@pytest.fixture(scope='module')
def detector():
    d = AnomalyDetector()
    d.fit(load_sample_dataset('normal'))
    return d


class TestVectorizedFeatureEquivalence:
    """Vectorized transform_user_features must match legacy semantics."""

    def test_engineered_features_deterministic(self, detector):
        df = load_sample_dataset('all').head(100)
        bm = detector.baseline_manager
        from core.feature_engineering import engineer_features
        a, _ = engineer_features(df, bm)
        b, _ = engineer_features(df, bm)
        pd.testing.assert_frame_equal(a, b)

    def test_unseen_user_gets_global_stats(self, detector):
        from core.feature_engineering import engineer_features
        df = pd.DataFrame([{
            'timestamp': '2024-01-01 10:00:00', 'user_id': 'brand_new_xyz',
            'role': 'student', 'source_ip': '192.0.2.1', 'event_type': 'login',
            'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 50,
            'data_mb': 1.0, 'token_age_minutes': 30, 'after_hours': 0}])
        out, _ = engineer_features(df, detector.baseline_manager)
        g = detector.baseline_manager.global_stats
        # Milestone 1.8.1: unseen users get TYPICAL-user statistics (mean
        # across training users of each per-user feature), not pooled
        # population aggregates.
        assert out['user_mean_api'].iloc[0] == pytest.approx(g['typical_mean_api'])
        # Milestone 1.8: unseen users get a TYPICAL user's event count
        # (per_user_event_count), not the pooled training total — the old
        # fallback made every unseen user a massive outlier (FP root cause)
        assert out['user_event_count'].iloc[0] == pytest.approx(
            g.get('per_user_event_count', g['event_count']))
        assert out['user_total_accounts'].iloc[0] == 0.0  # legacy schema quirk

    def test_predict_still_fast_path(self, detector):
        """predict output columns unchanged by vectorization."""
        df = detector.predict(load_sample_dataset('normal').head(20))
        for col in ('is_anomaly', 'anomaly_score', 'anomaly_score_raw',
                    'anomaly_score_normalized', 'anomaly_prediction'):
            assert col in df.columns


class TestRiskCalibration:
    """Fixed thresholds calibrated on separate data; eval metrics sane."""

    def test_calibrate_on_normal_sets_absolute_thresholds(self):
        engine = RiskEngine()
        cal = load_sample_dataset('normal').head(200)
        thresholds = engine.calibrate_thresholds(cal)
        assert set(thresholds) == {'medium', 'high', 'critical'}
        assert thresholds['medium'] <= thresholds['high'] <= thresholds['critical']
        assert engine.fixed_thresholds is not None

    def test_calibrated_classification_is_absolute(self):
        """Same risk_score -> same level regardless of dataset composition."""
        engine = RiskEngine()
        engine.calibrate_thresholds(load_sample_dataset('normal').head(200))
        t = engine.fixed_thresholds
        assert engine.classify_risk(t['critical'] + 0.001) == RiskLevel.CRITICAL
        assert engine.classify_risk(t['medium'] - 0.001) == RiskLevel.LOW

    def test_uncalibrated_keeps_legacy_behavior(self):
        engine = RiskEngine()
        assert engine.fixed_thresholds is None
        # legacy absolute fallback still works
        assert engine.classify_risk(0.8) == RiskLevel.CRITICAL

    def test_evaluate_calibration_metrics(self):
        engine = RiskEngine()
        engine.calibrate_thresholds(load_sample_dataset('normal').head(200))
        eval_df = load_sample_dataset('evaluation')
        # evaluation set has no anomaly_score -> risk uses neutral ML 0.5
        metrics = engine.evaluate_calibration(eval_df, mode='binary')
        for k in ('tp', 'fp', 'tn', 'fn', 'precision', 'recall',
                  'f1_score', 'false_positive_rate'):
            assert k in metrics
        assert 0 <= metrics['false_positive_rate'] <= 1

    def test_calibrate_rejects_empty(self):
        with pytest.raises(ValueError):
            RiskEngine().calibrate_thresholds(pd.DataFrame())


class TestStrictCleaning:
    """Row-level cleaning summaries for uploaded data."""

    def _dirty_df(self):
        return pd.DataFrame({
            'timestamp': ['2024-01-01 10:00:00', 'bad-date', '2024-01-02 11:00:00',
                          '2024-01-03 12:00:00'],
            'user_id': ['u1', 'u2', '', 'u4'],
            'role': ['student'] * 4,
            'source_ip': ['192.0.2.1'] * 4,
            'event_type': ['login'] * 4,
            'courses_accessed': [1, 2, 3, 'x'],
            'accounts_accessed': [1, 2, 3, 4],
            'api_requests': [10, 20, 30, 40],
            'data_mb': [1.0, 2.0, 3.0, 4.0],
            'token_age_minutes': [5, 6, 7, 8],
            'after_hours': [0, 1, 2, 0],
        })

    def test_clean_and_summarize_counts(self):
        clean, summary = clean_and_summarize(self._dirty_df())
        assert summary['input_rows'] == 4
        # row1: bad timestamp -> dropped; row2: empty user_id -> dropped
        assert summary['dropped_rows'] == 2
        assert summary['dropped_invalid_timestamp'] == 1
        assert summary['dropped_missing_user_id'] == 1
        assert summary['retained_rows'] == 2
        # retained u4 row: courses_accessed 'x' corrected to 0;
        # the after_hours=2 row was dropped (empty user_id) before correction
        assert summary['corrected_rows'] == 1
        assert summary['corrected_numeric_fields'] == 1
        assert summary['corrected_after_hours'] == 0
        assert set(clean['courses_accessed']) <= {1.0, 0.0}
        assert set(clean['after_hours']) <= {0, 1}

    def test_clean_preserves_original(self):
        raw = self._dirty_df()
        clean_and_summarize(raw)
        # original untouched (auditability)
        assert raw.loc[1, 'timestamp'] == 'bad-date'
        assert len(raw) == 4

    def test_all_invalid_returns_empty_with_summary(self):
        df = pd.DataFrame({'timestamp': ['x'], 'user_id': ['u']})
        clean, summary = clean_and_summarize(df)
        assert len(clean) == 0
        assert summary['dropped_rows'] == 1

    def test_after_hours_strict_binary(self):
        df = pd.DataFrame({
            'timestamp': ['2024-01-01'] * 3, 'user_id': ['a', 'b', 'c'],
            'role': ['s'] * 3, 'source_ip': ['192.0.2.1'] * 3,
            'event_type': ['login'] * 3,
            'courses_accessed': [1] * 3, 'accounts_accessed': [1] * 3,
            'api_requests': [1] * 3, 'data_mb': [1.0] * 3,
            'token_age_minutes': [1] * 3, 'after_hours': [0, 5, 'x'],
        })
        clean, _ = clean_and_summarize(df)
        assert set(clean['after_hours']) <= {0, 1}


class TestFormulaInjectionAllPaths:
    """Every export path must neutralize spreadsheet formulas."""

    EVIL = '=HYPERLINK("http://evil")'

    def _assert_safe(self, content: bytes):
        text = content.decode('utf-8')
        for line in text.splitlines():
            for cell in line.split(','):
                cell = cell.strip().strip('"')
                if cell and not cell.startswith(("'", '#')):
                    assert not cell.startswith(('=', '+', '-', '@')), cell

    def test_get_download_button_csv(self):
        self._assert_safe(get_download_button_csv(
            pd.DataFrame({'a': [self.EVIL]}), 'f.csv', 'x')[0])

    def test_export_csv(self):
        self._assert_safe(export_csv(pd.DataFrame({'a': [self.EVIL]})))

    def test_export_analysis_summary(self):
        df = pd.DataFrame({'user_id': [self.EVIL], 'risk_level': ['Low'],
                           'is_anomaly': [False]})
        self._assert_safe(export_analysis_summary(df))

    def test_export_filtered_data(self):
        df = pd.DataFrame({'user_id': [self.EVIL], 'v': [1]})
        self._assert_safe(export_filtered_data(df, {'user_id': self.EVIL}))

    def test_export_ip_investigation(self):
        self._assert_safe(export_ip_investigation(
            {'ip_address': self.EVIL, 'unusual_indicators': [self.EVIL]}))

    def test_export_user_investigation(self):
        self._assert_safe(export_user_investigation(
            {'user_id': self.EVIL, 'event_types': {self.EVIL: 2}}))

    def test_export_timeline_events(self):
        df = pd.DataFrame({'timestamp': ['2024-01-01'], 'user_id': [self.EVIL],
                           'anomaly_score': [0.1]})
        self._assert_safe(export_timeline_events(df))


class TestScoreContractCleanup:
    """anomaly_score_raw explicit; deprecated alias preserved."""

    def test_predict_emits_explicit_raw_field(self, detector):
        df = detector.predict(load_sample_dataset('normal').head(10))
        assert 'anomaly_score_raw' in df.columns
        # raw and deprecated alias carry identical values
        pd.testing.assert_series_equal(
            df['anomaly_score_raw'].reset_index(drop=True),
            df['anomaly_score'].reset_index(drop=True), check_names=False)

    def test_raw_field_direction_preserved(self, detector):
        normal = detector.predict(
            pd.DataFrame([{'timestamp': '2024-01-01 10:00', 'user_id': 'u',
                           'role': 'student', 'source_ip': '192.0.2.1',
                           'event_type': 'login', 'courses_accessed': 1,
                           'accounts_accessed': 1, 'api_requests': 50,
                           'data_mb': 1.0, 'token_age_minutes': 30,
                           'after_hours': 0}]))
        anom = detector.predict(
            pd.DataFrame([{'timestamp': '2024-01-01 10:00', 'user_id': 'u',
                           'role': 'student', 'source_ip': '203.0.113.9',
                           'event_type': 'bulk_download', 'courses_accessed': 40,
                           'accounts_accessed': 60, 'api_requests': 9000,
                           'data_mb': 900.0, 'token_age_minutes': 9000,
                           'after_hours': 1}]))
        assert anom['anomaly_score_raw'].iloc[0] < normal['anomaly_score_raw'].iloc[0]
        assert anom['anomaly_score_normalized'].iloc[0] > \
            normal['anomaly_score_normalized'].iloc[0]

    def test_detection_engine_prefers_raw_field(self):
        """anomaly_score_raw (higher=normal) must be inverted, not guessed."""
        engine = DetectionEngine()
        row = pd.Series({'user_id': 'u', 'timestamp': 't',
                         'is_anomaly': True, 'anomaly_score_raw': -0.4})
        det = engine.process_event(row)
        assert det.ml_anomaly_score == pytest.approx(0.4)

    def test_process_dataframe_uses_raw_column(self):
        engine = DetectionEngine()
        df = pd.DataFrame([
            {'user_id': 'u', 'timestamp': 't', 'is_anomaly': True,
             'anomaly_score_raw': -0.4, 'behavioral_deviation_score': 0.0},
            {'user_id': 'u', 'timestamp': 't', 'is_anomaly': False,
             'anomaly_score_raw': 0.4, 'behavioral_deviation_score': 0.0},
        ])
        out = engine.process_dataframe(df)
        assert out['ml_anomaly_score'].iloc[0] > out['ml_anomaly_score'].iloc[1]
