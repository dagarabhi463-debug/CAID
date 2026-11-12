"""
Tests for Milestone 1.5.3: Downstream detection and risk consistency audit.

Verifies score-direction semantics across the full pipeline:
AnomalyDetector -> RiskEngine -> DetectionRulesEngine ->
BehavioralAnalyzer -> DetectionEngine.

Score contract:
- anomaly_score: raw IsolationForest decision_function, HIGHER = MORE NORMAL
- anomaly_score_normalized: 0-1 calibrated, HIGHER = MORE ANOMALOUS
- rule_score / rule_detection_score: 0-1, higher = more rule evidence
- behavioral_deviation_score: 0-1, higher = more deviation
- risk_score / combined_score: 0-1, higher = riskier
"""

import numpy as np
import pandas as pd
import pytest

from core.log_processor import load_sample_dataset
from models.anomaly_detector import AnomalyDetector
from core.risk_engine import RiskEngine
from core.detection_engine import DetectionEngine, RiskLevel
from core.detection_rules import DetectionRulesEngine
from core.behavioural_analyzer import BehavioralAnalyzer


@pytest.fixture(scope='module')
def detector():
    """Train once per module - fitting IsolationForest is expensive."""
    d = AnomalyDetector()
    d.fit(load_sample_dataset('normal'))
    return d


def make_event(**overrides):
    """Build a single raw security-log event (normal-ish defaults)."""
    row = {
        'timestamp': '2024-03-01 10:00:00',
        'user_id': 'user_test',
        'role': 'student',
        'source_ip': '192.0.2.10',
        'event_type': 'course_view',
        'courses_accessed': 1,
        'accounts_accessed': 1,
        'api_requests': 50,
        'data_mb': 1.0,
        'token_age_minutes': 30,
        'after_hours': 0,
        'label': 'normal',
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestScoreDirection:
    """Mandatory consistency checks on score semantics."""

    def test_normal_not_more_anomalous_than_anomalous(self, detector):
        """A clearly anomalous event must score >= a normal event."""
        normal = detector.predict(make_event(api_requests=50, data_mb=1.0))
        anomalous = detector.predict(make_event(
            api_requests=5000, data_mb=500.0, courses_accessed=40,
            accounts_accessed=60, after_hours=1, event_type='bulk_download',
            token_age_minutes=5000,
        ))
        n_score = normal['anomaly_score_normalized'].iloc[0]
        a_score = anomalous['anomaly_score_normalized'].iloc[0]
        assert a_score >= n_score
        # Raw score direction must be opposite (higher = more normal)
        assert anomalous['anomaly_score'].iloc[0] <= normal['anomaly_score'].iloc[0]

    def test_higher_anomaly_does_not_reduce_risk(self):
        """risk_score must increase with anomaly_score_normalized, all else equal."""
        engine = RiskEngine()
        base = {
            'api_requests': 50, 'data_mb': 1.0, 'courses_accessed': 1,
            'accounts_accessed': 1, 'after_hours': 0,
        }
        low = pd.Series({**base, 'anomaly_score_normalized': 0.1})
        high = pd.Series({**base, 'anomaly_score_normalized': 0.9})
        s_low, _ = engine.calculate_risk_score(low)
        s_high, _ = engine.calculate_risk_score(high)
        assert s_high > s_low

    def test_raw_score_inverted_not_treated_as_anomalous(self):
        """Raw decision_function (higher=normal) must be inverted downstream.

        Uses process_dataframe so the batch-level raw-score detection applies:
        a frame whose anomaly_score spans negative values is recognized as raw
        decision_function output and inverted (higher raw = lower ml_score).
        """
        engine = DetectionEngine()
        df = pd.DataFrame([
            {'user_id': 'u', 'timestamp': 't', 'is_anomaly': True,
             'anomaly_score': -0.2,    # raw: more anomalous
             'rule_score': 0.0, 'behavioral_deviation_score': 0.0,
             'behavioral_score_missing': False},
            {'user_id': 'u', 'timestamp': 't', 'is_anomaly': False,
             'anomaly_score': 0.3,     # raw: more normal
             'rule_score': 0.0, 'behavioral_deviation_score': 0.0,
             'behavioral_score_missing': False},
        ])
        out = engine.process_dataframe(df)
        assert out['ml_anomaly_score'].iloc[0] > out['ml_anomaly_score'].iloc[1]
        assert out['combined_score'].iloc[0] > out['combined_score'].iloc[1]

    def test_single_row_raw_score_out_of_range_inverted(self):
        """Single-row fallback: anomaly_score outside [0,1] is treated as raw."""
        engine = DetectionEngine()
        row_bad = pd.Series({'user_id': 'u', 'timestamp': 't', 'is_anomaly': True,
                             'anomaly_score': -0.2})
        row_good = pd.Series({'user_id': 'u', 'timestamp': 't', 'is_anomaly': False,
                              'anomaly_score': 1.5})
        d_bad = engine.process_event(row_bad)
        d_good = engine.process_event(row_good)
        assert d_bad.ml_anomaly_score > d_good.ml_anomaly_score

    def test_normalized_score_not_double_inverted(self, detector):
        """Full predict -> detection path: anomalous event gets higher combined_score."""
        eval_df = pd.concat([
            load_sample_dataset('normal').head(20),
            load_sample_dataset('high_risk').head(20),
        ]).reset_index(drop=True)
        predicted = detector.predict(eval_df)
        # Milestone 1.9: provide all three channel inputs so the combined
        # score is computed in full mode (constant neutral rule/behavioral
        # channels isolate the ML score direction being tested).
        predicted['rule_score'] = 0.0
        predicted['behavioral_deviation_score'] = 0.0
        predicted['behavioral_score_missing'] = False

        engine = DetectionEngine()
        result = engine.process_dataframe(predicted)

        assert 'combined_score' in result.columns
        assert result['combined_score'].between(0, 1).all()
        # High-risk labeled rows should not score systematically lower
        hr = result[result['label'] == 'high_risk']['combined_score'].mean()
        nm = result[result['label'] == 'normal']['combined_score'].mean()
        assert hr >= nm

    def test_rule_score_column_compatibility(self):
        """Both 'rule_score' (production) and 'rule_detection_score' (legacy) work."""
        engine = DetectionEngine()
        row_prod = pd.Series({
            'user_id': 'u', 'timestamp': 't',
            'anomaly_score_normalized': 0.5,
            'rule_score': 0.8,
            'behavioral_deviation_score': 0.0,
        })
        row_legacy = pd.Series({
            'user_id': 'u', 'timestamp': 't',
            'anomaly_score_normalized': 0.5,
            'rule_detection_score': 0.8,
            'behavioral_deviation_score': 0.0,
        })
        assert engine.process_event(row_prod).combined_score == \
            pytest.approx(engine.process_event(row_legacy).combined_score)

    def test_rule_engine_output_feeds_detection(self):
        """Real DetectionRulesEngine output ('rule_score') must reach combined_score."""
        rules = DetectionRulesEngine()
        event = make_event(api_requests=5000, data_mb=200.0, after_hours=1,
                           event_type='bulk_download', accounts_accessed=60)
        ruled = rules.process_events(event)
        assert 'rule_score' in ruled.columns
        assert ruled['rule_score'].iloc[0] > 0  # rules actually fired
        # Milestone 1.9: add a real ML observation so two channels have
        # evidence (rules + ml -> degraded_ml_rules mode).
        ruled['anomaly_score_normalized'] = 0.5
        ruled['is_anomaly'] = False

        engine = DetectionEngine()
        det = engine.process_dataframe(ruled)
        expected_ml = float(np.clip(
            ruled.get('anomaly_score_normalized', pd.Series([0.5])).iloc[0], 0, 1)) \
            if 'anomaly_score_normalized' in ruled.columns else 0.5
        # Behavioral column absent -> missing flag; combined_score renormalizes
        # weights over ml+rules: (0.35*ml + 0.35*rule) / 0.70
        expected = (0.35 * expected_ml + 0.35 * float(ruled['rule_score'].iloc[0])) / 0.70
        assert det['combined_score'].iloc[0] == pytest.approx(expected, abs=1e-6)

    def test_missing_ml_flagged_not_silent(self):
        """Missing ML columns must be flagged, not silently treated as normal."""
        engine = DetectionEngine()
        row = pd.Series({'user_id': 'u', 'timestamp': 't'})
        det = engine.process_event(row)
        assert det.ml_anomaly_score == 0.5
        assert 'unavailable' in det.explanation.lower()

        risk = RiskEngine()
        score, factors = risk.calculate_risk_score(row)
        ml_factor = [f for f in factors if f['factor'] == 'ML Anomaly Score'][0]
        assert ml_factor['ml_score_missing'] is True

    def test_missing_behavioral_handled(self):
        """Absent behavioral columns must not crash and contribute 0."""
        engine = DetectionEngine()
        row = pd.Series({
            'user_id': 'u', 'timestamp': 't',
            'anomaly_score_normalized': 0.4,
            'rule_score': 0.2,
        })
        det = engine.process_event(row)
        # Missing behavioral result is NaN + explicit flag, never silent 0
        assert pd.isna(det.behavioral_deviation)
        assert det.behavioral_score_missing is True
        assert 0 <= det.combined_score <= 1

    def test_combined_score_bounds(self):
        """combined_score stays within [0, 1] for extreme inputs."""
        engine = DetectionEngine()
        row = pd.Series({
            'user_id': 'u', 'timestamp': 't',
            'anomaly_score_normalized': 1.0,
            'rule_score': 1.0,
            'behavioral_deviation_score': 1.0,
        })
        det = engine.process_event(row)
        assert det.combined_score == pytest.approx(1.0)
        assert det.risk_level == RiskLevel.CRITICAL

    def test_save_load_preserves_calibration(self, detector, tmp_path):
        """Score calibration survives save/load."""
        path = tmp_path / 'model.joblib'
        detector.save(str(path))
        loaded = AnomalyDetector().load(str(path))
        assert loaded._score_calibration == detector._score_calibration

        df = make_event(api_requests=500, data_mb=50)
        a = detector.predict(df)['anomaly_score_normalized'].iloc[0]
        b = loaded.predict(df)['anomaly_score_normalized'].iloc[0]
        assert a == pytest.approx(b)


class TestEndToEndScenarios:
    """Integration scenarios through the full detection pipeline."""

    @pytest.fixture(scope='class')
    def pipeline_df(self, detector):
        """Run representative scenarios through the whole pipeline once."""
        scenarios = pd.DataFrame([
            # normal student activity
            {'timestamp': '2024-03-01 10:00:00', 'user_id': 'stu_normal', 'role': 'student',
             'source_ip': '192.0.2.10', 'event_type': 'course_view',
             'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 50,
             'data_mb': 1.0, 'token_age_minutes': 30, 'after_hours': 0, 'label': 'normal'},
            # unusual login time
            {'timestamp': '2024-03-01 03:00:00', 'user_id': 'stu_late', 'role': 'student',
             'source_ip': '192.0.2.11', 'event_type': 'login',
             'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 40,
             'data_mb': 0.5, 'token_age_minutes': 20, 'after_hours': 1, 'label': 'suspicious'},
            # high API volume
            {'timestamp': '2024-03-01 11:00:00', 'user_id': 'stu_api', 'role': 'student',
             'source_ip': '192.0.2.12', 'event_type': 'course_view',
             'courses_accessed': 2, 'accounts_accessed': 1, 'api_requests': 2000,
             'data_mb': 2.0, 'token_age_minutes': 30, 'after_hours': 0, 'label': 'suspicious'},
            # bulk data export
            {'timestamp': '2024-03-01 14:00:00', 'user_id': 'ins_export', 'role': 'instructor',
             'source_ip': '203.0.113.5', 'event_type': 'bulk_download',
             'courses_accessed': 5, 'accounts_accessed': 8, 'api_requests': 800,
             'data_mb': 300.0, 'token_age_minutes': 4000, 'after_hours': 0, 'label': 'high_risk'},
            # unusual IP (TEST-NET-3, synthetic high_risk range)
            {'timestamp': '2024-03-01 15:00:00', 'user_id': 'stu_ip', 'role': 'student',
             'source_ip': '203.0.113.99', 'event_type': 'login',
             'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 60,
             'data_mb': 1.0, 'token_age_minutes': 30, 'after_hours': 0, 'label': 'suspicious'},
            # suspicious account access
            {'timestamp': '2024-03-01 16:00:00', 'user_id': 'ta_acct', 'role': 'ta',
             'source_ip': '192.0.2.13', 'event_type': 'user_view',
             'courses_accessed': 3, 'accounts_accessed': 40, 'api_requests': 300,
             'data_mb': 5.0, 'token_age_minutes': 200, 'after_hours': 0, 'label': 'high_risk'},
            # multiple simultaneous indicators
            {'timestamp': '2024-03-01 02:00:00', 'user_id': 'stu_multi', 'role': 'student',
             'source_ip': '203.0.113.77', 'event_type': 'student_export',
             'courses_accessed': 15, 'accounts_accessed': 30, 'api_requests': 3000,
             'data_mb': 400.0, 'token_age_minutes': 8000, 'after_hours': 1, 'label': 'high_risk'},
        ])

        # Full pipeline in production order
        df = detector.predict(scenarios)
        df = RiskEngine().process_events(df)
        df = DetectionRulesEngine().process_events(df)
        # Behavioral baselines fitted on the normal-activity TRAINING set —
        # never on the evaluation scenarios (leakage-free contract)
        behavioral = BehavioralAnalyzer()
        behavioral.fit(load_sample_dataset('normal'))
        df = behavioral.analyze_user_behavior(df)
        df = DetectionEngine().process_dataframe(df)
        return df

    def test_pipeline_columns_present(self, pipeline_df):
        for col in ('anomaly_score', 'anomaly_score_normalized', 'is_anomaly',
                    'risk_score', 'risk_level', 'risk_factors',
                    'rule_score', 'behavioral_deviation_score',
                    'combined_score', 'detection_id', 'detection_explanation'):
            assert col in pipeline_df.columns, col

    def test_scores_in_valid_ranges(self, pipeline_df):
        for col in ('anomaly_score_normalized', 'risk_score', 'rule_score',
                    'behavioral_deviation_score', 'combined_score'):
            vals = pd.to_numeric(pipeline_df[col], errors='coerce').dropna()
            assert vals.between(0, 1).all(), f"{col} out of [0,1]: {vals.min()}..{vals.max()}"

    def test_normal_event_lowest_risk(self, pipeline_df):
        normal = pipeline_df[pipeline_df['user_id'] == 'stu_normal'].iloc[0]
        worst = pipeline_df.loc[pipeline_df['combined_score'].idxmax()]
        assert normal['combined_score'] < worst['combined_score']

    def test_multi_indicator_event_high_risk(self, pipeline_df):
        multi = pipeline_df[pipeline_df['user_id'] == 'stu_multi'].iloc[0]
        assert multi['combined_score'] >= 0.5
        assert multi['risk_level'] in ('High', 'Critical')

    def test_rule_score_actually_contributes(self, pipeline_df):
        """Events that trigger rules must have rule_score > 0 in output."""
        export = pipeline_df[pipeline_df['event_type'] == 'bulk_download'].iloc[0]
        assert export['rule_score'] > 0 or export['rules_triggered'] > 0
