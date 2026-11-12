"""
Milestone 1.8.1 tests: unseen-entity fallback reliability and
false-positive reduction.

Covers FeatureBaselineManager semantics for known/unseen users, IPs and
event types; leave-one-out training z-scores; determinism; baseline
immutability during prediction; and end-to-end anomaly-score behavior.
"""

import numpy as np
import pandas as pd
import pytest

from core.feature_engineering import (
    FeatureBaselineManager,
    engineer_features,
    get_feature_matrix,
)
from core.log_processor import load_sample_dataset
from models.anomaly_detector import AnomalyDetector


@pytest.fixture(scope='module')
def detector():
    d = AnomalyDetector()
    d.fit(load_sample_dataset('normal'))
    return d


def _row(user_id='u_new', source_ip='192.0.2.99', event_type='login',
         api=50, data=1.0, courses=1, accounts=1, ts='2024-01-01 10:00:00'):
    return {
        'timestamp': ts, 'user_id': user_id, 'role': 'student',
        'source_ip': source_ip, 'event_type': event_type,
        'courses_accessed': courses, 'accounts_accessed': accounts,
        'api_requests': api, 'data_mb': data,
        'token_age_minutes': 30, 'after_hours': 0,
    }


def _toy_baseline():
    """Small deterministic baseline: 2 known users, 2 known IPs, 2 types."""
    rows = []
    for i, uid in enumerate(('user_a', 'user_b')):
        for j in range(3):
            rows.append({
                'timestamp': f'2024-01-0{i+1} 10:0{j}:00',
                'user_id': uid, 'role': 'student',
                'source_ip': '192.0.2.10' if i == 0 else '192.0.2.20',
                'event_type': 'login' if j < 2 else 'course_view',
                'courses_accessed': 1 + j, 'accounts_accessed': 1,
                'api_requests': 10 * (i + 1) + j,
                'data_mb': 1.0 + j * 0.1,
                'token_age_minutes': 30, 'after_hours': 0,
            })
    return pd.DataFrame(rows)


class TestKnownEntities:
    """Known user/IP/event-type rows must use stored per-entity baselines."""

    def test_known_user_uses_personal_baseline(self, detector):
        train = load_sample_dataset('normal')
        uid = train['user_id'].iloc[0]
        df = pd.DataFrame([_row(user_id=uid)])
        out, _ = engineer_features(df, detector.baseline_manager)
        b = detector.baseline_manager.user_baselines[uid]
        assert out['user_mean_api'].iloc[0] == pytest.approx(b['mean_api'])
        assert out['user_event_count'].iloc[0] == pytest.approx(
            detector.baseline_manager.user_totals[uid]['event_count'])

    def test_known_ip_uses_training_frequency(self, detector):
        train = load_sample_dataset('normal')
        ip = train['source_ip'].value_counts().idxmax()
        df = pd.DataFrame([_row(source_ip=ip)])
        out, _ = engineer_features(df, detector.baseline_manager)
        assert out['ip_event_count'].iloc[0] == pytest.approx(
            detector.baseline_manager.ip_frequency[ip])

    def test_known_event_type_uses_training_frequency(self, detector):
        train = load_sample_dataset('normal')
        et = train['event_type'].value_counts().idxmax()
        df = pd.DataFrame([_row(event_type=et)])
        out, _ = engineer_features(df, detector.baseline_manager)
        assert out['event_type_frequency'].iloc[0] == pytest.approx(
            detector.baseline_manager.event_type_frequency[et])


class TestUnseenEntities:
    """Unseen entities must get documented typical-per-entity fallbacks."""

    def test_unseen_user_typical_fallbacks(self, detector):
        df = pd.DataFrame([_row(user_id='never_seen_user_xyz')])
        out, _ = engineer_features(df, detector.baseline_manager)
        g = detector.baseline_manager.global_stats
        r = out.iloc[0]
        # Typical-user stats, NOT pooled population totals
        assert r['user_event_count'] == pytest.approx(g['per_user_event_count'])
        assert r['user_event_count'] < g['event_count']  # not the 500-total
        for c in ('api', 'data', 'courses'):
            assert r[f'user_mean_{c}'] == pytest.approx(g[f'typical_mean_{c}'])
            assert r[f'user_max_{c}'] == pytest.approx(g[f'typical_max_{c}'])
            assert r[f'user_total_{c}'] == pytest.approx(g[f'typical_total_{c}'])
        # No population-wide count anywhere
        assert r['user_total_api'] < g['total_api']
        assert r['user_total_data'] < g['total_data']

    def test_unseen_user_z_uses_population_stats(self, detector):
        df = pd.DataFrame([_row(user_id='never_seen_user_xyz', api=50)])
        out, _ = engineer_features(df, detector.baseline_manager)
        g = detector.baseline_manager.global_stats
        r = out.iloc[0]
        # Unseen-user z = deviation from the TYPICAL-user mean in units of
        # the population std (documented fallback semantics).
        expected = np.clip(
            (50 - g['typical_mean_api']) / g['std_api'], -10, 10)
        assert r['user_api_z'] == pytest.approx(expected)

    def test_unseen_ip_typical_fallback(self, detector):
        df = pd.DataFrame([_row(source_ip='203.0.113.250')])
        out, _ = engineer_features(df, detector.baseline_manager)
        bm = detector.baseline_manager
        if '203.0.113.250' in bm.ip_frequency:
            pytest.skip('IP unexpectedly present in training')
        # Fallback is the mean events-per-IP (a typical IP), not a total
        assert out['ip_event_count'].iloc[0] == pytest.approx(bm._global_ip_freq)
        assert out['ip_event_count'].iloc[0] < bm.global_stats['event_count']

    def test_unseen_event_type_typical_fallback(self, detector):
        df = pd.DataFrame([_row(event_type='quantum_teleport')])
        out, _ = engineer_features(df, detector.baseline_manager)
        bm = detector.baseline_manager
        assert out['event_type_frequency'].iloc[0] == pytest.approx(
            bm._global_event_freq)
        assert out['event_type_frequency'].iloc[0] < bm.global_stats['event_count']
        assert out['is_sensitive_event'].iloc[0] == 0

    def test_multiple_unseen_entities_same_batch(self, detector):
        df = pd.DataFrame([
            _row(user_id='ghost_1', source_ip='198.51.100.1', event_type='x1'),
            _row(user_id='ghost_2', source_ip='198.51.100.2', event_type='x2'),
            _row(user_id='ghost_3', source_ip='198.51.100.3', event_type='x3'),
        ])
        out, _ = engineer_features(df, detector.baseline_manager)
        g = detector.baseline_manager.global_stats
        # All unseen users get identical typical-user fallbacks
        for col in ('user_event_count', 'user_mean_api', 'user_max_api',
                    'user_total_api', 'user_total_data', 'user_total_courses'):
            assert out[col].nunique() == 1
        assert out['user_event_count'].iloc[0] == pytest.approx(
            g['per_user_event_count'])
        # No cross-contamination: each row keeps its own entity fallbacks
        assert out['ip_event_count'].nunique() == 1  # all unseen -> same fallback
        assert out['event_type_frequency'].nunique() == 1


class TestDegenerateBaselines:
    """Empty, tiny, zero-variance and malformed inputs must be explicit."""

    def test_unfitted_manager_predict_raises(self):
        det = AnomalyDetector()
        with pytest.raises(RuntimeError):
            det.predict(pd.DataFrame([_row()]))

    def test_empty_baseline_fit_rejected(self):
        with pytest.raises(ValueError):
            FeatureBaselineManager().fit(pd.DataFrame())
        with pytest.raises(ValueError):
            AnomalyDetector().fit(pd.DataFrame())

    def test_single_row_baseline(self):
        bm = FeatureBaselineManager().fit(pd.DataFrame([_row(user_id='solo')]))
        out, _ = engineer_features(pd.DataFrame([_row(user_id='solo')]), bm,
                                   leave_one_out=True)
        # Single-event user has no LOO baseline -> z = 0, finite features
        assert np.isfinite(out['user_api_z']).all()
        assert out['user_api_z'].iloc[0] == 0.0

    def test_zero_variance_baseline(self):
        rows = [_row(user_id='zuser', api=50, data=1.0, courses=1, accounts=1)
                for _ in range(3)]
        bm = FeatureBaselineManager().fit(pd.DataFrame(rows))
        out, _ = engineer_features(pd.DataFrame(rows), bm, leave_one_out=True)
        # Constant feature -> std floored to 1.0 -> z = 0 for identical values
        assert np.isfinite(out['user_api_z']).all()
        out2, _ = engineer_features(
            pd.DataFrame([_row(user_id='zuser', api=500)]), bm)
        assert np.isfinite(out2['user_api_z']).all()

    def test_missing_and_malformed_entity_values(self, detector):
        df = pd.DataFrame([
            _row(user_id=None, source_ip=None, event_type=None),
            _row(user_id=np.nan, source_ip=np.nan, event_type=np.nan),
            _row(user_id=12345, source_ip=999, event_type=3.5),
        ])
        out, _ = engineer_features(df, detector.baseline_manager)
        # All normalize to 'unknown' -> identical unseen fallback
        assert out['user_event_count'].nunique() == 1
        assert out['ip_event_count'].nunique() == 1
        assert out['event_type_frequency'].nunique() == 1
        assert np.isfinite(
            out[['user_api_z', 'ip_event_count',
                 'event_type_frequency']].to_numpy(dtype=float)).all()

    def test_missing_feature_columns(self, detector):
        df = pd.DataFrame([{'timestamp': '2024-01-01 10:00:00',
                            'user_id': 'u1'}])
        out, cols = engineer_features(df, detector.baseline_manager)
        X = get_feature_matrix(out, cols)
        assert np.isfinite(X).all()


class TestDeterminismAndImmutability:
    """Repeated prediction is identical; prediction never mutates baselines."""

    def test_repeated_predictions_identical(self, detector):
        df = load_sample_dataset('evaluation').head(50)
        a = detector.predict(df)
        b = detector.predict(df)
        pd.testing.assert_frame_equal(a, b)

    def test_transform_deterministic(self, detector):
        df = load_sample_dataset('evaluation').head(50)
        bm = detector.baseline_manager
        a, _ = engineer_features(df, bm)
        b, _ = engineer_features(df, bm)
        pd.testing.assert_frame_equal(a, b)

    def test_predict_does_not_mutate_baselines(self, detector):
        bm = detector.baseline_manager
        before = {
            'user_baselines': {k: dict(v) for k, v in bm.user_baselines.items()},
            'user_totals': {k: dict(v) for k, v in bm.user_totals.items()},
            'ip_frequency': dict(bm.ip_frequency),
            'event_type_frequency': dict(bm.event_type_frequency),
            'global_stats': dict(bm.global_stats),
        }
        detector.predict(load_sample_dataset('evaluation'))
        assert bm.user_baselines == before['user_baselines']
        assert bm.user_totals == before['user_totals']
        assert bm.ip_frequency == before['ip_frequency']
        assert bm.event_type_frequency == before['event_type_frequency']
        assert bm.global_stats == before['global_stats']


class TestLeaveOneOutTrainingZ:
    """Training-time z must exclude the scored event from its baseline."""

    def test_loo_z_excludes_self(self):
        bm = FeatureBaselineManager().fit(_toy_baseline())
        df = _toy_baseline()
        loo, _ = engineer_features(df, bm, leave_one_out=True)
        plain, _ = engineer_features(df, bm)
        # LOO z differs from self-referential z
        assert not np.allclose(loo['user_api_z'], plain['user_api_z'])
        # user_a api values are 10,11,12 -> LOO z for value 10:
        # others {11,12}: mean 11.5, std(ddof=1)=0.707 -> z=(10-11.5)/0.707=-2.12
        sub = loo[(loo['user_id'] == 'user_a') &
                  (loo['api_requests'] == 10)]
        assert sub['user_api_z'].iloc[0] == pytest.approx(-2.1213, abs=1e-3)

    def test_loo_single_event_user_z_zero(self):
        df = pd.DataFrame([_row(user_id='solo', api=50),
                           _row(user_id='solo', api=50)])
        # two identical events: others std -> 0 -> floor 1.0, z = 0
        bm = FeatureBaselineManager().fit(df)
        out, _ = engineer_features(df, bm, leave_one_out=True)
        assert (out['user_api_z'] == 0.0).all()

    def test_predict_z_unchanged_for_known_user(self, detector):
        """Prediction-time z for a known user still uses stored baseline."""
        train = load_sample_dataset('normal')
        uid = train['user_id'].iloc[0]
        df = pd.DataFrame([_row(user_id=uid, api=10_000)])
        out, _ = engineer_features(df, detector.baseline_manager)
        b = detector.baseline_manager.user_baselines[uid]
        expected = np.clip((10_000 - b['mean_api']) / b['std_api'], -10, 10)
        assert out['user_api_z'].iloc[0] == pytest.approx(expected)


class TestEndToEndScores:
    """Unseen normal entities must not get artificial population counts,
    and genuinely abnormal behavior must remain detectable."""

    def test_unseen_normal_user_not_population_outlier(self, detector):
        """An unseen user with typical activity must not receive a
        population-wide event count or totals."""
        df = pd.DataFrame([_row(user_id='eval_like_user', api=55, data=2.5,
                                courses=3, accounts=2)])
        out = detector.predict(df)
        g = detector.baseline_manager.global_stats
        eng, _ = engineer_features(df, detector.baseline_manager)
        r = eng.iloc[0]
        assert r['user_event_count'] == pytest.approx(g['per_user_event_count'])
        assert r['user_event_count'] < g['event_count']
        assert r['user_total_api'] < g['total_api']
        # And the normalized score must not saturate at the anomaly extreme
        assert out['anomaly_score_normalized'].iloc[0] < 0.9

    def test_genuinely_abnormal_still_detected(self, detector):
        df = pd.DataFrame([_row(
            user_id='attacker_1', source_ip='203.0.113.66',
            event_type='bulk_download', api=9000, data=900.0,
            courses=40, accounts=60, ts='2024-01-01 03:00:00')])
        df.loc[0, 'after_hours'] = 1
        out = detector.predict(df)
        assert out['is_anomaly'].iloc[0] == True  # noqa: E712
        # Clearly in the anomalous region (normalized score is not a
        # threshold, but this event must score far above normal activity)
        assert out['anomaly_score_normalized'].iloc[0] > 0.7
        normal = detector.predict(pd.DataFrame([_row(user_id='benign_9')]))
        assert (out['anomaly_score_normalized'].iloc[0]
                > normal['anomaly_score_normalized'].iloc[0])

    def test_eval_normal_false_positive_rate_improved(self, detector):
        """Measured FP rate on eval-normal must be far below the pre-fix
        ~0.68 (Milestone 1.8 baseline)."""
        ev = detector.predict(load_sample_dataset('evaluation'))
        normal = ev[ev['label'] == 'normal']
        fpr = float(normal['is_anomaly'].mean())
        assert fpr < 0.30
        # Detection on labeled anomalies must remain high
        for lab in ('suspicious', 'high_risk'):
            sub = ev[ev['label'] == lab]
            assert float(sub['is_anomaly'].mean()) > 0.8


class TestCalibrationIntegrity:
    """Milestone 1.8.1 Phase 6: threshold calibration must not leak."""

    def _processed_normal(self):
        from core.risk_engine import RiskEngine
        from core.detection_rules import DetectionRulesEngine
        from core.behavioural_analyzer import ensure_fitted_analyzer
        from core.detection_engine import DetectionEngine
        det = AnomalyDetector()
        det.fit(load_sample_dataset('normal'))
        df = det.predict(load_sample_dataset('normal').head(200))
        df = RiskEngine().process_events(df)
        df = DetectionRulesEngine().process_events(df)
        df = ensure_fitted_analyzer().analyze_user_behavior(df)
        df = DetectionEngine().process_dataframe(df)
        return df

    def test_calibration_uses_baseline_feature_space(self):
        """calibrate_thresholds consumes risk_score rows that already
        contain ML anomaly scores (intended feature space)."""
        from core.risk_engine import RiskEngine
        proc = self._processed_normal()
        assert 'anomaly_score_normalized' in proc.columns
        assert 'risk_score' in proc.columns
        engine = RiskEngine()
        info = engine.calibration_info
        thr = engine.calibrate_thresholds(proc)
        assert engine.calibration_info['calibration_rows'] == len(proc)
        assert thr['medium'] <= thr['high'] <= thr['critical']

    def test_thresholds_fixed_during_holdout_eval(self):
        from core.risk_engine import RiskEngine
        engine = RiskEngine()
        engine.calibrate_thresholds(self._processed_normal())
        before = dict(engine.fixed_thresholds)
        eval_df = load_sample_dataset('evaluation')
        engine.evaluate_calibration(eval_df, mode='binary')
        engine.evaluate_calibration(eval_df, mode='high-risk')
        assert engine.fixed_thresholds == before

    def test_evaluate_calibration_does_not_mutate_input(self):
        from core.risk_engine import RiskEngine
        engine = RiskEngine()
        engine.calibrate_thresholds(self._processed_normal())
        eval_df = load_sample_dataset('evaluation').head(20)
        snapshot = eval_df.copy(deep=True)
        engine.evaluate_calibration(eval_df, mode='binary')
        pd.testing.assert_frame_equal(eval_df, snapshot)

    def test_missing_ml_scores_explicit_neutral(self):
        """Rows without any anomaly score column get neutral 0.5 ML
        contribution flagged via ml_score_missing — never silent-normal."""
        from core.risk_engine import RiskEngine
        engine = RiskEngine()
        row = pd.Series({'api_requests': 10, 'data_mb': 1.0,
                         'courses_accessed': 1, 'accounts_accessed': 1,
                         'after_hours': 0})
        score, factors = engine.calculate_risk_score(row)
        ml = [f for f in factors if f['factor'] == 'ML Anomaly Score'][0]
        assert ml['ml_score_missing'] is True
        assert ml['score_source'] == 'unavailable'
        assert ml['normalized_value'] == pytest.approx(0.5)
