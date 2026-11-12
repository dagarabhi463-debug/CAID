"""
Tests for Milestone 1.5.2: Baseline integration and model consistency.

Proves that FeatureBaselineManager is fitted only on training data,
never modified during prediction, persisted with the model artifact,
and that predictions are invariant to batch composition and row order.
"""

import copy
import os
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest

from core.feature_engineering import (
    FeatureBaselineManager,
    engineer_features,
)
from core.log_processor import (
    load_security_log,
    load_sample_dataset,
    detect_duplicates,
    search_events,
    get_user_activity,
    get_ip_activity,
    generate_timeline,
)
from models.anomaly_detector import AnomalyDetector


def make_raw_df(n=30, user_prefix="user_t", seed=0):
    """Build a small synthetic raw security-log DataFrame."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n, freq='h').astype(str),
        'user_id': [f"{user_prefix}{i % 3}" for i in range(n)],
        'role': ['student'] * n,
        'source_ip': [f'192.0.2.{i % 5}' for i in range(n)],
        'event_type': ['login' if i % 4 else 'course_view' for i in range(n)],
        'courses_accessed': rng.randint(0, 3, n),
        'accounts_accessed': rng.randint(0, 3, n),
        'api_requests': rng.randint(10, 100, n),
        'data_mb': rng.rand(n) * 5,
        'token_age_minutes': rng.randint(0, 120, n),
        'after_hours': [0] * n,
        'label': ['normal'] * n,
    })


@pytest.fixture
def fitted_detector():
    detector = AnomalyDetector()
    detector.fit(load_sample_dataset('normal'))
    return detector


@pytest.fixture
def baseline_manager():
    return FeatureBaselineManager().fit(load_sample_dataset('normal'))


class TestBaselineFitting:
    """Baselines must be computed from training data only."""

    def test_baselines_ignore_prediction_batch(self, baseline_manager):
        """Aggregates for a known user reflect training data, not the batch."""
        train = make_raw_df(user_prefix='user_t')
        manager = FeatureBaselineManager().fit(train)
        known_user = 'user_t0'

        # Batch with an extreme api_requests value for the known user
        batch = pd.DataFrame([{
            'timestamp': '2024-02-01 10:00:00',
            'user_id': known_user,
            'source_ip': '192.0.2.1',
            'event_type': 'login',
            'courses_accessed': 1,
            'accounts_accessed': 1,
            'api_requests': 99999,
            'data_mb': 1.0,
            'token_age_minutes': 10,
            'after_hours': 0,
        }])
        out = manager.transform_user_features(batch)

        expected_mean = train[train['user_id'] == known_user]['api_requests'].mean()
        assert out.loc[0, 'user_mean_api'] == pytest.approx(expected_mean)
        assert out.loc[0, 'user_total_api'] == pytest.approx(
            train[train['user_id'] == known_user]['api_requests'].sum()
        )
        assert out.loc[0, 'user_event_count'] == len(train[train['user_id'] == known_user])

    def test_user_totals_not_zero(self, baseline_manager):
        """user_total_* and user_event_count must be populated, not stuck at 0."""
        df = load_sample_dataset('normal').head(200)
        out, cols = engineer_features(df, baseline_manager)
        for col in ('user_total_api', 'user_total_data', 'user_total_courses',
                    'user_event_count'):
            assert col in cols
            assert (out[col] > 0).any(), f"{col} is all zero"

    def test_zscore_features_present(self, baseline_manager):
        df = load_sample_dataset('normal').head(200)
        out, cols = engineer_features(df, baseline_manager)
        for col in ('user_api_z', 'user_data_z', 'user_courses_z', 'user_accounts_z'):
            assert col in cols
            assert np.isfinite(out[col]).all()


class TestPredictionImmutability:
    """predict() must never modify baseline or scaler state."""

    def test_predict_does_not_mutate_baselines(self, fitted_detector):
        manager = fitted_detector.baseline_manager
        before = {
            'user_baselines': copy.deepcopy(manager.user_baselines),
            'user_totals': copy.deepcopy(manager.user_totals),
            'ip_frequency': copy.deepcopy(manager.ip_frequency),
            'event_type_frequency': copy.deepcopy(manager.event_type_frequency),
            'global_stats': copy.deepcopy(manager.global_stats),
        }
        scaler_mean = fitted_detector.scaler.mean_.copy()
        scaler_scale = fitted_detector.scaler.scale_.copy()

        eval_df = pd.concat([
            load_sample_dataset('suspicious'),
            load_sample_dataset('high_risk'),
        ]).head(100)
        fitted_detector.predict(eval_df)

        assert manager.user_baselines == before['user_baselines']
        assert manager.user_totals == before['user_totals']
        assert manager.ip_frequency == before['ip_frequency']
        assert manager.event_type_frequency == before['event_type_frequency']
        assert manager.global_stats == before['global_stats']
        np.testing.assert_array_equal(fitted_detector.scaler.mean_, scaler_mean)
        np.testing.assert_array_equal(fitted_detector.scaler.scale_, scaler_scale)

    def test_batch_composition_invariance(self, fitted_detector):
        """The same event gets identical features regardless of batchmates."""
        manager = fitted_detector.baseline_manager
        eval_df = load_sample_dataset('suspicious').head(20)
        single = eval_df.iloc[[0]]

        _, cols = engineer_features(eval_df, manager)
        out_single, _ = engineer_features(single, manager)
        out_batch, _ = engineer_features(eval_df, manager)

        for col in cols:
            assert out_single[col].iloc[0] == pytest.approx(out_batch[col].iloc[0])

    def test_row_order_invariance(self, fitted_detector):
        """Predictions for an event do not depend on row ordering."""
        eval_df = pd.concat([
            load_sample_dataset('suspicious').head(30),
            load_sample_dataset('high_risk').head(30),
        ]).reset_index(drop=True)
        result_a = fitted_detector.predict(eval_df.copy())
        # Predict reversed rows with a clean index, then flip back for comparison
        result_b = fitted_detector.predict(eval_df.iloc[::-1].reset_index(drop=True))
        scores_b = result_b['anomaly_score'].values[::-1]
        preds_b = result_b['is_anomaly'].values[::-1]

        np.testing.assert_allclose(
            result_a['anomaly_score'].values,
            scores_b,
            rtol=1e-8,
        )
        assert (result_a['is_anomaly'].values == preds_b).all()

    def test_normalized_score_batch_invariant(self, fitted_detector):
        """anomaly_score_normalized uses fixed calibration, not batch min-max."""
        eval_df = load_sample_dataset('suspicious').head(40)
        single = eval_df.iloc[[0]]

        norm_single = fitted_detector.predict(single.copy())['anomaly_score_normalized'].iloc[0]
        norm_batch = fitted_detector.predict(eval_df.copy())['anomaly_score_normalized'].iloc[0]
        assert norm_single == pytest.approx(norm_batch)

    def test_unseen_user_fallback_consistent(self, baseline_manager):
        """Two different unseen users receive identical global fallback features."""
        df = pd.DataFrame([
            {'user_id': 'never_seen_a', 'source_ip': '10.1.1.1', 'event_type': 'login',
             'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 50,
             'data_mb': 1.0, 'timestamp': '2024-01-01 10:00:00'},
            {'user_id': 'never_seen_b', 'source_ip': '10.2.2.2', 'event_type': 'login',
             'courses_accessed': 1, 'accounts_accessed': 1, 'api_requests': 50,
             'data_mb': 1.0, 'timestamp': '2024-01-01 10:00:00'},
        ])
        out, _ = engineer_features(df, baseline_manager)
        user_cols = [c for c in out.columns
                     if c.startswith('user_') and c != 'user_id']
        for col in user_cols:
            assert out[col].iloc[0] == pytest.approx(out[col].iloc[1]), col

    def test_unseen_ip_and_missing_values_safe(self, fitted_detector):
        """Unknown IPs, NaN IPs, and missing event types must not break predict."""
        df = make_raw_df(n=20)
        df.loc[0, 'source_ip'] = np.nan
        df.loc[1, 'source_ip'] = '203.0.113.99'  # unseen TEST-NET IP
        df.loc[2, 'event_type'] = np.nan
        df.loc[3, 'user_id'] = np.nan

        result = fitted_detector.predict(df)
        assert len(result) == len(df)
        assert np.isfinite(result['anomaly_score']).all()

    def test_feature_schema_matches_training(self, fitted_detector):
        """Train and predict produce identical feature column sets."""
        eval_df = load_sample_dataset('suspicious').head(10)
        df_eng, cols = engineer_features(eval_df, fitted_detector.baseline_manager)
        assert cols == fitted_detector.feature_columns


class TestModelPersistence:
    """save()/load() must round-trip the complete model state."""

    def test_save_load_preserves_predictions(self, fitted_detector):
        eval_df = pd.concat([
            load_sample_dataset('suspicious').head(30),
            load_sample_dataset('high_risk').head(30),
        ])
        before = fitted_detector.predict(eval_df.copy())

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.joblib')
            fitted_detector.save(path)

            loaded = AnomalyDetector()
            loaded.load(path)

        after = loaded.predict(eval_df.copy())
        np.testing.assert_allclose(
            before['anomaly_score'].values, after['anomaly_score'].values, rtol=1e-8
        )
        assert (before['is_anomaly'].values == after['is_anomaly'].values).all()
        assert loaded.baseline_manager.user_baselines == \
            fitted_detector.baseline_manager.user_baselines
        assert loaded.feature_columns == fitted_detector.feature_columns

    def test_invalid_artifact_fails_safely(self, fitted_detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Wrong version
            path = os.path.join(tmpdir, 'bad_version.joblib')
            joblib.dump({'artifact_version': 999, 'model': fitted_detector.model}, path)
            with pytest.raises(ValueError, match="version"):
                AnomalyDetector().load(path)

            # Missing keys (legacy artifact shape)
            path2 = os.path.join(tmpdir, 'legacy.joblib')
            joblib.dump({
                'model': fitted_detector.model,
                'scaler': fitted_detector.scaler,
                'feature_columns': fitted_detector.feature_columns,
                'is_fitted': True,
            }, path2)
            with pytest.raises(ValueError):
                AnomalyDetector().load(path2)

            # Non-dict artifact
            path3 = os.path.join(tmpdir, 'junk.joblib')
            joblib.dump("not a model", path3)
            with pytest.raises(ValueError, match="dict"):
                AnomalyDetector().load(path3)

            # Detector must remain unfitted after failed loads
            d = AnomalyDetector()
            with pytest.raises(ValueError):
                d.load(path2)
            assert not d.is_fitted


class TestLogProcessorHardening:
    """log_processor must handle malformed input safely."""

    def test_load_security_log_bom_and_empty(self, tmp_path):
        # UTF-8 BOM file loads cleanly
        bom_file = tmp_path / 'bom.csv'
        bom_file.write_bytes(
            b'\xef\xbb\xbftimestamp,user_id\n2024-01-01,user_1\n'
        )
        df = load_security_log(str(bom_file))
        assert list(df.columns) == ['timestamp', 'user_id']

        # Empty file -> ValueError
        empty_file = tmp_path / 'empty.csv'
        empty_file.write_text('')
        with pytest.raises(ValueError):
            load_security_log(str(empty_file))

        # Missing file -> FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_security_log(str(tmp_path / 'nope.csv'))

    def test_detect_duplicates_without_timestamp(self):
        df = pd.DataFrame({'a': [1, 1, 2], 'b': ['x', 'x', 'y']})
        dupes = detect_duplicates(df)
        assert len(dupes) == 2

    def test_generate_timeline_string_timestamps(self):
        df = pd.DataFrame({
            'timestamp': ['2024-01-02 10:00:00', 'bad-date', '2024-01-01 09:00:00'],
            'user_id': ['u1', 'u2', 'u3'],
        })
        timeline = generate_timeline(df)
        assert len(timeline) == 2  # invalid timestamp dropped
        assert timeline.iloc[0]['user_id'] == 'u1'  # sorted desc

    def test_activity_functions_missing_columns(self):
        df = pd.DataFrame({'foo': [1, 2, 3]})
        assert 'error' in get_user_activity(df, 'u1')
        assert 'error' in get_ip_activity(df, '1.2.3.4')

    def test_search_events_literal_matching(self):
        """Regex metacharacters in the query are treated literally."""
        df = pd.DataFrame({
            'user_id': ['user.1', 'userx1', 'plain'],
            'event_type': ['login', 'login', 'login'],
        })
        result = search_events(df, 'user.1')
        assert list(result['user_id']) == ['user.1']
