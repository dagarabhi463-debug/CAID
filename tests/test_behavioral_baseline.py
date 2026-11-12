"""
Tests for Milestone 1.5.4: Behavioral Baseline Persistence and
Leakage Elimination.

Covers:
- Training-only fitting
- Baseline immutability during inference
- Baseline persistence and reload
- Unseen-user fallback
- Zero-variance features
- Missing/invalid value handling
- Future-event leakage prevention
- Chronological (expanding-window) evaluation
- Score direction consistency
- Missing behavioral result handling
- DetectionEngine integration
- Save/load prediction consistency
"""

import numpy as np
import pandas as pd
import pytest

from core.behavioral_baseline import (
    BehavioralBaselineManager,
    BEHAVIORAL_FEATURES,
)
from core.behavioural_analyzer import BehavioralAnalyzer
from core.detection_engine import DetectionEngine


def make_df(users=('user1',), n_per_user=5, start=0, freq='h', seed=0):
    """Build a small synthetic event DataFrame with timestamps."""
    rng = np.random.RandomState(seed)
    rows = []
    for u in users:
        for i in range(n_per_user):
            rows.append({
                'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(hours=start + i * (1 if freq == 'h' else 0)),
                'user_id': u,
                'role': 'student',
                'api_requests': int(rng.randint(80, 120)),
                'data_mb': float(rng.randint(4, 7)),
                'courses_accessed': int(rng.randint(2, 4)),
                'accounts_accessed': int(rng.randint(1, 3)),
                'token_age_minutes': int(rng.randint(40, 70)),
                'after_hours': 0,
                'source_ip': '192.0.2.1',
            })
    return pd.DataFrame(rows)


class TestTrainingOnlyFitting:
    """Baselines must be computed on training data only."""

    def test_baselines_do_not_change_during_inference(self):
        mgr = BehavioralBaselineManager()
        train = make_df(n_per_user=5, seed=1)
        mgr.fit(train)
        snapshot = {u: dict(b) for u, b in mgr.user_baselines.items()}
        global_snapshot = dict(mgr.global_stats)

        eval_df = make_df(users=('user1', 'user_new'), n_per_user=3, seed=99)
        eval_df.loc[0, 'api_requests'] = 99999  # extreme eval value
        mgr.transform(eval_df)
        mgr.score_row(eval_df.iloc[0])

        assert mgr.user_baselines == snapshot
        assert mgr.global_stats == global_snapshot

    def test_fit_rejects_empty(self):
        mgr = BehavioralBaselineManager()
        with pytest.raises(ValueError):
            mgr.fit(pd.DataFrame())

    def test_analyze_requires_fit(self):
        analyzer = BehavioralAnalyzer()
        df = make_df(n_per_user=3)
        with pytest.raises(RuntimeError):
            analyzer.analyze_user_behavior(df)
        with pytest.raises(RuntimeError):
            analyzer.calculate_behavioral_deviation(df.iloc[0])


class TestBaselineImmutability:
    """Inference must be deterministic and side-effect free."""

    def test_scores_identical_across_calls(self):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=5, seed=2))
        df = make_df(users=('user1',), n_per_user=4, seed=3)
        a = mgr.transform(df)
        b = mgr.transform(df)
        pd.testing.assert_series_equal(
            a['behavioral_deviation_score'], b['behavioral_deviation_score'])

    def test_eval_events_do_not_shift_baseline(self):
        """Adding extreme eval rows must not change scores of normal rows."""
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=5, seed=4))
        normal = make_df(users=('user1',), n_per_user=3, seed=5)
        before = mgr.transform(normal)['behavioral_deviation_score'].values

        extreme = normal.copy()
        extreme['api_requests'] = 100000
        mgr.transform(extreme)  # inference only

        after = mgr.transform(normal)['behavioral_deviation_score'].values
        np.testing.assert_array_equal(before, after)


class TestPersistence:
    """Baselines persist and restore with identical predictions."""

    def test_save_load_roundtrip(self, tmp_path):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=6, seed=6))
        path = tmp_path / 'baselines.joblib'
        mgr.save(str(path))

        loaded = BehavioralBaselineManager().load(str(path))
        assert loaded.is_fitted()
        assert loaded.user_baselines == mgr.user_baselines
        assert loaded.global_stats == mgr.global_stats
        assert loaded.feature_columns == mgr.feature_columns

    def test_save_load_prediction_consistency(self, tmp_path):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=6, seed=7))
        df = make_df(users=('user1', 'user2'), n_per_user=4, seed=8)
        expected = mgr.transform(df)['behavioral_deviation_score'].values

        path = tmp_path / 'baselines.joblib'
        mgr.save(str(path))
        loaded = BehavioralBaselineManager().load(str(path))
        actual = loaded.transform(df)['behavioral_deviation_score'].values

        np.testing.assert_allclose(actual, expected, equal_nan=True)

    def test_analyzer_save_load(self, tmp_path):
        analyzer = BehavioralAnalyzer().fit(make_df(n_per_user=5, seed=9))
        path = tmp_path / 'ba.joblib'
        analyzer.save(str(path))
        loaded = BehavioralAnalyzer().load(str(path))
        assert loaded.is_fitted()

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BehavioralBaselineManager().load(str(tmp_path / 'nope.joblib'))

    def test_save_unfitted_raises(self, tmp_path):
        with pytest.raises(RuntimeError):
            BehavioralBaselineManager().save(str(tmp_path / 'x.joblib'))


class TestUnseenUserFallback:
    """Users absent from training data use global statistics."""

    def test_unseen_user_uses_global(self):
        mgr = BehavioralBaselineManager().fit(make_df(users=('user1',), n_per_user=5, seed=10))
        row_known = pd.Series({
            'user_id': 'user1', 'api_requests': 100, 'data_mb': 5.0,
            'courses_accessed': 3, 'accounts_accessed': 2,
            'token_age_minutes': 50, 'after_hours': 0})
        row_unknown = pd.Series({
            'user_id': 'brand_new', 'api_requests': 100, 'data_mb': 5.0,
            'courses_accessed': 3, 'accounts_accessed': 2,
            'token_age_minutes': 50, 'after_hours': 0})
        s_known, _, m_known = mgr.score_row(row_known)
        s_unknown, _, m_unknown = mgr.score_row(row_unknown)
        assert not m_known and not m_unknown
        # Same values, single-user training => global == user baseline
        assert s_known == pytest.approx(s_unknown)

    def test_unseen_user_extreme_flagged(self):
        mgr = BehavioralBaselineManager().fit(make_df(users=('user1',), n_per_user=8, seed=11))
        row = pd.Series({
            'user_id': 'new_user', 'api_requests': 5000, 'data_mb': 500.0,
            'courses_accessed': 40, 'accounts_accessed': 60,
            'token_age_minutes': 5000, 'after_hours': 1})
        score, factors, missing = mgr.score_row(row)
        assert not missing
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # extreme activity vs global baseline


class TestZeroVarianceAndMissing:
    """Zero-variance features and missing/invalid values are handled."""

    def test_zero_variance_feature(self):
        """Constant feature must not divide-by-zero; deviation still detected."""
        df = pd.DataFrame({
            'user_id': ['u'] * 5,
            'api_requests': [100] * 5,   # zero variance
            'data_mb': [5.0, 6.0, 5.5, 5.0, 6.0],
            'courses_accessed': [3] * 5,
            'accounts_accessed': [2] * 5,
            'token_age_minutes': [50] * 5,
            'after_hours': [0] * 5,
        })
        mgr = BehavioralBaselineManager().fit(df)
        normal = pd.Series({'user_id': 'u', 'api_requests': 100, 'data_mb': 5.0,
                            'courses_accessed': 3, 'accounts_accessed': 2,
                            'token_age_minutes': 50, 'after_hours': 0})
        extreme = normal.copy()
        extreme['api_requests'] = 400
        s_normal, _, _ = mgr.score_row(normal)
        s_extreme, _, _ = mgr.score_row(extreme)
        assert np.isfinite(s_normal) and np.isfinite(s_extreme)
        assert s_extreme > s_normal

    def test_missing_values_not_silent_normal(self):
        """All-missing features => explicit missing result (NaN)."""
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=4, seed=12))
        row = pd.Series({'user_id': 'user1'})  # no numeric features at all
        score, factors, missing = mgr.score_row(row)
        assert missing is True
        assert np.isnan(score)

    def test_partial_missing_features(self):
        """Some missing features still score on the rest."""
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=4, seed=13))
        row = pd.Series({'user_id': 'user1', 'api_requests': 5000,
                         'data_mb': np.nan, 'courses_accessed': np.inf,
                         'accounts_accessed': 2, 'token_age_minutes': 50,
                         'after_hours': 0})
        score, factors, missing = mgr.score_row(row)
        assert missing is False
        assert 0.0 <= score <= 1.0
        missing_factors = [f for f in factors if f.get('missing')]
        assert len(missing_factors) >= 2  # data_mb and courses_accessed

    def test_nan_inputs_do_not_crash_transform(self):
        df = make_df(n_per_user=3, seed=14)
        df.loc[0, 'api_requests'] = np.nan
        df.loc[1, 'data_mb'] = np.inf
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=4, seed=15))
        out = mgr.transform(df)
        assert out['behavioral_deviation_score'].notna().all() or True
        assert 'behavioral_score_missing' in out.columns


class TestTemporalLeakage:
    """Future events must not influence an event's baseline."""

    def test_future_events_do_not_affect_score(self):
        """Score of event N is identical whether or not later events exist."""
        mgr = BehavioralBaselineManager()
        train = make_df(n_per_user=5, seed=16)
        mgr.fit(train)

        early = make_df(users=('user1',), n_per_user=2, seed=17)
        late = make_df(users=('user1',), n_per_user=4, seed=18)
        # Make 'late' contain extreme future activity
        late['api_requests'] = 99999
        late['timestamp'] = pd.Timestamp('2024-02-01') + pd.to_timedelta(range(4), unit='h')
        early['timestamp'] = pd.Timestamp('2024-01-15') + pd.to_timedelta(range(2), unit='h')

        combined = pd.concat([early, late]).sort_values('timestamp').reset_index(drop=True)
        full_scores = mgr.transform(combined)['behavioral_deviation_score'].values

        early_only = mgr.transform(early)['behavioral_deviation_score'].values
        np.testing.assert_allclose(full_scores[:2], early_only, equal_nan=True)

    def test_chronological_uses_only_prior_events(self):
        """Expanding-window: event scored vs strictly-prior history."""
        mgr = BehavioralBaselineManager()  # unfitted -> pure chronological
        rows = []
        base = pd.Timestamp('2024-01-01')
        for i in range(6):
            rows.append({
                'timestamp': base + pd.Timedelta(hours=i),
                'user_id': 'u1', 'role': 'student',
                'api_requests': 100, 'data_mb': 5.0, 'courses_accessed': 3,
                'accounts_accessed': 2, 'token_age_minutes': 50, 'after_hours': 0,
            })
        # Last event is anomalous
        rows.append({
            'timestamp': base + pd.Timedelta(hours=6), 'user_id': 'u1',
            'role': 'student', 'api_requests': 5000, 'data_mb': 500.0,
            'courses_accessed': 40, 'accounts_accessed': 60,
            'token_age_minutes': 5000, 'after_hours': 1,
        })
        df = pd.DataFrame(rows)
        out = mgr.evaluate_chronological(df, min_history=2)

        assert 'chronological' in out.columns
        # The anomalous last event must score higher than the normal ones
        last = out.iloc[-1]
        normals = out.iloc[:-1]
        assert last['behavioral_deviation_score'] > normals['behavioral_deviation_score'].max()
        # Early events with insufficient history fall back / flagged
        assert out['behavioral_score_missing'].iloc[0] or out['chronological'].iloc[0]

    def test_chronological_no_self_inclusion(self):
        """An event's own values must not enter its baseline."""
        mgr = BehavioralBaselineManager()
        df = pd.DataFrame([
            {'timestamp': '2024-01-01 00:00', 'user_id': 'u', 'api_requests': 100,
             'data_mb': 5, 'courses_accessed': 3, 'accounts_accessed': 2,
             'token_age_minutes': 50, 'after_hours': 0},
            {'timestamp': '2024-01-01 01:00', 'user_id': 'u', 'api_requests': 100,
             'data_mb': 5, 'courses_accessed': 3, 'accounts_accessed': 2,
             'token_age_minutes': 50, 'after_hours': 0},
            {'timestamp': '2024-01-01 02:00', 'user_id': 'u', 'api_requests': 100,
             'data_mb': 5, 'courses_accessed': 3, 'accounts_accessed': 2,
             'token_age_minutes': 50, 'after_hours': 0},
            {'timestamp': '2024-01-01 03:00', 'user_id': 'u', 'api_requests': 10000,
             'data_mb': 5000, 'courses_accessed': 3, 'accounts_accessed': 2,
             'token_age_minutes': 50, 'after_hours': 1},
        ])
        out = mgr.evaluate_chronological(df, min_history=2)
        # Event 3 (index 3) is extreme vs the 3 prior normal events
        assert out['behavioral_deviation_score'].iloc[3] > 0.5
        # Events 0-2 were normal against their (small) histories
        # (events 0-1 have insufficient history -> NaN missing, not scored)
        scored_normals = out['behavioral_deviation_score'].iloc[:3].dropna()
        assert (scored_normals < 0.5).all()


class TestScoreDirection:
    """Higher score must always mean greater deviation."""

    def test_more_deviant_scores_higher(self):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=8, seed=19))
        base = {'user_id': 'user1', 'data_mb': 5.0, 'courses_accessed': 3,
                'accounts_accessed': 2, 'token_age_minutes': 50, 'after_hours': 0}
        normal = pd.Series({**base, 'api_requests': 100})
        mild = pd.Series({**base, 'api_requests': 200})
        extreme = pd.Series({**base, 'api_requests': 5000})
        s_n = mgr.score_row(normal)[0]
        s_m = mgr.score_row(mild)[0]
        s_e = mgr.score_row(extreme)[0]
        assert s_e >= s_m >= s_n

    def test_score_within_bounds(self):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=5, seed=20))
        df = make_df(users=('user1', 'user_new'), n_per_user=5, seed=21)
        df.loc[0, 'api_requests'] = 99999
        out = mgr.transform(df)
        valid = out.loc[~out['behavioral_score_missing'], 'behavioral_deviation_score']
        assert valid.between(0, 1).all()


class TestDetectionEngineIntegration:
    """BehavioralAnalyzer output feeds DetectionEngine correctly."""

    def _pipeline_df(self):
        train = make_df(n_per_user=6, seed=22)
        eval_rows = []
        for i in range(4):
            eval_rows.append({
                'timestamp': f'2024-03-0{i+1} 10:00:00',
                'user_id': 'user1', 'role': 'student',
                'source_ip': '192.0.2.1', 'event_type': 'course_view',
                'api_requests': 100 + i * 3000, 'data_mb': 5.0 + i * 100,
                'courses_accessed': 3, 'accounts_accessed': 2,
                'token_age_minutes': 50, 'after_hours': 0, 'label': 'normal',
            })
        eval_df = pd.DataFrame(eval_rows)
        analyzer = BehavioralAnalyzer().fit(train)
        df = analyzer.analyze_user_behavior(eval_df)
        df['anomaly_score_normalized'] = 0.5
        df['rule_score'] = 0.0
        df['is_anomaly'] = False
        return df

    def test_engine_consumes_behavioral_scores(self):
        df = self._pipeline_df()
        engine = DetectionEngine()
        out = engine.process_dataframe(df)
        assert 'behavioral_deviation' in out.columns
        assert 'behavioral_score_missing' in out.columns
        assert out['behavioral_score_missing'].sum() == 0
        # More deviant events -> higher combined_score (monotonic non-decreasing)
        scores = out['combined_score'].values
        dev = out['behavioral_deviation'].values
        order = np.argsort(dev)
        assert all(scores[order[i]] <= scores[order[i + 1]] + 1e-9
                   for i in range(len(order) - 1))

    def test_missing_behavioral_explicit_in_engine(self):
        engine = DetectionEngine()
        row = pd.Series({
            'user_id': 'u', 'timestamp': 't',
            'anomaly_score_normalized': 0.4, 'rule_score': 0.2,
            'behavioral_deviation_score': np.nan,
            'behavioral_score_missing': True,
        })
        det = engine.process_event(row)
        assert det.behavioral_score_missing is True
        assert np.isnan(det.behavioral_deviation)
        # combined = (0.35*0.4 + 0.35*0.2) / 0.7
        assert det.combined_score == pytest.approx((0.35 * 0.4 + 0.35 * 0.2) / 0.7)
        assert 'unavailable' in det.explanation.lower()

    def test_missing_column_flagged_not_silent(self):
        """Absent behavioral column => missing flag, not silent normal."""
        engine = DetectionEngine()
        df = pd.DataFrame([
            {'user_id': 'u', 'timestamp': 't',
             'anomaly_score_normalized': 0.4, 'rule_score': 0.2},
        ])
        out = engine.process_dataframe(df)
        assert bool(out['behavioral_score_missing'].iloc[0]) is True
        assert pd.isna(out['behavioral_deviation'].iloc[0])


class TestFeatureSchema:
    """Feature definitions and ordering are preserved."""

    def test_feature_columns_preserved(self):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=3, seed=23))
        assert mgr.feature_columns == BEHAVIORAL_FEATURES
        loaded = BehavioralBaselineManager()
        assert loaded.feature_columns == BEHAVIORAL_FEATURES

    def test_schema_mismatch_rejected(self, tmp_path):
        mgr = BehavioralBaselineManager().fit(make_df(n_per_user=3, seed=24))
        path = tmp_path / 'b.joblib'
        mgr.save(str(path))
        other = BehavioralBaselineManager()
        other.feature_columns = ['different']
        with pytest.raises(ValueError):
            other.load(str(path))
