"""Milestone 2.4 drift monitor tests — observational drift only."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.drift_monitor import (
    DriftConfig,
    DriftMonitor,
    STATUS_OK,
    STATUS_WARNING,
    STATUS_CRITICAL,
    STATUS_INSUFFICIENT,
)


def _ref_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        'api_requests': rng.poisson(50, n).astype(float),
        'data_mb': rng.gamma(2.0, 5.0, n),
        'event_type': rng.choice(['login', 'course_view', 'file_view'], n),
    })


class TestFitAndMeasure:
    def test_identical_distribution_is_ok(self):
        # n=1000 keeps PSI/KS sampling noise below warning bands;
        # at n=200 identical distributions can legitimately warn.
        mon = DriftMonitor().fit(_ref_df(n=1000, seed=1))
        rep = mon.measure(_ref_df(n=1000, seed=2))
        assert rep.status == STATUS_OK
        assert rep.max_psi < 0.25
        assert rep.reference_rows == 1000
        assert rep.current_rows == 1000
        assert 'api_requests' in rep.columns_measured

    def test_deterministic_report(self):
        mon = DriftMonitor().fit(_ref_df(n=500, seed=1))
        cur = _ref_df(n=500, seed=3)
        a = mon.measure(cur).to_dict()
        b = mon.measure(cur).to_dict()
        a.pop('measured_at_utc'); b.pop('measured_at_utc')
        assert a == b

    def test_shifted_distribution_detected(self):
        ref = _ref_df(seed=1)
        cur = _ref_df(seed=2)
        cur['api_requests'] = cur['api_requests'] + 500  # massive shift
        rep = DriftMonitor().fit(ref).measure(cur)
        assert rep.status in (STATUS_WARNING, STATUS_CRITICAL)
        api = [f for f in rep.feature_drift if f['column'] == 'api_requests'][0]
        assert api['severity'] in ('warning', 'critical')
        assert api['psi'] > 0.25 or api['ks'] > 0.2

    def test_categorical_drift(self):
        ref = _ref_df(seed=1)
        cur = _ref_df(seed=2)
        cur['event_type'] = 'login'  # collapse categories
        cfg = DriftConfig(categorical_columns=['event_type'])
        rep = DriftMonitor(cfg).fit(ref).measure(cur)
        et = [f for f in rep.feature_drift
              if f['column'] == 'event_type'][0]
        assert et['kind'] == 'categorical'
        assert et['tvd'] > 0.5


class TestExplicitStatus:
    def test_unfitted_monitor_reports_insufficient(self):
        rep = DriftMonitor().measure(_ref_df())
        assert rep.status == STATUS_INSUFFICIENT
        assert 'no fitted reference' in rep.summary

    def test_empty_current_reports_insufficient(self):
        mon = DriftMonitor().fit(_ref_df())
        rep = mon.measure(pd.DataFrame({'api_requests': []}))
        assert rep.status == STATUS_INSUFFICIENT

    def test_small_reference_rejected_at_fit(self):
        # Fewer usable values than min_reference_rows -> explicit
        # ValueError at fit time (no silent empty monitor).
        with pytest.raises(ValueError):
            DriftMonitor(DriftConfig(min_reference_rows=30)).fit(
                _ref_df(n=10))

    def test_missing_column_is_skipped_explicitly(self):
        mon = DriftMonitor().fit(_ref_df())
        cur = _ref_df().drop(columns=['data_mb'])
        rep = mon.measure(cur)
        assert any(s['column'] == 'data_mb' for s in rep.columns_skipped)
        assert 'data_mb' not in rep.columns_measured

    def test_all_nan_column_skipped(self):
        ref = _ref_df()
        ref['dead'] = np.nan
        mon = DriftMonitor().fit(ref)
        rep = mon.measure(_ref_df())
        assert any(s['column'] == 'dead' for s in rep.columns_skipped)


class TestContract:
    def test_fit_rejects_empty(self):
        with pytest.raises(ValueError):
            DriftMonitor().fit(pd.DataFrame({'a': []}))

    def test_no_mutation_of_inputs(self):
        ref = _ref_df()
        before = ref.copy()
        DriftMonitor().fit(ref).measure(_ref_df(seed=5))
        pd.testing.assert_frame_equal(ref, before)

    def test_report_is_not_detection_evidence(self):
        """Drift reports must never be mistaken for detections."""
        rep = DriftMonitor().fit(_ref_df()).measure(_ref_df(seed=9))
        d = rep.to_dict()
        assert d['status'] in (STATUS_OK, STATUS_WARNING,
                               STATUS_CRITICAL, STATUS_INSUFFICIENT)
        assert 'not evidence of malicious activity' in rep.summary

    def test_severity_bands(self):
        cfg = DriftConfig(psi_warn=0.05, psi_crit=0.5,
                          ks_warn=0.01, ks_crit=0.9)
        mon = DriftMonitor(cfg).fit(_ref_df(seed=1))
        rep = mon.measure(_ref_df(seed=2))
        # tiny drift -> warning band under tight thresholds
        assert rep.status in (STATUS_OK, STATUS_WARNING)
