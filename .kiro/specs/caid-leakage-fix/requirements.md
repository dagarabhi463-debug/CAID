# Requirements Document

## Introduction

This bugfix addresses critical data leakage issues identified in the CAID repository audit. The issues cause model performance to be artificially inflated during testing because baselines are computed using ALL events (including those being evaluated), which violates the fundamental principle that baselines must be fixed before inference begins.

**Issues to Fix:**

1. **Feature Engineering Data Leakage**: `extract_user_features()` and `extract_ip_features()` compute aggregations using the entire dataset, causing future events to influence baselines
2. **Feature Column Order Mismatch**: No validation ensures feature columns are in the same order during training vs. prediction
3. **Prediction-time Scaler Refitting**: The scaler may refit during prediction instead of only transforming, causing data leakage
4. **Behavioral Baseline Future Event Leakage**: Per-user baselines include ALL events for a user, including the event being evaluated

## Glossary

- **C(X)**: Bug Condition - The condition where feature engineering uses all events including future ones, or baselines include events being evaluated
- **P(result)**: Property - Baselines must be computed from historical data only, not from the current DataFrame being evaluated
- **¬C(X)**: Non-buggy inputs - Cases where all events are from the training set (no future events)
- **Feature Engineering**: Process of extracting aggregations (user stats, IP frequency) from security log data
- **Baseline**: Statistical summary (mean, std) of user or IP behavior used for deviation calculation
- **Future Event Leakage**: When baselines include events that occur after the baseline computation time

## Requirements

### Feature Engineering Data Leakage Prevention

R1.1 WHEN `extract_user_features()` is called on a DataFrame THEN the system SHALL use pre-computed user baselines stored in `user_baselines` from training data only, NOT compute aggregations from the current DataFrame

R1.2 WHEN `extract_ip_features()` is called on a DataFrame THEN the system SHALL use pre-computed IP frequency baselines stored in `ip_frequency` from training data only, NOT compute frequency from the current DataFrame

R1.3 WHEN user or IP statistics cannot be computed from historical data (new user/IP not in baseline) THEN the system SHALL use global statistics from training data with appropriate handling

### Feature Column Order Validation

R2.1 WHEN `AnomalyDetector.fit()` is called THEN the system SHALL store the exact feature column names and their order in `self.feature_columns`

R2.2 WHEN `AnomalyDetector.predict()` is called THEN the system SHALL validate that feature columns match the stored order in `self.feature_columns`

R2.3 WHEN `AnomalyDetector.predict()` is called and feature columns are in different order than training THEN the system SHALL reorder features to match training order

### Prediction-time Scaler Refitting Prevention

R3.1 WHEN `AnomalyDetector.predict()` is called THEN the system SHALL call `scaler.transform()` ONLY, NEVER `scaler.fit_transform()` or `scaler.fit()`

R3.2 WHEN `AnomalyDetector.save()` is called THEN the system SHALL persist the scaler's fitted state in `_scaler_fitted` flag

R3.3 WHEN `AnomalyDetector.load()` is called THEN the system SHALL restore the scaler's fitted state from saved data

R3.4 WHEN `AnomalyDetector.predict()` is called and scaler is not fitted THEN the system SHALL raise `RuntimeError` indicating model must be fitted

### Behavioral Baseline Future Event Prevention

R4.1 WHEN `BehavioralAnalyzer.fit()` is called THEN the system SHALL store baselines computed from training data only

R4.2 WHEN `BehavioralAnalyzer.analyze_user_behavior()` is called to evaluate event N for user X THEN the system SHALL compute deviation using only events that occurred BEFORE event N for user X (historical data only), OR use overall statistics with a warning if no historical data exists

R4.3 WHEN `BehavioralAnalyzer.analyze_user_behavior()` is called and no historical data exists for user X THEN the system SHALL use overall statistics with a warning logged

## Unchanged Behavior (Regression Prevention)

R5.1 WHEN feature engineering is applied to a DataFrame THEN the system SHALL CONTINUE TO produce the same feature column names and structure as before (backward compatibility)

R5.2 WHEN `AnomalyDetector.save()` and `AnomalyDetector.load()` are called THEN the system SHALL CONTINUE TO persist and restore all model parameters, scaler state, and feature column information

R5.3 WHEN `BehavioralAnalyzer.analyze_user_behavior()` is called on a DataFrame WHERE all events are from the training set (no future events) THEN the system SHALL CONTINUE TO produce correct deviation scores

R5.4 WHEN user or IP statistics cannot be computed from historical data THEN the system SHALL CONTINUE TO fall back to global statistics with appropriate handling
