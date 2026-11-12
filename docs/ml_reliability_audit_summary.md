# CAID - Machine Learning Reliability and Evaluation Audit (Milestone 1.5)

## Executive Summary

This audit implemented a comprehensive machine learning reliability framework for the CAID system. The implementation separates ML anomaly detection from business rule-based risk classification, ensuring proper data flow, model persistence, and evaluation methodology.

**Test Results:** 50/50 tests passing

---

## 1. Architecture Overview

### ML Anomaly Detection (Isolation Forest)

**File:** `models/anomaly_detector.py`

The anomaly detector implements a clean separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AnomalyDetector Class                            │
├─────────────────────────────────────────────────────────────────────┤
│ • fit()        → Train on baseline data (normal_activity.csv)      │
│ • predict()    → Use fitted model for predictions (no training)    │
│ • save()       → Persist model and preprocessing state             │
│ • load()       → Restore model and preprocessing state             │
│ • evaluate()   → Compute metrics using ground truth labels         │
└─────────────────────────────────────────────────────────────────────┘
```

**Key Principles:**
- **No training in predict()**: The predict method ONLY performs inference
- **Data leakage prevention**: Training on baseline data only
- **Feature consistency**: Same columns, same order, same preprocessing
- **Model persistence**: Saved in `models/saved_models/anomaly_detector.joblib`

### Risk Engine (Business Rules)

**File:** `core/risk_engine.py`

The risk engine combines ML predictions with business rules:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RiskEngine Class                                 │
├─────────────────────────────────────────────────────────────────────┤
│ • ML Anomaly Score (40%)    → Statistical outlier detection        │
│ • API Volume (20%)          → High request count                   │
│ • Data Transfer (15%)       → Large data transfers                 │
│ • Course Access (10%)       → Broad access patterns                │
│ • Account Access (10%)      → Multiple account access              │
│ • After Hours (5%)          → Off-hours activity                   │
├─────────────────────────────────────────────────────────────────────┤
│ Percentile-based thresholds (relative to current dataset)          │
│ • ≥85th percentile = Critical                                      │
│ • ≥65th percentile = High                                          │
│ • ≥30th percentile = Medium                                        │
│ • <30th percentile = Low                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Implementation Details

### Task 1 - Model Training

**Before:** `predict()` contained implicit training logic:
```python
def predict(self, df):
    # ... train on input data
    self.model.fit(X_scaled)
```

**After:** Explicit separation:
```python
def fit(self, df):
    # Train ONLY on baseline data
    X = get_feature_matrix(df_engineered, feature_columns)
    X_scaled = self.scaler.fit_transform(X)
    self.model.fit(X_scaled)
    self.is_fitted = True

def predict(self, df):
    # Only predict using fitted model
    if not self.is_fitted:
        raise RuntimeError("Model must be fitted before prediction")
    X_scaled = self.scaler.transform(X)  # NOT fit_transform!
    predictions = self.model.predict(X_scaled)
```

### Task 2 - Training Data

**Training Data:** `data/raw/normal_activity.csv` (baseline)

**Evaluation Data:** 
- `data/raw/suspicious_activity.csv`
- `data/raw/high_risk_activity.csv`

**Validation:** The evaluation dataset is never used for training. The model learns normal patterns from the baseline and identifies deviations in evaluation data.

### Task 3 - Feature Consistency

**Validation Checks:**
1. **Missing values**: Handled with `np.nan_to_num(X, nan=0.0)`
2. **Feature ordering**: Same `feature_columns` list for fit and predict
3. **Numerical conversion**: Skipped (already numeric in source data)
4. **Scaling**: `StandardScaler` fitted on training data ONLY
5. **Unknown categories**: N/A (all features are continuous)
6. **Infinite values**: Handled with `np.nan_to_num(X, posinf=0.0, neginf=0.0)`

**Code:**
```python
def fit(self, df, feature_columns):
    X = get_feature_matrix(df_engineered, feature_columns)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    self.scaler.fit(X)  # Fit ONLY on training data
    X_scaled = self.scaler.transform(X)
    self.model.fit(X_scaled)

def predict(self, df):
    X = get_feature_matrix(df_engineered, self.feature_columns)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_scaled = self.scaler.transform(X)  # Transform, not fit_transform!
    predictions = self.model.predict(X_scaled)
```

### Task 4 - Model Persistence

**Save Path:** `models/saved_models/anomaly_detector.joblib`

**Loading Logic (in `app.py`):**
```python
def ensure_model_exists():
    model_path = get_model_path()
    
    # Check if saved model exists
    if os.path.exists(model_path):
        try:
            detector.load(model_path)
            return detector
        except Exception as e:
            print(f"Failed to load model: {e}. Training new...")
    
    # Train new model using baseline dataset
    df_normal = load_sample_dataset('normal')
    detector.fit(df_normal)
    detector.save(model_path)
    
    return detector
```

**Error Handling:**
- File not found: Train new model
- Corrupted file: Train new model
- Unfitted model: RuntimeError on save/load

### Task 5 - Model Evaluation

**Metrics Implemented:**
- **True Positives (TP)**: High-risk events correctly identified as anomalies
- **False Positives (FP)**: Normal events incorrectly flagged as anomalies
- **True Negatives (TN)**: Normal events correctly identified as normal
- **False Negatives (FN)**: High-risk events missed by anomaly detection

**Derived Metrics:**
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precision * Recall) / (Precision + Recall)
- **Accuracy**: (TP + TN) / Total

**Important Note:** This evaluation uses synthetic data only. Performance metrics should NOT be interpreted as real-world detection accuracy.

**Evaluation Method:**
```python
def evaluate(self, df, ground_truth_column='label', 
             positive_label='high_risk', negative_label='normal'):
    y_true = df[ground_truth_column].isin([positive_label]).astype(int)
    y_pred = df['is_anomaly'].astype(int)
    
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    # ... calculate all metrics
```

### Task 6 - Dashboard Updates

**Model Information Display:**
- Model type: Isolation Forest
- Model status: Fitted (with timestamp)
- Training dataset size: Number of records from normal_activity.csv
- Anomalies detected: Count of `is_anomaly == True`
- Anomaly percentage: `(anomalies / total) * 100`

**Evaluation Metrics Display:**
- Precision, Recall, F1-Score, Accuracy
- Confusion matrix visualization

**Classification Labels:**
1. **ML Anomaly Prediction**: Isolation Forest statistical outlier detection
2. **Risk Scoring**: Weighted combination of ML score + business rules
3. **Risk Classification**: Percentile-based categorization

### Task 7 - Risk Classification

**Percentile-Based Thresholds (Relative to Current Dataset):**
- The thresholds are NOT fixed values
- They adapt to the distribution of risk scores in the current dataset
- This prevents artificial distribution constraints

**Explanation in Dashboard:**
> "Risk level is assigned using PERCENTILE-based thresholds relative to the current dataset's risk score distribution. Thresholds: >=85th percentile = Critical, >=65th = High, >=30th = Medium, <30th = Low."

**Risk Factor Weight Breakdown:**
- ML Anomaly Score: 40%
- API Volume: 20%
- Data Transfer: 15%
- Course Access: 10%
- Account Access: 10%
- After Hours: 5%

### Task 8 - Testing

**Test Coverage (50 tests):**

**Anomaly Detector Tests (29 tests):**
- `TestAnomalyDetectorInit`: Default and custom initialization
- `TestAnomalyDetectorFit`: Fitting on baseline data, custom features, error handling
- `TestAnomalyDetectorPredict`: Prediction without retraining, missing features, consistency
- `TestAnomalyDetectorSaveLoad`: Model persistence and loading
- `TestAnomalyDetectorEvaluate`: Metrics calculation, confusion matrix
- `TestAnomalyDetectorFeatureConsistency`: Feature column consistency
- `TestEnsureModelExists`: Auto-train/load logic
- `TestRealWorldScenarios`: End-to-end pipelines

**Risk Engine Tests (21 tests):**
- `TestRiskEngineInit`: Initialization
- `TestRiskEngineFit`: Statistics calculation
- `TestRiskEngineCalculateRiskScore`: Weighted scoring
- `TestRiskEngineClassifyRisk`: Percentile-based thresholds
- `TestRiskEngineProcessEvents`: Full event processing
- `TestRiskEngineExplainClassification`: Explanation generation
- `TestRealWorldScenarios`: Risk factor breakdown

---

## 3. Technical Differences

### Anomaly Detection vs. Risk Scoring

| Aspect | Anomaly Detection | Risk Scoring |
|--------|-------------------|--------------|
| **Type** | Unsupervised ML | Rule-based scoring |
| **Training** | On baseline (normal) data only | No training required |
| **Input** | Feature matrix (numerical) | Any data with required columns |
| **Output** | Binary prediction + anomaly score | Weighted risk score (0-1) |
| **Purpose** | Identify statistical outliers | Combine ML with business rules |
| **Explainability** | "Statistical deviation from baseline" | "Weighted combination of factors" |

### Risk Scoring vs. Risk Classification

| Aspect | Risk Scoring | Risk Classification |
|--------|--------------|---------------------|
| **Type** | Numerical computation | Categorical assignment |
| **Output** | Float (0-1) | Enum: Low/Medium/High/Critical |
| **Method** | Weighted sum of factors | Percentile-based thresholds |
| **Dependency** | Independent | Depends on scoring |
| **Interpretation** | Absolute score | Relative to dataset distribution |

### Supervised Evaluation vs. Unsupervised Training

| Aspect | Training (Unsupervised) | Evaluation (Supervised) |
|--------|-------------------------|-------------------------|
| **Data** | Normal activity only | Labeled data (normal/suspicious/high_risk) |
| **Labels Used** | No | Yes (for ground truth) |
| **Goal** | Learn normal patterns | Measure detection accuracy |
| **Labels Leaked** | Never | Required for evaluation |

---

## 4. Files Modified/Created

### Modified Files

1. **`models/anomaly_detector.py`** - Complete rewrite with proper separation of fit/predict/save/load/evaluate
2. **`app.py`** - Integrated model persistence and evaluation metrics display
3. **`core/risk_engine.py`** - Added explain_classification method
4. **`modules/dashboard.py`** - Added model info display

### New Files Created

1. **`models/saved_models/`** - Directory for persisted models
2. **`models/saved_models/anomaly_detector.joblib`** - Fitted model (auto-created on first run)
3. **`tests/test_anomaly_detector.py`** - 29 comprehensive tests
4. **`tests/test_risk_engine.py`** - 21 comprehensive tests

---

## 5. Verification Steps

### Run Tests
```bash
python -m pytest tests/ -v
```

### Test End-to-End Pipeline
```python
from models.anomaly_detector import AnomalyDetector
from core.log_processor import load_sample_dataset

detector = AnomalyDetector()
normal = load_sample_dataset('normal')
detector.fit(normal)

eval_data = load_sample_dataset('suspicious')
result = detector.predict(eval_data)

eval_data['label'] = eval_data['label'].fillna('normal')
metrics = detector.evaluate(eval_data, ground_truth_column='label')
print(metrics)
```

### Run Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 6. Limitations and Next Steps

### Current Limitations

1. **Synthetic Data Evaluation**: All metrics are on synthetic data only
2. **Baseline Model**: Only normal_activity.csv used for training
3. **No Incremental Learning**: Model must be retrained from scratch

### Recommended Next Steps

1. **Real-World Validation**: Collect labeled production data
2. **Performance Monitoring**: Track false positive/negative rates over time
3. **Feature Importance**: Analyze which factors contribute most to risk scores
4. **Threshold Tuning**: Adjust business rule thresholds based on false positive analysis
5. **Model Retraining**: Periodic retraining with new baseline data

---

## 7. Conclusion

This audit successfully implemented a reliable machine learning architecture for CAID with:

- ✅ Proper separation of fit and predict operations
- ✅ Data leakage prevention (training on baseline only)
- ✅ Feature consistency guarantees
- ✅ Model persistence and auto-loading
- ✅ Comprehensive evaluation metrics
- ✅ Clear explanation of ML vs. rule-based classification
- ✅ Percentile-based relative thresholds
- ✅ 50 passing tests covering all scenarios

The implementation maintains the existing Streamlit dashboard while adding model status and evaluation metrics display. The architecture is now production-ready for synthetic data demonstration and can be extended for real-world deployment with labeled production data.
