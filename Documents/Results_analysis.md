# Results Analysis

## Evaluation Metrics

The project evaluates performance using multiple metrics.

---

# Accuracy

```python
accuracy_score()
```

Measures:

```text
Correct Predictions
-------------------
Total Predictions
```

---

# Precision

Measures:

```text
How many predicted attacks
are actually attacks
```

---

# Recall

Measures:

```text
How many real attacks
were successfully detected
```

---

# F1 Score

Combines:

- Precision
- Recall

into a single metric.

---

# Cross Validation

```python
StratifiedKFold(
    n_splits=5
)
```

Used to evaluate model stability.

---

# Confusion Matrix

Structure:

| | Predicted Normal | Predicted Fake |
|----|----|----|
| Actual Normal | TN | FP |
| Actual Fake | FN | TP |

---

# Feature Importance

The Random Forest model identifies the most influential features.

Examples:

```text
sat_ratio
heading_abs_diff
velocity_diff
sat_discrepancy
```

---

# Prediction Confidence

The system generates prediction confidence using:

```python
predict_proba()
```

Example:

```text
Prediction: Fake
Confidence: 98.4%
```

---

# Expected Performance

Target:

```text
Accuracy > 97%
```

The framework automatically checks whether the target accuracy has been achieved.

---

# Generated Reports

The system automatically generates:

```text
confusion_matrix_test_set.png
confusion_matrix_full_dataset.png
model_performance.png
feature_importance.png

gps_predictions.csv
model_report.txt
```