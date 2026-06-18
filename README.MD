# GPS Spoofing Detection System using Ensemble Machine Learning

## Overview

This project presents an optimized Machine Learning framework for detecting GPS Spoofing attacks in autonomous vehicle and intelligent transportation environments.

The system analyzes GPS-related measurements and automatically classifies records as either:

- Normal GPS Data
- Fake (Spoofed) GPS Data

To maximize detection accuracy, the framework combines advanced feature engineering, data preprocessing techniques, and an ensemble of multiple machine learning algorithms.

The final model uses Soft Voting Ensemble Learning to improve robustness and generalization compared to individual models.

---

## Project Structure

```text
GPS-Spoofing-Detector/
│
├── README.md
├── requirements.txt
├── PROJECT_1.py
├── AV-GPS-Dataset-1.csv
├── gps_predictions_20260618_165935.csv
├── model_report_20260618_165935.txt
|
├── Documents/
│   ├── Model_documentation.md
│   ├── Feature_engineering.md
│   ├── Results_analysis.md
|   ├── Security Threats on the Electronic Bracelets for the Convicts in Jordan.pdf
│
├── plots/
|   ├── confusion_matrix_full_dataset.png
|   ├── confusion_matrix_test_set.png
|   ├── feature_importance.csv
|   ├── feature_importance.png
|   ├── model_performance.png
|
├── models/
|   ├── best_hist_gradient_boosting_20260618_165935.pkl
|   ├── ensemble_model_20260618_165935.pkl
|   ├── preprocessing_20260618_165935.pkl
|   

```

---

## Installation

```bash
pip install -r requirements.txt
```
## Motivation

GPS Spoofing is one of the most dangerous attacks targeting navigation systems. Attackers can transmit counterfeit GPS signals that mislead vehicles, drones, or tracking devices into believing they are located somewhere else.

Such attacks may affect:

- Autonomous Vehicles
- Smart Transportation Systems
- Drones (UAVs)
- Fleet Tracking Systems
- Military Navigation Systems
- Electronic Monitoring Systems
- Location-Based Services

This project aims to automatically identify spoofed GPS data before it can influence critical decisions.

---

## Dataset

The project expects a CSV dataset containing GPS measurements.

```python
DATASET_FILE = "AV-GPS-Dataset-1.csv"
```

### Required Dataset Columns

| Column Name | Description |
|------------|-------------|
| Data Type | Target Label (0 = Normal, 1 = Fake) |
| Satellite Count | Number of visible satellites |
| Satellite Locks | Number of locked satellites |
| Velocity (m/s) | Vehicle speed |
| Heading (deg) | Vehicle heading |
| GPS Course | GPS direction |

---

## Project Pipeline

```text
Load Dataset
      ↓
Feature Engineering
      ↓
Data Quality Analysis
      ↓
Missing Value Handling
      ↓
Feature Scaling
      ↓
Feature Normalization
      ↓
Train Machine Learning Models
      ↓
Build Ensemble Model
      ↓
Model Evaluation
      ↓
Prediction Generation
      ↓
Visualization & Reporting
      ↓
Save Models
```

---

# Technologies Used

## Programming Language

- Python 3

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib
- Seaborn

## Machine Learning

- Scikit-Learn

## Model Persistence

- Joblib

---

# Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier
)

from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support
)

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

import joblib
```

---

# Feature Engineering

One of the most important components of the system is feature engineering.

The framework creates additional intelligent features from raw GPS measurements to improve spoofing detection.

---

## Satellite Ratio

```python
sat_ratio = sat_locks / sat_count
```

Represents the percentage of visible satellites that are successfully locked.

Spoofed signals often create abnormal ratios.

---

## Satellite Discrepancy

```python
sat_discrepancy = abs(
    sat_count - sat_locks
)
```

Measures inconsistencies between visible and locked satellites.

Large discrepancies may indicate spoofing activity.

---

## Velocity Difference

```python
velocity_diff
```

Calculates sudden changes in speed between consecutive records.

GPS spoofing often produces unrealistic velocity jumps.

---

## Heading Difference

```python
heading_abs_diff
```

Measures the difference between:

- Vehicle Heading
- GPS Course

Large differences may indicate manipulated GPS readings.

---

## Rolling Statistical Features

Window size:

```python
window = 5
```

Generated features:

```text
velocity_mean_5
velocity_std_5

sat_count_mean_5
sat_count_std_5

sat_ratio_mean_5
sat_ratio_std_5
```

These features help capture temporal behavior and signal stability.

---

# Data Preprocessing

Before model training, the data passes through several preprocessing stages.

---

## Missing Value Imputation

```python
SimpleImputer(strategy="median")
```

Benefits:

- Handles missing values
- Robust against outliers
- Prevents training failures

---

## Standardization

```python
StandardScaler()
```

Transforms features to:

```text
Mean = 0
Standard Deviation = 1
```

Benefits:

- Faster learning
- Improved neural network performance

---

## Power Transformation

```python
PowerTransformer(
    method="yeo-johnson"
)
```

Benefits:

- Reduces skewness
- Improves feature distributions
- Enhances model accuracy

---

# Machine Learning Models

The framework uses three optimized machine learning models.

---

## 1. Random Forest Classifier

### Model Type

Tree-Based Ensemble Learning

### Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
```

### Characteristics

- 300 Decision Trees
- High robustness
- Handles noisy data
- Supports feature importance analysis

### Advantages

- Strong performance on structured data
- Resistant to overfitting
- Good interpretability

---

## 2. Multi-Layer Perceptron (MLP)

### Model Type

Artificial Neural Network

### Architecture

```text
Input Layer
      ↓
Hidden Layer (100 Neurons)
      ↓
Hidden Layer (50 Neurons)
      ↓
Output Layer
```

### Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(100,50),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True
)
```

### Characteristics

- Deep nonlinear learning
- Learns complex relationships
- Automatic optimization using Adam

### Advantages

- Captures nonlinear patterns
- High predictive capability
- Reduced overfitting through Early Stopping

---

## 3. Hist Gradient Boosting Classifier

### Model Type

Boosting Algorithm

### Configuration

```python
HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=10,
    learning_rate=0.1
)
```

### Characteristics

- Sequential tree learning
- Learns from previous errors
- Faster than traditional gradient boosting

### Advantages

- High accuracy
- Excellent generalization
- Efficient training

---

# Ensemble Learning

The final detector uses Soft Voting Ensemble Learning.

```python
VotingClassifier(
    voting='soft'
)
```

## How It Works

Each model outputs prediction probabilities.

Example:

| Model | Fake Probability |
|---------|---------|
| Random Forest | 0.92 |
| Neural Network | 0.85 |
| Hist Gradient Boosting | 0.88 |

Average Probability:

```text
(0.92 + 0.85 + 0.88) / 3
= 0.883
```

Final Prediction:

```text
Fake
```

---

## Benefits of Ensemble Learning

- Higher accuracy
- Better generalization
- Lower variance
- More stable predictions
- Improved robustness

---

# Model Evaluation Metrics

The framework evaluates performance using multiple metrics.

## Accuracy

```python
accuracy_score()
```

Measures overall prediction correctness.

---

## Precision

Measures:

```text
How many detected attacks
are actually attacks
```

---

## Recall

Measures:

```text
How many real attacks
were successfully detected
```

---

## F1 Score

Harmonic mean of:

- Precision
- Recall

---

## Cross Validation

```python
StratifiedKFold(
    n_splits=5
)
```

Provides robust evaluation and measures model stability.

---

# Confusion Matrix

The system generates confusion matrices for both:

- Test Dataset
- Complete Dataset

Structure:

| | Predicted Normal | Predicted Fake |
|----|----|----|
| Actual Normal | TN | FP |
| Actual Fake | FN | TP |

Used to identify:

- False Positives
- False Negatives
- Detection reliability

---

# Feature Importance Analysis

Feature importance is extracted from the Random Forest model.

Example important features:

```text
sat_ratio
heading_abs_diff
velocity_diff
sat_discrepancy
sat_count_mean_5
velocity_std_5
```

This analysis helps explain model decisions and identify the most influential GPS indicators.

---

# Prediction Output

After training, the model predicts every GPS record.

Possible labels:

```text
normal
```

or

```text
fake
```

Each prediction includes a confidence score.

Example:

```text
Prediction : Fake
Confidence : 98.4%
```

---

# Generated Outputs

## Visualizations

```text
plots/
```

Generated files:

```text
confusion_matrix_test_set.png
confusion_matrix_full_dataset.png
model_performance.png
feature_importance.png
feature_importance.csv
```

---

## Saved Models

```text
models/
```

Generated files:

```text
ensemble_model.pkl
best_model.pkl
preprocessing.pkl
```

---

## Prediction Results

```text
gps_predictions_*.csv
```

Contains:

- Original records
- Predicted labels
- Confidence scores
- Prediction correctness

---

## Detailed Report

```text
model_report_*.txt
```

Contains:

- Dataset information
- Performance metrics
- Model configuration
- Feature summary
- Accuracy statistics

---

# Expected Performance

Target performance:

```text
Accuracy > 97%
```

The framework automatically checks whether the target accuracy has been achieved.

```python
if accuracy >= 97:
    print("TARGET ACHIEVED")
```

---

# Future Improvements

Potential future enhancements include:

- XGBoost Integration
- LightGBM Integration
- Real-Time GPS Monitoring
- Deep Learning Architectures
- Explainable AI (SHAP)
- Online Learning Systems
- Deployment as REST API
- Edge Computing Deployment

---

# Conclusion

This project provides a complete Machine Learning-based GPS Spoofing Detection Framework capable of identifying malicious GPS manipulation using advanced feature engineering, ensemble learning, neural networks, and boosting techniques.

By combining Random Forest, MLP Neural Networks, and Hist Gradient Boosting within a Soft Voting Ensemble, the system achieves highly accurate classification while generating detailed analytics, visualizations, prediction reports, and deployable trained models.

The framework is suitable for research, intelligent transportation systems, autonomous vehicles, UAV security, and GPS anomaly detection applications.