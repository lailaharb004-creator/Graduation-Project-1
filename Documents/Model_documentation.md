# Model Documentation

## Introduction

This project uses an ensemble of three machine learning models to detect GPS spoofing attacks.

The models were selected because they provide complementary strengths:

- Tree-Based Learning
- Neural Network Learning
- Gradient Boosting Learning

Combining them improves overall performance.

---

# Model 1: Random Forest

## Type

Ensemble Tree-Based Classifier

## Configuration

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
```

## How It Works

Random Forest builds multiple decision trees.

Each tree makes an independent prediction.

The final prediction is determined by majority voting.

### Advantages

- Robust to noisy data
- Handles nonlinear relationships
- Supports feature importance extraction
- Resistant to overfitting

---

# Model 2: Multi-Layer Perceptron

## Type

Artificial Neural Network

## Architecture

```text
Input Layer
      ↓
100 Neurons
      ↓
50 Neurons
      ↓
Output Layer
```

## Configuration

```python
MLPClassifier(
    hidden_layer_sizes=(100,50),
    activation='relu',
    solver='adam',
    max_iter=500,
    early_stopping=True
)
```

## Advantages

- Learns complex nonlinear patterns
- Adaptive optimization
- High predictive power

---

# Model 3: Hist Gradient Boosting

## Type

Boosting Algorithm

## Configuration

```python
HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=10,
    learning_rate=0.1
)
```

## How It Works

Each tree learns from previous errors.

The model gradually improves prediction quality.

### Advantages

- High accuracy
- Excellent generalization
- Fast training

---

# Ensemble Learning

## Voting Classifier

```python
VotingClassifier(
    voting='soft'
)
```

## Soft Voting

Each model predicts probabilities.

Example:

| Model | Fake Probability |
|---------|---------|
| Random Forest | 0.92 |
| MLP | 0.85 |
| HistGB | 0.88 |

Average:

```text
0.883
```

Final prediction:

```text
Fake
```

---

## Benefits

- Higher Accuracy
- Better Stability
- Lower Variance
- Improved Generalization