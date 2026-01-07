# Training Module

**Fairness-aware model training with constraints and regularization**

## Overview

The Training Module integrates fairness directly into model training through:

- ✅ **ReductionsWrapper** - Fairlearn-based fairness constraints (ExponentiatedGradient)
- ✅ **FairnessRegularizedLoss** - PyTorch loss with fairness penalty
- ✅ **GroupFairnessCalibrator** - Post-training group-specific calibration
- ✅ **Visualization** - Pareto frontiers and trade-off analysis

## Quick Start

```python
from training_module import ReductionsWrapper
from sklearn.linear_model import LogisticRegression

# Train with fairness constraints
model = ReductionsWrapper(
    base_estimator=LogisticRegression(),
    constraint='demographic_parity',
    eps=0.05  # 5% tolerance
)

model.fit(X_train, y_train, sensitive_features=s_train)
y_pred = model.predict(X_test)
```

## Methods

### 1. Fairness Constraints (Sklearn)

**ReductionsWrapper** uses Fairlearn's ExponentiatedGradient algorithm to train models that satisfy fairness constraints.

#### Supported Constraints

- **Demographic Parity**: Equal positive prediction rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups  
- **Equal Opportunity**: Equal TPR across groups
- **Error Rate Parity**: Equal error rates across groups

#### Usage

```python
from training_module import ReductionsWrapper
from sklearn.linear_model import LogisticRegression

model = ReductionsWrapper(
    base_estimator=LogisticRegression(),
    constraint='demographic_parity',  # or 'equalized_odds', 'equal_opportunity'
    eps=0.01,  # Constraint violation tolerance (smaller = stricter)
    max_iter=50
)

model.fit(X_train, y_train, sensitive_features=s_train)
```

**Parameters**:
- `eps`: Maximum allowed constraint violation (0.01 = 1%)
- `max_iter`: Maximum optimization iterations
- `constraint`: Fairness constraint type

**Trade-off**: Smaller `eps` → more fair but potentially lower accuracy

### 2. Fairness Regularization (PyTorch)

**FairnessRegularizedLoss** adds a fairness penalty to the standard loss function.

**Loss Formula**:
```
L_total = L_accuracy + λ * L_fairness
```

where `L_fairness = (mean_pred_group0 - mean_pred_group1)^2`

#### Usage

```python
import torch
from training_module import FairnessRegularizedLoss

# Create loss
criterion = FairnessRegularizedLoss(
    fairness_weight=0.5,  # λ parameter
    fairness_type='demographic_parity'
)

# In training loop
for epoch in range(num_epochs):
    logits = model(X_batch)
    loss = criterion(logits, y_batch, sensitive_batch)
    loss.backward()
    optimizer.step()
```

**Parameters**:
- `fairness_weight`: Weight for fairness penalty (0=ignore, 1=equal to accuracy)
- `base_loss`: Standard loss (default: BCEWithLogitsLoss)

**Trade-off**: Higher `fairness_weight` → more fair but potentially lower accuracy

### 3. Group Calibration

**GroupFairnessCalibrator** calibrates predictions separately for each protected group to ensure fair confidence scores.

#### Methods

- **Isotonic Regression**: Non-parametric, flexible
- **Platt Scaling**: Parametric (sigmoid), faster

#### Usage

```python
from training_module import GroupFairnessCalibrator
from sklearn.linear_model import LogisticRegression

# Train base model
base_model = LogisticRegression()
base_model.fit(X_train, y_train)

# Calibrate on validation set
calibrator = GroupFairnessCalibrator(
    base_estimator=base_model,
    method='isotonic'  # or 'sigmoid'
)

calibrator.fit(X_val, y_val, sensitive_features=s_val)

# Get calibrated predictions
proba = calibrator.predict_proba(X_test, sensitive_features=s_test)
```

**When to use**:
- After training to improve probability calibration
- When confidence scores matter (e.g., ranking, thresholding)
- To ensure equal calibration across groups

### 4. Visualization

#### Pareto Frontier

Visualize accuracy-fairness trade-offs across different hyperparameters.

```python
from training_module import plot_pareto_frontier

results = []
for weight in [0.0, 0.1, 0.5, 1.0]:
    model = train_with_fairness_weight(weight)
    results.append({
        'accuracy': evaluate(model),
        'fairness': measure_bias(model),
        'param': weight
    })

fig = plot_pareto_frontier(results)
fig.savefig('pareto_frontier.png')
```

#### Fairness Comparison

Compare multiple models side-by-side.

```python
from training_module import plot_fairness_comparison

models = {
    'Baseline': {'demographic_parity': 0.25, 'equalized_odds': 0.20},
    'Fair': {'demographic_parity': 0.08, 'equalized_odds': 0.10}
}

fig = plot_fairness_comparison(models)
```

## Examples

### Example 1: End-to-End Fair Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from training_module import ReductionsWrapper
from measurement_module import FairnessAnalyzer

# Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Train fair model
model = ReductionsWrapper(
    base_estimator=LogisticRegression(),
    constraint='demographic_parity',
    eps=0.05
)

model.fit(X_train, y_train, sensitive_features=s_train)

# Evaluate
analyzer = FairnessAnalyzer()
result = analyzer.compute_metric(
    y_test, model.predict(X_test), s_test,
    metric='demographic_parity'
)

print(f"Accuracy: {model.score(X_test, y_test):.3f}")
print(f"Fair: {result.is_fair}")
print(f"DP Difference: {result.value:.3f}")
```

### Example 2: Grid Search Over Constraints

```python
from training_module import GridSearchReductions
from sklearn.linear_model import LogisticRegression

grid_search = GridSearchReductions(
    base_estimator=LogisticRegression(),
    constraints=['demographic_parity', 'equalized_odds'],
    eps_values=[0.01, 0.05, 0.1]
)

grid_search.fit(X_train, y_train, sensitive_features=s_train)

print(f"Best model: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
y_pred = grid_search.predict(X_test)
```

### Example 3: PyTorch Training

```python
import torch
import torch.nn as nn
from training_module import FairnessRegularizedLoss

# Define model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Fair loss
criterion = FairnessRegularizedLoss(fairness_weight=0.5)
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    logits = model(X_batch)
    loss = criterion(logits, y_batch, s_batch)
    loss.backward()
    optimizer.step()
```

### Example 4: Calibration Pipeline

```python
from training_module import GroupFairnessCalibrator

# Split data: train, calibration, test
X_train, X_temp, y_train, y_temp, s_train, s_temp = train_test_split(
    X, y, s, test_size=0.4
)
X_cal, X_test, y_cal, y_test, s_cal, s_test = train_test_split(
    X_temp, y_temp, s_temp, test_size=0.5
)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Calibrate
calibrator = GroupFairnessCalibrator(model, method='isotonic')
calibrator.fit(X_cal, y_cal, sensitive_features=s_cal)

# Test
proba = calibrator.predict_proba(X_test, sensitive_features=s_test)
```

## API Reference

### ReductionsWrapper

```python
ReductionsWrapper(
    base_estimator,
    constraint='demographic_parity',
    eps=0.01,
    max_iter=50,
    nu=None,
    eta0=2.0
)
```

**Methods**:
- `fit(X, y, sensitive_features)` - Train with constraints
- `predict(X)` - Predict labels
- `predict_proba(X)` - Predict probabilities (if supported)
- `score(X, y)` - Calculate accuracy

### FairnessRegularizedLoss

```python
FairnessRegularizedLoss(
    base_loss=nn.BCEWithLogitsLoss(),
    fairness_weight=0.5,
    fairness_type='demographic_parity'
)
```

**Methods**:
- `forward(logits, targets, sensitive_features)` - Compute loss

### GroupFairnessCalibrator

```python
GroupFairnessCalibrator(
    base_estimator,
    method='isotonic',
    cv=5
)
```

**Methods**:
- `fit(X, y, sensitive_features)` - Fit calibrators
- `predict_proba(X, sensitive_features)` - Calibrated probabilities
- `predict(X, sensitive_features)` - Calibrated predictions
- `score(X, y, sensitive_features)` - Accuracy

## Testing

```bash
# Run tests
python test_training_module.py

# Expected output:
# ✅ ReductionsWrapper trains with fairness constraints
# ✅ GroupFairnessCalibrator calibrates per group
# ✅ Visualization functions work
# ✅ PyTorch fairness loss works (if torch available)
```

## Limitations (48-Hour Scope)

**Implemented**:
- ✅ Fairlearn reductions (demographic parity, equalized odds, equal opportunity)
- ✅ PyTorch demographic parity regularization
- ✅ Group-specific calibration (isotonic, sigmoid)
- ✅ Pareto frontier visualization

**Not Implemented** (documented for future):
- ❌ Adversarial debiasing
- ❌ Equalized odds PyTorch loss (requires label info during forward pass)
- ❌ Temperature scaling for neural networks
- ❌ Intersectional fairness constraints

## Dependencies

```bash
# Required
pip install fairlearn>=0.9.0  # For ReductionsWrapper
pip install scikit-learn>=1.3.0
pip install matplotlib>=3.7.0

# Optional
pip install torch>=2.0.0  # For FairnessRegularizedLoss
```

## Next Steps

After training module:
1. **Monitoring Module** → Production fairness tracking
2. **Integration** → End-to-end orchestration
3. **Demo Notebook** → Complete walkthrough

---

**Questions?** See test file for examples or main README for project overview.