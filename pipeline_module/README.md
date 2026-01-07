# Pipeline Module

**Bias detection and mitigation for data pipelines**

## Overview

The Pipeline Module provides systematic bias detection and mitigation transformers that integrate seamlessly with scikit-learn pipelines:

- ✅ **BiasDetector** - Identify three types of bias automatically
- ✅ **InstanceReweighting** - Balance groups through sample weighting  
- ✅ **GroupBalancer** - Balance groups through resampling
- ✅ **sklearn compatible** - Works in Pipeline objects

## Quick Start

```python
from pipeline_module import BiasDetector, InstanceReweighting

# Step 1: Detect bias
detector = BiasDetector()
results = detector.detect_all_bias_types(
    df,
    protected_attribute='gender'
)

# Check what was found
for bias_type, result in results.items():
    if result.detected:
        print(f"{bias_type}: {result.severity} severity")
        print(f"Recommendations: {result.recommendations}")

# Step 2: Mitigate with reweighting
reweighter = InstanceReweighting()
X, y, weights = reweighter.fit_transform(
    X_train, y_train,
    sensitive_features=sensitive_train
)

# Step 3: Train model with weights
model.fit(X, y, sample_weight=weights)
```

## Bias Detection

### BiasDetector

Detects three types of bias:

#### 1. Representation Bias

**What it detects**: When demographic distribution differs from expected.

**Example**: Dataset is 30% female when population is 51% female.

```python
detector = BiasDetector(representation_threshold=0.2)

result = detector.detect_representation_bias(
    df,
    protected_attribute='gender',
    reference_distribution={'Female': 0.51, 'Male': 0.49}
)

if result.detected:
    print(f"Severity: {result.severity}")
    print(f"Difference: {result.evidence['max_difference']:.1%}")
```

**Parameters**:
- `representation_threshold`: Maximum acceptable difference (default: 0.2 = 20%)
- `reference_distribution`: Expected distribution (default: uniform)

#### 2. Proxy Variable Detection

**What it detects**: Features highly correlated with protected attribute.

**Example**: ZIP code correlated 0.7 with race.

```python
result = detector.detect_proxy_variables(
    df,
    protected_attribute='race',
    feature_columns=['zip_code', 'income', 'age']
)

if result.detected:
    proxy_features = result.evidence['proxy_features']
    print(f"Found {len(proxy_features)} proxy variables")
    for feat in proxy_features:
        corr = result.evidence['correlations'][feat]['correlation']
        print(f"  {feat}: r={corr:.3f}")
```

**Parameters**:
- `proxy_threshold`: Minimum correlation to flag (default: 0.5)
- `statistical_alpha`: Significance level (default: 0.05)

#### 3. Statistical Disparity

**What it detects**: Features distributed differently across groups.

**Example**: Average income differs significantly between groups.

```python
result = detector.detect_statistical_disparity(
    df,
    protected_attribute='gender',
    feature_columns=['income', 'experience', 'education']
)

if result.detected:
    for feat in result.evidence['disparate_features']:
        test = result.evidence['test_results'][feat]
        print(f"{feat}: p={test['p_value']:.4f}, d={test['effect_size']:.3f}")
```

**Uses**: Mann-Whitney U test + Cohen's d effect size

### Comprehensive Detection

Run all checks at once:

```python
all_results = detector.detect_all_bias_types(
    df,
    protected_attribute='gender',
    reference_distribution={'Female': 0.5, 'Male': 0.5}
)

for bias_type, result in all_results.items():
    print(f"{bias_type}: {'DETECTED' if result.detected else 'OK'}")
```

## Bias Mitigation

### InstanceReweighting

**sklearn-compatible transformer** that assigns weights to samples to balance groups.

**How it works**: Weights samples inversely proportional to group size.

```python
from pipeline_module import InstanceReweighting

reweighter = InstanceReweighting(
    method='inverse_propensity',
    alpha=1.0  # 1.0 = full reweighting, 0.0 = no reweighting
)

# Fit and get weights
reweighter.fit(X_train, y_train, sensitive_features=sensitive_train)
weights = reweighter.get_sample_weights(sensitive_train)

# Or use fit_transform
X, y, weights = reweighter.fit_transform(
    X_train, y_train, sensitive_features=sensitive_train
)

# Train model with weights
model.fit(X, y, sample_weight=weights)
```

**Parameters**:
- `method`: 'inverse_propensity' or 'uniform'
- `alpha`: Smoothing parameter (1.0 = full, 0.0 = none)

**Example weights**:
```
Group A: 200 samples → weight = 1.5
Group B: 300 samples → weight = 1.0
Effective balance: 300 vs 300
```

### GroupBalancer

**Resample data** to achieve exact demographic parity.

More aggressive than reweighting - actually changes dataset size.

```python
from pipeline_module import GroupBalancer

balancer = GroupBalancer(
    strategy='oversample',  # or 'undersample'
    random_state=42
)

X_balanced, y_balanced = balancer.fit_resample(
    X_train, y_train,
    sensitive_features=sensitive_train
)

# Now train on balanced data
model.fit(X_balanced, y_balanced)
```

**Strategies**:
- `'oversample'`: Duplicate minority group samples (increases size)
- `'undersample'`: Reduce majority group samples (decreases size)

**When to use**:
- InstanceReweighting: When you want to keep all data
- GroupBalancer: When you want exact 50/50 split

## Integration Examples

### With sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pipeline_module import InstanceReweighting

# Note: Reweighting needs special handling in pipelines
reweighter = InstanceReweighting()

# Fit reweighter separately
reweighter.fit(X_train, y_train, sensitive_features=sensitive_train)
train_weights = reweighter.get_sample_weights(sensitive_train)

# Create pipeline for preprocessing + model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Fit with weights
pipeline.fit(X_train, y_train, model__sample_weight=train_weights)
```

### With Measurement Module

```python
from measurement_module import FairnessAnalyzer
from pipeline_module import BiasDetector, InstanceReweighting

# 1. Detect bias
detector = BiasDetector()
bias_results = detector.detect_all_bias_types(df, 'gender')

# 2. If bias detected, mitigate
if any(r.detected for r in bias_results.values()):
    reweighter = InstanceReweighting()
    X, y, weights = reweighter.fit_transform(
        X_train, y_train, sensitive_features=sensitive_train
    )
    model.fit(X, y, sample_weight=weights)
else:
    model.fit(X_train, y_train)

# 3. Measure fairness
analyzer = FairnessAnalyzer()
result = analyzer.compute_metric(
    y_test, model.predict(X_test), sensitive_test,
    metric='demographic_parity'
)

print(f"Fair after mitigation: {result.is_fair}")
```

## API Reference

### BiasDetector

```python
detector = BiasDetector(
    representation_threshold=0.2,  # 20% max difference
    proxy_threshold=0.5,           # 0.5 correlation
    statistical_alpha=0.05         # 5% significance
)
```

**Methods**:
- `detect_representation_bias(df, protected_attribute, reference_distribution)`
- `detect_proxy_variables(df, protected_attribute, feature_columns)`
- `detect_statistical_disparity(df, protected_attribute, feature_columns)`
- `detect_all_bias_types(df, protected_attribute, ...)`

### InstanceReweighting

```python
reweighter = InstanceReweighting(
    method='inverse_propensity',  # or 'uniform'
    alpha=1.0                     # smoothing
)
```

**Methods**:
- `fit(X, y, sensitive_features)` - Learn group weights
- `transform(X)` - Returns X unchanged
- `get_sample_weights(sensitive_features)` - Get weights array
- `fit_transform(X, y, sensitive_features)` - Returns (X, y, weights)

### GroupBalancer

```python
balancer = GroupBalancer(
    strategy='oversample',        # or 'undersample'
    target_distribution=None,     # None = uniform
    random_state=42
)
```

**Methods**:
- `fit_resample(X, y, sensitive_features)` - Returns (X_resampled, y_resampled)

## BiasDetectionResult Schema

```python
@dataclass
class BiasDetectionResult:
    bias_type: str              # 'representation', 'proxy', 'measurement'
    detected: bool              # Was bias found?
    severity: str               # 'low', 'medium', 'high'
    affected_groups: List[str]  # Which groups/features affected
    evidence: Dict              # Detailed statistics
    recommendations: List[str]  # Actionable suggestions
    timestamp: datetime
```

## Testing

Run comprehensive tests:

```bash
python test_pipeline_module.py
```

Expected output:
```
============================================================
Testing Pipeline Module
============================================================

[1/5] Testing imports...
✅ All imports successful

[2/5] Generating biased dataset...
✅ Generated 500 samples

[3/5] Testing BiasDetector...
   Representation Bias:
   - Detected: True
   - Severity: high
   - Max difference: 40.0%

   Proxy Variables:
   - Detected: True
   - Proxy features: ['income']

   Statistical Disparity:
   - Detected: True
   - Disparate features: ['income', 'credit_score']

[4/5] Testing InstanceReweighting...
   - Balance ratio: 0.857 → 1.000
✅ fit_transform works correctly

[5/5] Testing GroupBalancer...
✅ Resampling completed

============================================================
✅ ALL PIPELINE MODULE TESTS COMPLETED!
============================================================
```

## Limitations (48-Hour Scope)

Current implementation:
- ✅ Three bias types (representation, proxy, statistical)
- ✅ Binary protected attributes only
- ✅ Simple reweighting (inverse propensity)
- ✅ Oversampling/undersampling

Not implemented:
- ❌ SMOTE (Synthetic Minority Oversampling)
- ❌ Disparate Impact Remover (feature transformation)
- ❌ Advanced reweighting schemes
- ❌ Multiple protected attributes simultaneously

## Next Steps

After pipeline module:
1. **Training Module** → Fairness-constrained model training
2. **Monitoring Module** → Production fairness tracking
3. **Integration** → End-to-end orchestration

---

**Questions?** See main project README or test file for examples.