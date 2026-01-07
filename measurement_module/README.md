# Measurement Module

**Statistical fairness metrics with uncertainty quantification**

## Overview

The Measurement Module is the foundation of the fairness pipeline, providing:

- ✅ **Three core fairness metrics** (demographic parity, equalized odds, equal opportunity)
- ✅ **Bootstrap confidence intervals** (uncertainty quantification)
- ✅ **Effect sizes** (Cohen's d, risk ratios)
- ✅ **Statistical significance tests**
- ✅ **Unified API** through `FairnessAnalyzer`

## Quick Start

```python
from measurement_module import FairnessAnalyzer

# Initialize analyzer
analyzer = FairnessAnalyzer()

# Compute fairness metric with statistical validation
result = analyzer.compute_metric(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test['gender'],
    metric='demographic_parity',
    threshold=0.1
)

# Check results
print(f"Fair: {result.is_fair}")
print(f"Value: {result.value:.4f}")
print(f"95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
print(f"Interpretation: {result.interpretation}")
```

## Fairness Metrics

### 1. Demographic Parity

**Definition**: Groups should receive positive predictions at equal rates.

**Formula**: `|P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|`

**When to use**: When you want equal representation in positive outcomes, regardless of ground truth.

**Example**: Loan approvals should be equally distributed across demographic groups.

```python
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    threshold=0.1  # 10% difference threshold
)
```

### 2. Equalized Odds

**Definition**: Groups should have equal true positive rates AND false positive rates.

**Formula**: `max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)`

**When to use**: When both types of errors matter equally.

**Example**: Criminal risk assessment should have equal accuracy across groups.

```python
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='equalized_odds',
    threshold=0.1
)
```

### 3. Equal Opportunity

**Definition**: Groups should have equal true positive rates.

**Formula**: `|TPR_0 - TPR_1|`

**When to use**: When false negatives are more costly than false positives.

**Example**: Disease screening should catch disease equally across groups.

```python
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='equal_opportunity',
    threshold=0.1
)
```

## Statistical Validation

### Bootstrap Confidence Intervals

Every metric includes a 95% confidence interval computed via bootstrap resampling (1000 samples by default):

```python
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity'
)

print(f"Point estimate: {result.value:.4f}")
print(f"95% CI: {result.confidence_interval}")
```

**Interpretation**:
- If CI includes 0, difference may not be statistically meaningful
- Narrow CI = precise estimate
- Wide CI = uncertain estimate (may need more data)

### Effect Sizes

Cohen's d measures standardized difference between groups:

```python
# Computed automatically for demographic parity
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    compute_effect_size=True
)

print(f"Cohen's d: {result.effect_size:.3f}")
```

**Interpretation**:
- |d| < 0.2: negligible
- |d| < 0.5: small
- |d| < 0.8: medium
- |d| ≥ 0.8: large

## API Reference

### `FairnessAnalyzer`

Main entry point for fairness measurement.

```python
analyzer = FairnessAnalyzer(
    confidence_level=0.95,      # 95% confidence intervals
    bootstrap_samples=1000,     # Number of bootstrap iterations
    min_group_size=30          # Minimum samples per group
)
```

#### `compute_metric()`

Compute single fairness metric with full statistical validation.

**Parameters**:
- `y_true`: True labels (binary)
- `y_pred`: Predicted labels (binary)
- `sensitive_features`: Protected attribute (binary)
- `metric`: Metric name ('demographic_parity', 'equalized_odds', 'equal_opportunity')
- `threshold`: Fairness threshold (default: 0.1)
- `compute_ci`: Whether to compute confidence interval (default: True)
- `compute_effect_size`: Whether to compute effect size (default: True)

**Returns**: `FairnessMetricResult` object

#### `compute_all_metrics()`

Compute all three fairness metrics at once.

```python
results = analyzer.compute_all_metrics(
    y_true, y_pred, sensitive_features,
    threshold=0.1
)

for metric_name, result in results.items():
    print(f"{metric_name}: {result.value:.4f} (fair: {result.is_fair})")
```

#### `analyze_group_metrics()`

Get detailed per-group statistics (TPR, FPR, precision, etc.).

```python
group_metrics = analyzer.analyze_group_metrics(
    y_true, y_pred, sensitive_features
)

for group, metrics in group_metrics.items():
    print(f"{group}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
```

## FairnessMetricResult Schema

```python
@dataclass
class FairnessMetricResult:
    metric_name: str                        # 'demographic_parity', etc.
    value: float                            # Metric value
    confidence_interval: Tuple[float, float] # (lower, upper)
    group_metrics: Dict[str, float]         # Per-group rates
    group_sizes: Dict[str, int]             # Sample sizes
    interpretation: str                     # Human-readable explanation
    is_fair: bool                           # Passes threshold?
    threshold: float                        # Threshold used
    effect_size: Optional[float]            # Cohen's d
    timestamp: datetime                     # When computed
```

## Advanced Usage

### Custom Thresholds

Different use cases may require different thresholds:

```python
# High-stakes application (strict)
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    threshold=0.05  # 5% difference
)

# Initial pilot (lenient)
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    threshold=0.20  # 20% difference
)
```

### Speed vs Precision Trade-off

```python
# Fast (for iteration)
analyzer = FairnessAnalyzer(bootstrap_samples=100)

# Precise (for final analysis)
analyzer = FairnessAnalyzer(bootstrap_samples=5000)

# Skip CI entirely (fastest)
result = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    compute_ci=False
)
```

### Intersectional Analysis

Analyze fairness across multiple protected attributes:

```python
result = analyzer.compute_intersectional_metrics(
    y_true, y_pred,
    sensitive_features_1=df['gender'],
    sensitive_features_2=df['race'],
    metric='demographic_parity'
)
```

⚠️ **Warning**: Requires large sample sizes. Small groups yield unreliable estimates.

## Testing

Run comprehensive tests:

```bash
python test_measurement_module.py
```

Expected output:
```
============================================================
Testing Measurement Module
============================================================

[1/6] Testing imports...
✅ All imports successful

[2/6] Generating test data...
✅ Generated 500 samples

[3/6] Testing metric computation...
✅ Demographic Parity: 0.1234
✅ Equalized Odds: 0.0987
✅ Equal Opportunity: 0.0876

[4/6] Testing bootstrap confidence intervals...
✅ Bootstrap CI computed
   95% CI: [0.0890, 0.1578]

[5/6] Testing effect size computation...
✅ Cohen's d: 0.3214

[6/6] Testing FairnessAnalyzer (main API)...
✅ All tests passed!
```

## Limitations (48-Hour Scope)

Current implementation:
- ✅ Binary classification only
- ✅ Binary protected attributes only
- ✅ Three core metrics (DP, EO, EqOpp)
- ✅ Bootstrap CI (no parametric alternatives)

Not implemented (documented extensions):
- ❌ Multi-class classification
- ❌ Regression fairness metrics
- ❌ Multiple protected attributes (except intersectional)
- ❌ Calibration fairness metrics
- ❌ Individual fairness metrics

## References

1. **Demographic Parity**: Dwork et al. (2012) "Fairness Through Awareness"
2. **Equalized Odds**: Hardt et al. (2016) "Equality of Opportunity in Supervised Learning"
3. **Bootstrap Methods**: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
4. **Effect Sizes**: Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"

## Next Steps

After measurement module:
1. **Pipeline Module** → Bias detection and mitigation transformers
2. **Training Module** → Fairness-constrained model training
3. **Monitoring Module** → Production fairness tracking
4. **Integration** → End-to-end orchestration

---

**Questions?** See main project README or contact FairML Consulting.