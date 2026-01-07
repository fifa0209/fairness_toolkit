# Statistical Methods

**A comprehensive guide to statistical validation techniques in the Fairness Pipeline Development Toolkit**

---

## Overview

This document explains the statistical methods used to validate fairness measurements, distinguish signal from noise, and quantify uncertainty in the toolkit's metrics.

---

## Table of Contents

1. [Why Statistical Validation Matters](#why-statistical-validation-matters)
2. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
3. [Effect Size Calculations](#effect-size-calculations)
4. [Hypothesis Testing](#hypothesis-testing)
5. [Multiple Comparison Correction](#multiple-comparison-correction)
6. [Minimum Sample Size Requirements](#minimum-sample-size-requirements)
7. [Drift Detection Methods](#drift-detection-methods)
8. [Implementation Examples](#implementation-examples)

---

## Why Statistical Validation Matters

### The Core Problem

**Question**: If you measure a demographic parity difference of 0.12, is that:
- A. Real bias that requires intervention?
- B. Random sampling variation?

**Without statistical validation, you cannot tell the difference.**

### Real-World Implications

**Scenario**: A hiring algorithm shows:
```
Male selection rate:   52%
Female selection rate: 48%
Difference: 4%
```

**Critical questions**:
1. Is 4% statistically significant?
2. How confident are we in this estimate?
3. Could this happen by chance alone?
4. What's the practical significance?

**Statistical validation answers these questions** through:
- Confidence intervals (quantifying uncertainty)
- Effect sizes (measuring practical significance)
- Hypothesis tests (determining statistical significance)

---

## Bootstrap Confidence Intervals

### What Are Bootstrap Confidence Intervals?

Bootstrap CI is a **resampling method** that estimates the sampling distribution of a statistic without making parametric assumptions.

### Why Bootstrap?

âœ… **Advantages**:
- No distributional assumptions (non-parametric)
- Works for complex metrics (e.g., demographic parity)
- Provides intuitive uncertainty quantification
- Handles small samples better than asymptotic methods

❌ **Limitations**:
- Computationally intensive
- Requires sufficient original sample size
- May underestimate variance with very small samples

### How Bootstrap Works

**Step-by-step process**:

```
1. Original sample: n observations → compute metric M

2. Repeat B times (typically B = 1000):
   a. Resample n observations WITH REPLACEMENT
   b. Compute metric M* on resampled data
   c. Store M*

3. Bootstrap distribution: {M*₁, M*₂, ..., M*_B}

4. Confidence interval: [Percentile α/2, Percentile 1-α/2]
   For 95% CI: [Percentile 2.5, Percentile 97.5]
```

### Mathematical Formulation

Given:
- Original data: D = {(x₁, y₁, s₁), ..., (xₙ, yₙ, sₙ)}
- Metric of interest: M(D)
- Desired confidence level: 1 - α (typically 0.95)

**Bootstrap procedure**:

1. Generate B bootstrap samples: D*ᵦ, b = 1, ..., B
   - Each D*ᵦ has n observations sampled with replacement from D

2. Compute metric on each sample: M*ᵦ = M(D*ᵦ)

3. Bootstrap distribution: F̂ = {M*₁, ..., M*_B}

4. **Percentile CI**: 
   ```
   CI = [F̂⁻¹(α/2), F̂⁻¹(1 - α/2)]
   ```
   Where F̂⁻¹ is the empirical quantile function

### Example: Demographic Parity CI

**Setup**:
```python
n = 500 samples
Group 0: n₀ = 250, positives = 150 (60%)
Group 1: n₁ = 250, positives = 112 (45%)

Observed DP_diff = |0.60 - 0.45| = 0.15
```

**Bootstrap process** (B = 1000):
```
Bootstrap sample 1:
  Resample 500 with replacement
  Group 0: 152/248 = 61.3%
  Group 1: 109/252 = 43.3%
  DP*₁ = 0.18

Bootstrap sample 2:
  Resample 500 with replacement
  Group 0: 147/253 = 58.1%
  Group 1: 115/247 = 46.6%
  DP*₂ = 0.12

... (998 more samples)

Bootstrap sample 1000:
  DP*₁₀₀₀ = 0.14
```

**Result**: 95% CI = [0.08, 0.22]

**Interpretation**:
- Point estimate: 0.15
- We're 95% confident the true disparity is between 8% and 22%
- The CI does NOT include 0 → statistically significant bias
- The CI does NOT include 0.10 (threshold) → exceeds acceptable limit even accounting for uncertainty

### Implementation

```python
import numpy as np
from typing import Tuple

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_func: callable,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a fairness metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Protected attribute
        metric_func: Function that computes the metric
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    np.random.seed(random_state)
    
    n = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        sensitive_boot = sensitive[indices]
        
        # Compute metric on bootstrap sample
        metric_value = metric_func(y_true_boot, y_pred_boot, sensitive_boot)
        bootstrap_metrics.append(metric_value)
    
    # Compute percentile CI
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    
    return lower, upper


# Usage example
def demographic_parity(y_true, y_pred, sensitive):
    """Compute demographic parity difference."""
    rate_0 = y_pred[sensitive == 0].mean()
    rate_1 = y_pred[sensitive == 1].mean()
    return abs(rate_0 - rate_1)

ci_lower, ci_upper = bootstrap_ci(
    y_true, y_pred, sensitive,
    metric_func=demographic_parity,
    n_bootstrap=1000,
    confidence_level=0.95
)

print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### Choosing Bootstrap Parameters

**Number of bootstrap samples (B)**:
- Minimum: 1000 (standard)
- Recommended: 10,000 (more stable)
- Trade-off: Accuracy vs. computation time

**Confidence level**:
- Standard: 95% (α = 0.05)
- Conservative: 99% (α = 0.01)
- Exploratory: 90% (α = 0.10)

---

## Effect Size Calculations

### What Is Effect Size?

**Definition**: Effect size measures the **magnitude** of a difference, independent of sample size.

**Why it matters**: Statistical significance ≠ practical significance
- With large samples, tiny differences become "significant"
- Effect size tells you if the difference actually matters

### Cohen's d for Fairness Metrics

**Cohen's d** measures standardized difference between two groups.

**Formula**:
```
d = (M₁ - M₀) / SD_pooled

where:
  M₁ = mean for group 1
  M₀ = mean for group 0
  
  SD_pooled = √[(SD₀² + SD₁²) / 2]
```

**Interpretation** (Cohen's guidelines):
- |d| < 0.2: Negligible effect
- 0.2 ≤ |d| < 0.5: Small effect
- 0.5 ≤ |d| < 0.8: Medium effect
- |d| ≥ 0.8: Large effect

### Risk Ratio (Relative Risk)

**Definition**: Ratio of probabilities between groups.

**Formula**:
```
RR = P(Ŷ = 1 | A = 1) / P(Ŷ = 1 | A = 0)
```

**Interpretation**:
- RR = 1.0: No difference (perfect parity)
- RR < 1.0: Group 1 has lower rate
- RR > 1.0: Group 1 has higher rate

**Example**:
```
Male approval rate:   60%
Female approval rate: 45%

RR = 0.45 / 0.60 = 0.75

Interpretation: Women are approved at 75% the rate of men
              (or 25% lower approval rate)
```

**Legal context**: Four-fifths rule requires RR ≥ 0.80

### Odds Ratio

**Formula**:
```
OR = [P(Ŷ=1|A=1) / P(Ŷ=0|A=1)] / [P(Ŷ=1|A=0) / P(Ŷ=0|A=0)]
```

**Interpretation**:
- OR = 1.0: No association
- OR < 1.0: Negative association
- OR > 1.0: Positive association

**When to use**: Logistic regression, case-control studies

### Implementation

```python
def cohens_d(group_0: np.ndarray, group_1: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.
    
    Args:
        group_0: Outcomes for group 0
        group_1: Outcomes for group 1
    
    Returns:
        Cohen's d value
    """
    mean_0 = np.mean(group_0)
    mean_1 = np.mean(group_1)
    
    std_0 = np.std(group_0, ddof=1)
    std_1 = np.std(group_1, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt((std_0**2 + std_1**2) / 2)
    
    if pooled_std == 0:
        return 0.0
    
    d = (mean_1 - mean_0) / pooled_std
    return d


def risk_ratio(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """
    Compute risk ratio between groups.
    
    Args:
        y_pred: Binary predictions
        sensitive: Binary protected attribute
    
    Returns:
        Risk ratio (RR)
    """
    rate_0 = y_pred[sensitive == 0].mean()
    rate_1 = y_pred[sensitive == 1].mean()
    
    if rate_0 == 0:
        return np.inf
    
    return rate_1 / rate_0


# Usage
y_pred_0 = y_pred[sensitive == 0]
y_pred_1 = y_pred[sensitive == 1]

d = cohens_d(y_pred_0, y_pred_1)
rr = risk_ratio(y_pred, sensitive)

print(f"Cohen's d: {d:.3f}")
print(f"Risk Ratio: {rr:.3f}")

if abs(d) >= 0.8:
    print("Large effect size - practical significance")
elif abs(d) >= 0.5:
    print("Medium effect size")
elif abs(d) >= 0.2:
    print("Small effect size")
else:
    print("Negligible effect size")
```

---

## Hypothesis Testing

### The Null Hypothesis Framework

**Null hypothesis (H₀)**: No disparity exists between groups
- DP_diff = 0
- Groups have equal rates

**Alternative hypothesis (H₁)**: Disparity exists
- DP_diff ≠ 0
- Groups have different rates

**Goal**: Determine if observed disparity is statistically significant or could occur by chance.

### Permutation Test for Fairness Metrics

**Concept**: If H₀ is true (no group effect), randomly permuting group labels shouldn't change the metric much.

**Procedure**:
```
1. Compute observed metric: M_obs

2. Repeat K times (typically K = 1000):
   a. Randomly shuffle group labels
   b. Compute metric on permuted data: M_perm
   c. Store M_perm

3. P-value = (# times M_perm ≥ M_obs) / K

4. If p-value < α (typically 0.05), reject H₀
```

**Implementation**:

```python
def permutation_test(
    y_pred: np.ndarray,
    sensitive: np.ndarray,
    metric_func: callable,
    n_permutations: int = 1000,
    random_state: int = 42
) -> float:
    """
    Perform permutation test for fairness metric.
    
    Returns:
        p-value
    """
    np.random.seed(random_state)
    
    # Observed metric
    observed = metric_func(y_pred, sensitive)
    
    # Permutation distribution
    null_distribution = []
    for _ in range(n_permutations):
        # Shuffle sensitive attribute
        sensitive_perm = np.random.permutation(sensitive)
        metric_perm = metric_func(y_pred, sensitive_perm)
        null_distribution.append(metric_perm)
    
    # P-value (two-tailed)
    p_value = np.mean(np.abs(null_distribution) >= abs(observed))
    
    return p_value


# Usage
p_val = permutation_test(y_pred, sensitive, demographic_parity)

if p_val < 0.05:
    print(f"Statistically significant bias (p={p_val:.4f})")
else:
    print(f"Not statistically significant (p={p_val:.4f})")
```

### Mann-Whitney U Test

**Use case**: Compare distributions between two groups (non-parametric alternative to t-test).

**When to use**:
- Comparing continuous fairness metrics
- Non-normal distributions
- Small samples

```python
from scipy.stats import mannwhitneyu

# Compare score distributions
scores_0 = model_scores[sensitive == 0]
scores_1 = model_scores[sensitive == 1]

statistic, p_value = mannwhitneyu(scores_0, scores_1, alternative='two-sided')

if p_value < 0.05:
    print("Significant difference in score distributions")
```

---

## Multiple Comparison Correction

### The Multiple Testing Problem

**Problem**: Testing multiple hypotheses increases false positive rate.

**Example**: Testing 10 fairness metrics at α = 0.05
- Expected false positives: 10 × 0.05 = 0.5
- Probability of ≥1 false positive: 1 - (0.95)¹⁰ = 0.40 (40%!)

### Bonferroni Correction

**Approach**: Divide significance level by number of tests.

**Formula**:
```
α_adjusted = α / m

where m = number of tests
```

**Example**:
```
Testing 5 metrics at family-wise α = 0.05
α_adjusted = 0.05 / 5 = 0.01

Each test uses α = 0.01 instead of 0.05
```

**Implementation**:

```python
def bonferroni_correction(p_values: list, alpha: float = 0.05) -> list:
    """
    Apply Bonferroni correction to p-values.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
    
    Returns:
        List of adjusted significance decisions
    """
    m = len(p_values)
    alpha_adjusted = alpha / m
    
    significant = [p < alpha_adjusted for p in p_values]
    return significant


# Usage
p_values = [0.02, 0.04, 0.001, 0.15, 0.03]
significant = bonferroni_correction(p_values, alpha=0.05)

for metric, p, sig in zip(metric_names, p_values, significant):
    print(f"{metric}: p={p:.3f}, significant={sig}")
```

**Pros**: Simple, controls family-wise error rate
**Cons**: Conservative (reduces power)

### Benjamini-Hochberg (FDR Control)

**Approach**: Control false discovery rate instead of family-wise error.

**Less conservative** than Bonferroni, more power to detect true effects.

**Procedure**:
```
1. Sort p-values: p₍₁₎ ≤ p₍₂₎ ≤ ... ≤ p₍ₘ₎

2. Find largest k where:
   p₍ₖ₎ ≤ (k/m) × α

3. Reject H₀ for all p₍ᵢ₎ where i ≤ k
```

**Implementation**:

```python
def benjamini_hochberg(p_values: list, alpha: float = 0.05) -> list:
    """
    Apply Benjamini-Hochberg FDR correction.
    
    Args:
        p_values: List of p-values
        alpha: False discovery rate
    
    Returns:
        List of adjusted significance decisions
    """
    m = len(p_values)
    
    # Sort p-values with original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    
    # Find critical value
    thresholds = (np.arange(1, m + 1) / m) * alpha
    significant_sorted = sorted_p <= thresholds
    
    # If any significant, reject all up to largest significant
    if significant_sorted.any():
        max_sig_idx = np.where(significant_sorted)[0][-1]
        significant_sorted[:max_sig_idx + 1] = True
    
    # Unsort results
    significant = np.zeros(m, dtype=bool)
    significant[sorted_indices] = significant_sorted
    
    return significant.tolist()
```

---

## Minimum Sample Size Requirements

### Why Sample Size Matters

**Statistical power**: Ability to detect true effects
**Precision**: Width of confidence intervals
**Reliability**: Stability of estimates

### General Guidelines

**Minimum requirements**:
- **Per group**: n ≥ 30 (Central Limit Theorem applies)
- **Recommended**: n ≥ 100 per group (stable estimates)
- **Intersectional analysis**: n ≥ 50 per subgroup

**Toolkit enforcement**:
```python
MIN_GROUP_SIZE = 30  # From shared/constants.py

# Validation
if (sensitive == 0).sum() < MIN_GROUP_SIZE:
    raise ValueError(f"Group 0 too small: {(sensitive == 0).sum()} < {MIN_GROUP_SIZE}")
```

### Power Analysis

**Question**: How many samples do I need to detect a disparity of size δ?

**Formula** (simplified for proportions):
```
n ≈ 2 × (z_α/2 + z_β)² × p(1-p) / δ²

where:
  z_α/2 = critical value for significance level (1.96 for α=0.05)
  z_β = critical value for power (0.84 for 80% power)
  p = expected proportion
  δ = minimum detectable difference
```

**Example**: Detect 10% difference at 80% power, α=0.05
```
n ≈ 2 × (1.96 + 0.84)² × 0.5(1-0.5) / 0.10²
n ≈ 2 × 7.84 × 0.25 / 0.01
n ≈ 392 per group
```

---

## Drift Detection Methods

### Kolmogorov-Smirnov (KS) Test

**Purpose**: Detect if two distributions differ.

**Use case**: Compare current vs. reference period distributions.

**Test statistic**:
```
D = max|F₁(x) - F₂(x)|

where F₁, F₂ are empirical CDFs
```

**Implementation**:

```python
from scipy.stats import ks_2samp

def detect_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    alpha: float = 0.05
) -> dict:
    """
    Detect distribution drift using KS test.
    
    Returns:
        Dictionary with drift detection results
    """
    statistic, p_value = ks_2samp(reference_scores, current_scores)
    
    drift_detected = p_value < alpha
    
    return {
        'drift_detected': drift_detected,
        'ks_statistic': statistic,
        'p_value': p_value,
        'severity': 'high' if statistic > 0.2 else 'medium' if statistic > 0.1 else 'low'
    }


# Usage in monitoring
reference_data = historical_predictions[:1000]
current_data = recent_predictions

drift_result = detect_drift(reference_data, current_data)

if drift_result['drift_detected']:
    print(f"âš ï¸ Drift detected! KS={drift_result['ks_statistic']:.3f}")
```

### Chi-Square Test for Categorical Distributions

**Purpose**: Test if group distributions have changed.

```python
from scipy.stats import chi2_contingency

def test_representation_drift(
    reference_groups: np.ndarray,
    current_groups: np.ndarray
) -> dict:
    """
    Test if group proportions have changed.
    """
    # Create contingency table
    ref_counts = pd.Series(reference_groups).value_counts()
    cur_counts = pd.Series(current_groups).value_counts()
    
    contingency = pd.DataFrame({
        'reference': ref_counts,
        'current': cur_counts
    }).fillna(0)
    
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    
    return {
        'drift_detected': p_value < 0.05,
        'chi2_statistic': chi2,
        'p_value': p_value
    }
```

---

## Implementation Examples

### Complete Fairness Validation Pipeline

```python
class StatisticalFairnessValidator:
    """
    Complete statistical validation for fairness metrics.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
        alpha: float = 0.05,
        min_group_size: int = 30
    ):
        self.confidence_level = confidence_level
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.min_group_size = min_group_size
    
    def validate_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive: np.ndarray,
        metric_name: str
    ) -> dict:
        """
        Complete statistical validation of a fairness metric.
        
        Returns:
            Dictionary with:
            - value: Point estimate
            - ci_lower, ci_upper: Confidence interval
            - p_value: Significance test
            - effect_size: Standardized effect
            - interpretation: Human-readable summary
        """
        # Check sample sizes
        n_0 = (sensitive == 0).sum()
        n_1 = (sensitive == 1).sum()
        
        if n_0 < self.min_group_size or n_1 < self.min_group_size:
            raise ValueError(
                f"Insufficient group sizes: {n_0}, {n_1} "
                f"(minimum: {self.min_group_size})"
            )
        
        # Compute point estimate
        metric_func = self._get_metric_func(metric_name)
        value = metric_func(y_true, y_pred, sensitive)
        
        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_ci(
            y_true, y_pred, sensitive,
            metric_func=lambda yt, yp, s: metric_func(yt, yp, s),
            n_bootstrap=self.n_bootstrap,
            confidence_level=self.confidence_level
        )
        
        # Permutation test
        p_value = permutation_test(
            y_pred, sensitive, 
            metric_func=lambda yp, s: metric_func(y_true, yp, s)
        )
        
        # Effect size
        y_pred_0 = y_pred[sensitive == 0]
        y_pred_1 = y_pred[sensitive == 1]
        effect_size = cohens_d(y_pred_0, y_pred_1)
        
        # Interpretation
        is_significant = p_value < self.alpha
        is_large_effect = abs(effect_size) >= 0.5
        
        if is_significant and is_large_effect:
            interpretation = "Statistically and practically significant bias"
        elif is_significant:
            interpretation = "Statistically significant but small effect"
        elif is_large_effect:
            interpretation = "Large effect but not statistically significant"
        else:
            interpretation = "No significant bias detected"
        
        return {
            'value': value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'effect_size': effect_size,
            'is_significant': is_significant,
            'interpretation': interpretation,
            'sample_sizes': {'group_0': n_0, 'group_1': n_1}
        }
    
    def _get_metric_func(self, metric_name: str):
        """Get metric computation function."""
        if metric_name == 'demographic_parity':
            return self._demographic_parity
        elif metric_name == 'equalized_odds':
            return self._equalized_odds
        else:
            raise ValueError(f"Unknown metric: {metric_name}")
    
    @staticmethod
    def _demographic_parity(y_true, y_pred, sensitive):
        rate_0 = y_pred[sensitive == 0].mean()
        rate_1 = y_pred[sensitive == 1].mean()
        return abs(rate_0 - rate_1)
    
    @staticmethod
    def _equalized_odds(y_true, y_pred, sensitive):
        # TPR difference
        tpr_0 = y_pred[(y_true == 1) & (sensitive == 0)].mean()
        tpr_1 = y_pred[(y_true == 1) & (sensitive == 1)].mean()
        tpr_diff = abs(tpr_0 - tpr_1)
        
        # FPR difference
        fpr_0 = y_pred[(y_true == 0) & (sensitive == 0)].mean()
        fpr_1 = y_pred[(y_true == 0) & (sensitive == 1)].mean()
        fpr_diff = abs(fpr_0 - fpr_1)
        
        return max(tpr_diff, fpr_diff)


# Usage
validator = StatisticalFairnessValidator()

result = validator.validate_metric(
    y_true=y_test,
    y_pred=y_pred,
    sensitive=gender,
    metric_name='demographic_parity'
)

print(f"Metric: {result['value']:.3f}")
print(f"95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"P-value: {result['p_value']:.4f}")
print(f"Effect size: {result['effect_size']:.3f}")
print(f"Interpretation: {result['interpretation']}")
```

---

## Best Practices

### 1. Always Report Uncertainty

❌ **Bad**: "Demographic parity = 0.15"
âœ… **Good**: "Demographic parity = 0.15, 95% CI [0.10, 0.20]"

### 2. Use Effect Sizes

❌ **Bad**: "Statistically significant (p < 0.05)"
âœ… **Good**: "Statistically significant (p < 0.05) with medium effect size (d = 0.6)"

### 3. Check Sample Sizes

```python
# Always validate before computing metrics
if (sensitive == 0).sum() < 30:
    print("⚠️ Warning: Group 0 too small for reliable estimates")
```

### 4. Correct for Multiple Testing

```python
# When testing multiple metrics
metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
p_values = [test_metric(m) for m in metrics]

# Apply correction
significant = benjamini_hochberg(p_values, alpha=0.05)
```

### 5. Document Assumptions

Always document:
- Sample size requirements
- Independence assumptions
- Distribution assumptions (or lack thereof)
- Multiple testing corrections applied

---

## References

### Statistical Methods

1. **Efron, B., & Tibshirani, R. J. (1994)**. *An Introduction to the Bootstrap*. Chapman & Hall.

2. **Cohen, J. (1988)**. *Statistical Power Analysis for the Behavioral Sciences*. Routledge.

3. **Benjamini, Y., & Hochberg, Y. (1995)**. "Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing." *JRSS-B*.

### Implementation Resources

- **SciPy**: https://scipy.org/ (scipy.stats module)
- **statsmodels**: https://www.statsmodels.org/
- **scikit-learn**: https://scikit-learn.org/ (metrics, resampling)

---
 