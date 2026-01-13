# Best Practices Guide

## Overview

This guide provides recommendations for effectively using the Fairness Pipeline Toolkit in research, development, and production environments. Follow these practices to ensure robust, reproducible, and ethically sound fairness analysis.

---

## Table of Contents

1. [General Principles](#general-principles)
2. [Data Preparation](#data-preparation)
3. [Measurement Best Practices](#measurement-best-practices)
4. [Bias Detection & Mitigation](#bias-detection--mitigation)
5. [Model Training](#model-training)
6. [Production Monitoring](#production-monitoring)
7. [Experiment Tracking](#experiment-tracking)
8. [Documentation](#documentation)
9. [Common Pitfalls](#common-pitfalls)
10. [Performance Optimization](#performance-optimization)

---

## General Principles

### 1. Fairness is Context-Dependent

**Do:**
- Consult domain experts and stakeholders when defining fairness criteria
- Document why specific fairness metrics were chosen
- Consider multiple fairness definitions for comprehensive analysis
- Recognize that fairness requirements may differ across use cases

**Don't:**
- Apply fairness metrics mechanically without understanding implications
- Assume one fairness definition fits all scenarios
- Ignore stakeholder input in metric selection

**Example:**
```yaml
# Document fairness rationale in config
fairness_metrics:
  - demographic_parity  # Required by regulation
  - equalized_odds      # Stakeholder preference for balanced errors
  
# Add comment explaining choice
# demographic_parity: Ensures equal opportunity across groups
# equalized_odds: Balances false positive and false negative rates
```

### 2. Transparency Over Optimization

**Do:**
- Report full results, including metrics that don't meet thresholds
- Document all preprocessing steps and their rationale
- Track failed experiments alongside successful ones
- Maintain audit trails of model versions and configurations

**Don't:**
- Cherry-pick metrics that show favorable results
- Hide preprocessing steps that improve fairness metrics
- Delete failed experiments from version control

### 3. Iteration is Essential

**Do:**
- Start with baseline measurements before applying mitigations
- Compare multiple mitigation strategies
- Validate improvements on held-out test sets
- Re-evaluate fairness metrics periodically in production

**Don't:**
- Apply mitigation without measuring baseline first
- Commit to first solution without exploring alternatives
- Deploy without validation on representative test data

---

## Data Preparation

### Sample Size Requirements

**Minimum Requirements:**
- **Per group:** ≥30 samples for statistical validity
- **Per group × class:** ≥10 samples for reliable metrics
- **Total dataset:** ≥200 samples for bootstrap confidence intervals

**Example validation:**
```python
def validate_group_sizes(df, protected_attr, target_col, min_size=30):
    """Ensure sufficient samples per group."""
    group_counts = df.groupby([protected_attr, target_col]).size()
    
    for (group, label), count in group_counts.items():
        if count < min_size:
            logger.warning(
                f"Group {group}, Label {label}: Only {count} samples "
                f"(minimum {min_size} recommended)"
            )
    
    return group_counts
```

### Train/Test Splitting

**Best Practice:**
```python
from sklearn.model_selection import train_test_split

# Stratify by both target AND protected attribute
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive_features,
    test_size=0.3,
    random_state=42,
    stratify=pd.DataFrame({'y': y, 's': sensitive_features}).apply(tuple, axis=1)
)
```

**Why:** Ensures both groups are represented in train and test sets proportionally.

### Handling Missing Data

**Recommended approach:**
```python
# 1. Analyze missing patterns by group
missing_by_group = df.groupby('protected_attr').apply(
    lambda x: x.isnull().sum() / len(x)
)

# 2. Document disparities
if missing_by_group.std().max() > 0.1:
    logger.warning("Missing data patterns differ across groups")

# 3. Use group-aware imputation
from sklearn.preprocessing import SimpleImputer

imputers = {}
for group in df['protected_attr'].unique():
    mask = df['protected_attr'] == group
    imputers[group] = SimpleImputer(strategy='median')
    imputers[group].fit(df.loc[mask, feature_cols])
```

---

## Measurement Best Practices

### Choosing Fairness Metrics

**Decision Tree:**

1. **Outcome fairness (equal results)?**
   - Yes → Demographic Parity
   - No → Continue

2. **Error rate fairness (equal mistakes)?**
   - Yes → Equalized Odds
   - No → Continue

3. **Focus on true positives only?**
   - Yes → Equal Opportunity
   - No → Consider multiple metrics

**Metric Comparison Table:**

| Use Case | Primary Metric | Rationale |
|----------|---------------|-----------|
| Loan approval | Demographic Parity | Equal opportunity to receive loans |
| Criminal justice | Equalized Odds | Balance false positives and false negatives |
| Medical screening | Equal Opportunity | Ensure disease detection across groups |
| Hiring | Demographic Parity + Equalized Odds | Both opportunity and accuracy matter |

### Statistical Rigor

**Bootstrap Configuration:**
```yaml
# For research/publication
bootstrap_samples: 5000
confidence_level: 0.99  # 99% CI

# For development
bootstrap_samples: 1000
confidence_level: 0.95  # 95% CI (default)

# For quick iteration
bootstrap_samples: 500
confidence_level: 0.95
```

**Interpreting Results:**
```python
result = analyzer.compute_metric(y_true, y_pred, sensitive_features)

# 1. Check confidence interval
ci_width = result.confidence_interval[1] - result.confidence_interval[0]
if ci_width > 0.1:
    logger.warning(f"Wide confidence interval: {ci_width:.3f}")

# 2. Check effect size
if abs(result.effect_size) < 0.2:
    print("Small effect size - may not be practically significant")
elif abs(result.effect_size) < 0.5:
    print("Medium effect size")
else:
    print("Large effect size")

# 3. Don't rely solely on is_fair flag
print(f"Metric value: {result.value:.3f}")
print(f"Threshold: {threshold}")
print(f"Fair: {result.is_fair}")
```

---

## Bias Detection & Mitigation

### Detection Thresholds

**Conservative approach (research/high-stakes):**
```yaml
bias_detection:
  thresholds:
    representation: 0.1      # 10% deviation
    proxy_correlation: 0.3   # Moderate correlation
    statistical_alpha: 0.01  # 99% confidence
```

**Standard approach (development):**
```yaml
bias_detection:
  thresholds:
    representation: 0.2      # 20% deviation (default)
    proxy_correlation: 0.5   # Strong correlation
    statistical_alpha: 0.05  # 95% confidence (default)
```

### Mitigation Strategy Selection

**Decision Matrix:**

| Data Characteristic | Recommended Method | Rationale |
|---------------------|-------------------|-----------|
| Severe imbalance (1:10 ratio) | Reweighting | Preserves all samples |
| Moderate imbalance | Reweighting or Resampling | Either works well |
| Small dataset (<500 samples) | Reweighting | Avoids reducing dataset size |
| Large dataset (>10k samples) | Resampling | Computational efficiency |

**Progressive approach:**
```yaml
# Experiment 1: No mitigation (baseline)
bias_mitigation:
  method: 'none'

# Experiment 2: Partial reweighting
bias_mitigation:
  method: 'reweighting'
  params:
    alpha: 0.5

# Experiment 3: Full reweighting
bias_mitigation:
  method: 'reweighting'
  params:
    alpha: 1.0

# Experiment 4: Resampling (if reweighting insufficient)
bias_mitigation:
  method: 'resampling'
  params:
    strategy: 'oversample'
```

### Validating Mitigation Effectiveness

**Before/After Comparison:**
```python
# 1. Measure baseline
baseline_result = analyzer.compute_metric(y_test, baseline_pred, sensitive_test)

# 2. Apply mitigation and retrain
X_mit, y_mit, weights = reweighter.fit_transform(X_train, y_train, s_train)
model_mit = LogisticRegression().fit(X_mit, y_mit, sample_weight=weights)
mit_pred = model_mit.predict(X_test)

# 3. Measure after mitigation
mit_result = analyzer.compute_metric(y_test, mit_pred, sensitive_test)

# 4. Compare
improvement = baseline_result.value - mit_result.value
print(f"Bias reduction: {improvement:.3f}")

# 5. Check if statistically significant
ci_overlap = (
    baseline_result.confidence_interval[0] <= mit_result.confidence_interval[1] and
    mit_result.confidence_interval[0] <= baseline_result.confidence_interval[1]
)
if ci_overlap:
    logger.warning("Confidence intervals overlap - improvement may not be significant")
```

---

## Model Training

### Constraint Selection

**Fairness-Accuracy Tradeoff:**

```python
# Grid search over constraint strengths
eps_values = [0.01, 0.02, 0.05, 0.1, 0.2]
results = []

for eps in eps_values:
    model = ReductionsWrapper(
        base_estimator=LogisticRegression(),
        constraint='demographic_parity',
        eps=eps
    )
    model.fit(X_train, y_train, sensitive_features=s_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    fairness = analyzer.compute_metric(y_test, y_pred, s_test)
    
    results.append({
        'eps': eps,
        'accuracy': accuracy,
        'fairness': fairness.value
    })

# Visualize Pareto frontier
import matplotlib.pyplot as plt
plt.scatter([r['fairness'] for r in results], [r['accuracy'] for r in results])
plt.xlabel('Fairness Violation')
plt.ylabel('Accuracy')
plt.title('Fairness-Accuracy Tradeoff')
```

### Hyperparameter Tuning

**Fair cross-validation:**
```python
from sklearn.model_selection import StratifiedKFold

# Stratify by protected attribute to ensure fairness in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(X, sensitive_features):
    X_train_cv, X_val_cv = X[train_idx], X[val_idx]
    y_train_cv, y_val_cv = y[train_idx], y[val_idx]
    s_train_cv, s_val_cv = sensitive_features[train_idx], sensitive_features[val_idx]
    
    # Train and evaluate fairness on each fold
    model.fit(X_train_cv, y_train_cv, sensitive_features=s_train_cv)
    fairness_scores.append(
        analyzer.compute_metric(y_val_cv, model.predict(X_val_cv), s_val_cv).value
    )

# Report mean and std of fairness across folds
print(f"Fairness: {np.mean(fairness_scores):.3f} ± {np.std(fairness_scores):.3f}")
```

---

## Production Monitoring

### Alert Configuration

**Tiered alerting:**
```yaml
monitoring:
  thresholds:
    # Warning level (investigate)
    demographic_parity_warn: 0.08
    equalized_odds_warn: 0.08
    
    # Critical level (immediate action)
    demographic_parity_critical: 0.15
    equalized_odds_critical: 0.15
```

**Implementation:**
```python
tracker = RealTimeFairnessTracker(window_size=1000)

for batch in production_stream:
    metrics = tracker.add_batch(
        batch['predictions'],
        batch['labels'],
        batch['sensitive']
    )
    
    # Check warning thresholds
    for metric_name, value in metrics.items():
        if value > thresholds[f'{metric_name}_critical']:
            send_alert(f"CRITICAL: {metric_name} = {value:.3f}", priority='high')
        elif value > thresholds[f'{metric_name}_warn']:
            send_alert(f"WARNING: {metric_name} = {value:.3f}", priority='medium')
```

### Drift Detection

**Adaptive window sizing:**
```python
# Use smaller windows for rapid detection, larger for stability
if deployment_phase == 'initial':
    window_size = 500      # Quick feedback
elif deployment_phase == 'stable':
    window_size = 2000     # More stable
else:
    window_size = 1000     # Default
```

### Reporting Cadence

**Recommended schedule:**
- **Real-time:** Alert on threshold violations
- **Daily:** Automated dashboard updates
- **Weekly:** Detailed fairness reports to stakeholders
- **Quarterly:** Comprehensive fairness audits

---

## Experiment Tracking

### MLflow Best Practices

**Comprehensive logging:**
```python
import mlflow

with mlflow.start_run(run_name='fair_model_v1'):
    # 1. Log configuration
    mlflow.log_params(config['training'])
    mlflow.log_params(config['bias_mitigation'])
    
    # 2. Log data characteristics
    mlflow.log_metric('train_size', len(X_train))
    mlflow.log_metric('test_size', len(X_test))
    mlflow.log_metric('imbalance_ratio', imbalance_ratio)
    
    # 3. Log baseline metrics
    for metric_name, result in baseline_results.items():
        mlflow.log_metric(f'baseline_{metric_name}', result.value)
        mlflow.log_metric(f'baseline_{metric_name}_ci_width', 
                          result.confidence_interval[1] - result.confidence_interval[0])
    
    # 4. Log final metrics
    mlflow.log_metric('final_accuracy', accuracy)
    for metric_name, result in final_results.items():
        mlflow.log_metric(f'final_{metric_name}', result.value)
    
    # 5. Log artifacts
    mlflow.sklearn.log_model(model, 'model')
    mlflow.log_artifact('config.yml')
    mlflow.log_artifact('reports/fairness_report.md')
```

### Version Control

**Git workflow:**
```bash
# 1. Create experiment branch
git checkout -b experiment/reweighting_alpha_0.8

# 2. Update config
vim config.yml

# 3. Run experiment
python run_pipeline.py --config config.yml

# 4. Commit results
git add config.yml reports/ mlruns/
git commit -m "Experiment: Reweighting alpha=0.8, DP=0.045"

# 5. Merge if successful
git checkout main
git merge experiment/reweighting_alpha_0.8
```

---

## Documentation

### Configuration Documentation

**Always include:**
```yaml
# config.yml

# EXPERIMENT: Loan Approval Fairness
# DATE: 2024-01-15
# OBJECTIVE: Reduce demographic parity violation below 0.05
# PREVIOUS BEST: 0.08 (baseline model)

data:
  path: 'data/loan_data_v2.csv'
  # NOTE: Updated dataset with corrected labels from 2024-01-10

bias_detection:
  # Using 0.1 threshold per regulatory guidance document XYZ
  thresholds:
    representation: 0.1

bias_mitigation:
  method: 'reweighting'
  params:
    alpha: 0.8  # Reduced from 1.0 due to accuracy drop
```

### Result Interpretation

**Template for summarizing results:**
```markdown
## Experiment Summary

**Date:** 2024-01-15
**Objective:** Reduce demographic parity violation to <0.05

### Configuration
- Mitigation: Reweighting (alpha=0.8)
- Constraint: Demographic Parity (eps=0.05)

### Results

| Metric | Baseline | After Mitigation | Improvement |
|--------|----------|------------------|-------------|
| Accuracy | 0.82 | 0.80 | -0.02 |
| Demographic Parity | 0.12 | 0.045 | -0.075 ✓ |
| Equalized Odds | 0.08 | 0.06 | -0.02 |

### Key Findings
- ✓ Successfully met fairness objective (DP < 0.05)
- ⚠ Small accuracy decrease (2%) - acceptable per stakeholders
- ✓ No significant change in equalized odds

### Decision
**APPROVED FOR DEPLOYMENT** - Fairness improvement justifies minor accuracy tradeoff.

### Next Steps
- Deploy to production with monitoring alerts
- Schedule 2-week review of production metrics
```

---

## Common Pitfalls

### 1. Ignoring Intersectionality

**Problem:** Analyzing only single protected attributes.

**Solution:**
```python
# Analyze combinations of protected attributes
df['intersectional_group'] = (
    df['gender'].astype(str) + '_' + df['race'].astype(str)
)

# Measure fairness for intersectional groups
result = analyzer.compute_metric(
    y_test, predictions, 
    sensitive_features=df['intersectional_group']
)
```

### 2. Overfitting to Fairness Metrics

**Problem:** Repeatedly tuning until metrics pass thresholds on test set.

**Solution:**
- Use separate validation set for tuning
- Reserve test set for final evaluation only
- Report all attempted configurations, not just successful ones

### 3. Ignoring Base Rates

**Problem:** Comparing groups with very different outcome rates.

**Solution:**
```python
# Check base rates before fairness analysis
base_rates = df.groupby('protected_attr')['target'].mean()
print(f"Base rates: {base_rates}")

if base_rates.std() > 0.1:
    logger.warning(
        "Large base rate differences - consider whether equal "
        "outcome rates are appropriate fairness criterion"
    )
```

### 4. Insufficient Sample Sizes

**Problem:** Computing fairness metrics on small groups.

**Solution:**
```python
# Enforce minimum group sizes
min_group_size = 30
group_sizes = df.groupby('protected_attr').size()

for group, size in group_sizes.items():
    if size < min_group_size:
        raise ValueError(
            f"Group {group} has only {size} samples "
            f"(minimum {min_group_size} required)"
        )
```

---

## Performance Optimization

### Bootstrap Optimization

**For large datasets:**
```yaml
# Use fewer bootstrap samples for initial exploration
bootstrap_samples: 500  # Instead of 1000

# Parallelize bootstrap computation
n_jobs: -1  # Use all CPU cores
```

### Memory Management

**For datasets >1GB:**
```python
# Process in chunks
chunk_size = 10000

for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # Process chunk
    results.append(process_chunk(chunk))

# Aggregate results
final_result = aggregate_results(results)
```

### Caching

**Cache expensive computations:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_expensive_metric(y_true_hash, y_pred_hash, sensitive_hash):
    # Convert back from hash and compute
    return analyzer.compute_metric(y_true, y_pred, sensitive)
```

---

## Checklist

Before deploying a fairness-aware model:

- [ ] Documented why specific fairness metrics were chosen
- [ ] Measured baseline fairness on representative test data
- [ ] Validated all protected groups have ≥30 samples
- [ ] Computed confidence intervals for all fairness metrics
- [ ] Compared multiple mitigation strategies
- [ ] Validated fairness improvements are statistically significant
- [ ] Assessed fairness-accuracy tradeoff with stakeholders
- [ ] Configured production monitoring with appropriate thresholds
- [ ] Documented all configuration choices and rationale
- [ ] Set up alerting for fairness violations
- [ ] Scheduled periodic fairness audits
- [ ] Logged all experiments in MLflow
- [ ] Committed configuration to version control

---

## See Also

- [Configuration Guide](configuration.md) - Detailed parameter reference
- [API Reference](api_reference.md) - Programmatic usage
- Module-specific best practices in individual README files