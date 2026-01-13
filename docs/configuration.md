# Configuration Guide

## Overview

The Fairness Pipeline uses a centralized YAML configuration file (`config.yml`) to control all aspects of the pipeline execution. This guide explains each configuration section and available options.

## Configuration File Structure

```yaml
data:
  # Data loading configuration
  
bias_detection:
  # Bias detection parameters
  
bias_mitigation:
  # Mitigation strategy configuration
  
training:
  # Model training parameters
  
fairness_metrics:
  # Metrics to track
  
monitoring:
  # Production monitoring settings
  
mlflow:
  # Experiment tracking
```

---

## Data Configuration

Controls data loading and feature specification.

```yaml
data:
  path: 'data/sample_loan_data.csv'
  target_column: 'loan_approved'
  protected_attribute: 'gender'
  feature_columns: null  # null = auto-detect
```

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `path` | string | Yes | Path to CSV data file |
| `target_column` | string | Yes | Name of target/label column |
| `protected_attribute` | string | Yes | Sensitive attribute for fairness analysis |
| `feature_columns` | list/null | No | Feature columns to use. If null, auto-detects all columns except target and protected attribute |

### Examples

**Explicit feature selection:**
```yaml
data:
  path: 'data/my_data.csv'
  target_column: 'outcome'
  protected_attribute: 'race'
  feature_columns:
    - age
    - income
    - education_level
    - employment_status
```

**Auto-detection:**
```yaml
data:
  path: 'data/my_data.csv'
  target_column: 'outcome'
  protected_attribute: 'race'
  feature_columns: null  # Uses all columns except outcome and race
```

---

## Bias Detection Configuration

Controls how bias is detected in the dataset.

```yaml
bias_detection:
  protected_attribute: 'gender'
  
  reference_distribution:
    0: 0.5  # Expected proportion for group 0
    1: 0.5  # Expected proportion for group 1
  
  checks:
    - representation
    - proxy
    - statistical_disparity
  
  thresholds:
    representation: 0.2
    proxy_correlation: 0.5
    statistical_alpha: 0.05
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `protected_attribute` | string | Required | Sensitive attribute to check for bias |
| `reference_distribution` | dict | None | Expected demographic distribution (if known) |
| `checks` | list | All checks | Types of bias detection to perform |
| `thresholds.representation` | float | 0.2 | Max deviation from expected representation (20%) |
| `thresholds.proxy_correlation` | float | 0.5 | Correlation threshold for proxy feature detection |
| `thresholds.statistical_alpha` | float | 0.05 | Significance level for statistical tests |

### Bias Check Types

- **representation**: Detects underrepresentation or overrepresentation of protected groups
- **proxy**: Identifies features highly correlated with protected attribute
- **statistical_disparity**: Tests for statistical differences in label distributions across groups

### Examples

**Strict bias detection:**
```yaml
bias_detection:
  protected_attribute: 'gender'
  thresholds:
    representation: 0.1      # Only 10% deviation allowed
    proxy_correlation: 0.3   # Lower threshold for proxy detection
    statistical_alpha: 0.01  # More stringent significance test
```

**Known demographic distribution:**
```yaml
bias_detection:
  protected_attribute: 'ethnicity'
  reference_distribution:
    asian: 0.15
    black: 0.20
    hispanic: 0.25
    white: 0.40
```

---

## Bias Mitigation Configuration

Controls how bias is mitigated during preprocessing.

```yaml
bias_mitigation:
  method: 'reweighting'
  
  params:
    method: 'inverse_propensity'
    alpha: 1.0
```

### Parameters

| Parameter | Type | Options | Description |
|-----------|------|---------|-------------|
| `method` | string | `reweighting`, `resampling`, `none` | Mitigation strategy |
| `params.method` | string | `inverse_propensity`, `balanced` | Reweighting method |
| `params.alpha` | float | 0.0-1.0 | Reweighting strength (1.0 = full, 0.0 = none) |

### Mitigation Methods

#### 1. Reweighting
Assigns sample weights to balance group influence without changing dataset size.

```yaml
bias_mitigation:
  method: 'reweighting'
  params:
    method: 'inverse_propensity'  # or 'balanced'
    alpha: 1.0
```

**Methods:**
- `inverse_propensity`: Weight inversely proportional to group size
- `balanced`: Normalize weights to balance positive/negative rates

#### 2. Resampling
Changes dataset composition through oversampling or undersampling.

```yaml
bias_mitigation:
  method: 'resampling'
  params:
    strategy: 'oversample'  # or 'undersample'
    sampling_ratio: 1.0
```

#### 3. No Mitigation
Baseline comparison without preprocessing.

```yaml
bias_mitigation:
  method: 'none'
```

### Examples

**Partial reweighting:**
```yaml
bias_mitigation:
  method: 'reweighting'
  params:
    method: 'inverse_propensity'
    alpha: 0.5  # 50% reweighting, 50% original
```

**Aggressive resampling:**
```yaml
bias_mitigation:
  method: 'resampling'
  params:
    strategy: 'oversample'
    sampling_ratio: 2.0  # Double minority class
```

---

## Training Configuration

Controls model training with or without fairness constraints.

```yaml
training:
  use_fairness_constraints: false
  constraint_type: 'demographic_parity'
  eps: 0.05
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_fairness_constraints` | boolean | false | Enable fairness constraints during training |
| `constraint_type` | string | `demographic_parity` | Type of fairness constraint |
| `eps` | float | 0.05 | Constraint violation tolerance |

### Constraint Types

- **demographic_parity**: Equal positive prediction rates across groups
- **equalized_odds**: Equal TPR and FPR across groups
- **equal_opportunity**: Equal TPR across groups (relaxed equalized odds)
- **bounded_group_loss**: Maximum loss per group constraint

### Examples

**Training with constraints:**
```yaml
training:
  use_fairness_constraints: true
  constraint_type: 'equalized_odds'
  eps: 0.01  # Strict constraint (1% tolerance)
```

**Standard training with sample weights:**
```yaml
training:
  use_fairness_constraints: false
  # Uses sample weights from mitigation step
```

**Multiple constraint comparison:**
Run multiple experiments with different constraints:
```yaml
# Experiment 1: Demographic Parity
training:
  use_fairness_constraints: true
  constraint_type: 'demographic_parity'
  eps: 0.05

# Experiment 2: Equalized Odds
training:
  use_fairness_constraints: true
  constraint_type: 'equalized_odds'
  eps: 0.05
```

---

## Fairness Metrics Configuration

Specifies which fairness metrics to compute and track.

```yaml
fairness_metrics:
  - demographic_parity
  - equalized_odds

fairness_threshold: 0.1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fairness_metrics` | list | All metrics | Metrics to compute |
| `fairness_threshold` | float | 0.1 | Maximum acceptable fairness violation (10%) |

### Available Metrics

| Metric | Definition | When to Use |
|--------|------------|-------------|
| `demographic_parity` | P(ŷ=1\|A=0) = P(ŷ=1\|A=1) | Equal opportunity regardless of group |
| `equalized_odds` | TPR and FPR equal across groups | Balanced error rates |
| `equal_opportunity` | TPR equal across groups | Focus on true positive rate |

### Examples

**Single metric focus:**
```yaml
fairness_metrics:
  - demographic_parity
fairness_threshold: 0.05  # Strict 5% threshold
```

**Comprehensive tracking:**
```yaml
fairness_metrics:
  - demographic_parity
  - equalized_odds
  - equal_opportunity
fairness_threshold: 0.1
```

---

## Monitoring Configuration

Controls real-time fairness monitoring in production.

```yaml
monitoring:
  enabled: true
  window_size: 1000
  drift_alpha: 0.05
  
  thresholds:
    demographic_parity: 0.1
    equalized_odds: 0.1
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable monitoring tracker |
| `window_size` | integer | 1000 | Sliding window size for drift detection |
| `drift_alpha` | float | 0.05 | Significance level for KS drift test |
| `thresholds` | dict | 0.1 | Alert thresholds per metric |

### Examples

**Sensitive monitoring:**
```yaml
monitoring:
  enabled: true
  window_size: 500      # Smaller window = faster detection
  drift_alpha: 0.01     # More sensitive drift detection
  
  thresholds:
    demographic_parity: 0.05  # Stricter thresholds
    equalized_odds: 0.05
```

**Coarse-grained monitoring:**
```yaml
monitoring:
  enabled: true
  window_size: 5000     # Larger window = more stable
  drift_alpha: 0.1      # Less sensitive
  
  thresholds:
    demographic_parity: 0.15
```

---

## MLflow Configuration

Controls experiment tracking and artifact logging.

```yaml
mlflow:
  enabled: true
  experiment_name: 'fairness_pipeline'
  tracking_uri: null
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | boolean | true | Enable MLflow tracking |
| `experiment_name` | string | `fairness_pipeline` | MLflow experiment name |
| `tracking_uri` | string/null | null | MLflow tracking server URI (null = local ./mlruns) |

### Examples

**Local tracking:**
```yaml
mlflow:
  enabled: true
  experiment_name: 'loan_approval_fairness'
  tracking_uri: null  # Uses ./mlruns directory
```

**Remote tracking server:**
```yaml
mlflow:
  enabled: true
  experiment_name: 'production_fairness'
  tracking_uri: 'http://mlflow-server:5000'
```

**Disable tracking:**
```yaml
mlflow:
  enabled: false
```

---

## Bootstrap Configuration

Controls statistical validation in measurement module.

```yaml
bootstrap_samples: 1000
confidence_level: 0.95
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bootstrap_samples` | integer | 1000 | Number of bootstrap resamples for CI |
| `confidence_level` | float | 0.95 | Confidence level (95% = 2σ) |

### Examples

**Quick evaluation:**
```yaml
bootstrap_samples: 500
confidence_level: 0.95
```

**High-precision measurement:**
```yaml
bootstrap_samples: 5000
confidence_level: 0.99
```

---

## Reporting Configuration

Controls output report generation.

```yaml
reporting:
  generate_json: true
  generate_markdown: true
  output_dir: 'reports/'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_json` | boolean | true | Generate JSON results file |
| `generate_markdown` | boolean | true | Generate Markdown report |
| `output_dir` | string | `reports/` | Output directory for reports |

---

## Complete Configuration Examples

### Example 1: Strict Fairness Pipeline

```yaml
data:
  path: 'data/hiring_data.csv'
  target_column: 'hired'
  protected_attribute: 'gender'
  feature_columns: null

bias_detection:
  protected_attribute: 'gender'
  thresholds:
    representation: 0.1
    proxy_correlation: 0.3
    statistical_alpha: 0.01

bias_mitigation:
  method: 'reweighting'
  params:
    method: 'inverse_propensity'
    alpha: 1.0

training:
  use_fairness_constraints: true
  constraint_type: 'equalized_odds'
  eps: 0.01

fairness_metrics:
  - demographic_parity
  - equalized_odds
  - equal_opportunity

fairness_threshold: 0.05

monitoring:
  enabled: true
  window_size: 500
  drift_alpha: 0.01
  thresholds:
    demographic_parity: 0.05
    equalized_odds: 0.05

mlflow:
  enabled: true
  experiment_name: 'strict_hiring_fairness'

bootstrap_samples: 5000
confidence_level: 0.99
```

### Example 2: Baseline Evaluation

```yaml
data:
  path: 'data/loan_data.csv'
  target_column: 'approved'
  protected_attribute: 'race'

bias_detection:
  protected_attribute: 'race'
  checks:
    - representation
    - statistical_disparity

bias_mitigation:
  method: 'none'  # No mitigation for baseline

training:
  use_fairness_constraints: false

fairness_metrics:
  - demographic_parity

fairness_threshold: 0.1

monitoring:
  enabled: false

mlflow:
  enabled: true
  experiment_name: 'baseline_evaluation'

bootstrap_samples: 1000
```

### Example 3: Production Deployment

```yaml
data:
  path: 'data/credit_scoring.csv'
  target_column: 'default'
  protected_attribute: 'age_group'

bias_detection:
  protected_attribute: 'age_group'
  reference_distribution:
    young: 0.30
    middle: 0.50
    senior: 0.20

bias_mitigation:
  method: 'reweighting'
  params:
    method: 'balanced'
    alpha: 0.8

training:
  use_fairness_constraints: true
  constraint_type: 'demographic_parity'
  eps: 0.05

fairness_metrics:
  - demographic_parity
  - equal_opportunity

fairness_threshold: 0.1

monitoring:
  enabled: true
  window_size: 2000
  drift_alpha: 0.05
  thresholds:
    demographic_parity: 0.1
    equal_opportunity: 0.1

mlflow:
  enabled: true
  experiment_name: 'credit_scoring_production'
  tracking_uri: 'http://mlflow.company.com:5000'

bootstrap_samples: 1000
confidence_level: 0.95

reporting:
  generate_json: true
  generate_markdown: true
  output_dir: 'production_reports/'
```

---

## Configuration Validation

The pipeline validates configuration on startup. Common errors:

### Missing Required Fields
```
Error: Missing required field 'data.target_column'
```
**Solution:** Ensure all required fields are specified.

### Invalid Values
```
Error: fairness_threshold must be between 0 and 1, got 1.5
```
**Solution:** Check parameter ranges in this guide.

### File Not Found
```
Error: Data file not found: data/missing.csv
```
**Solution:** Verify file paths are correct and files exist.

---

## Best Practices

1. **Version Control**: Commit config files to track experimental settings
2. **Naming Conventions**: Use descriptive experiment names in MLflow
3. **Iterative Tuning**: Start with default thresholds, then tune based on results
4. **Documentation**: Comment YAML files to explain non-obvious choices
5. **Validation**: Run on small sample first to validate configuration

---

## Command-Line Overrides

Override config file settings via command line:

```bash
# Override data path
python run_pipeline.py --config config.yml --data new_data.csv

# Use alternative config
python run_pipeline.py --config experiments/strict_config.yml
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Config file not found | Check path relative to working directory |
| YAML parsing error | Validate YAML syntax (proper indentation) |
| Invalid metric name | Check `fairness_metrics` against available options |
| MLflow connection error | Verify tracking_uri or set to null for local |
| Bootstrap too slow | Reduce `bootstrap_samples` to 500 or less |

---

## See Also

- [API Reference](api_reference.md) - Programmatic configuration
- [Best Practices](best_practices.md) - Configuration recommendations
- [Architecture](architecture_diagram.png) - System overview