# Comprehensive Testing Guide - Fairness Toolkit

**Complete testing guide for all modules: Shared, Measurement, Pipeline, Training, and Monitoring**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Testing Each Module](#testing-each-module)
   - [Shared Modules](#1-shared-modules)
   - [Measurement Module](#2-measurement-module)
   - [Pipeline Module](#3-pipeline-module)
   - [Training Module](#4-training-module)
   - [Monitoring Module](#5-monitoring-module)
3. [Integration Testing](#integration-testing)
4. [Real Dataset Testing](#real-dataset-testing)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Quick Start

### Run All Tests (Recommended First Step)

```bash
# Run complete test suite
python run_all_tests.py

# Run specific module
python run_all_tests.py --module measurement

# Verbose output
python run_all_tests.py --verbose

# Save detailed report
python run_all_tests.py --save-report
```

**Expected output**: All modules show ✅ PASSED with summary statistics.

---

### Run Individual Test Files

```bash
# Test each module separately
python test_shared_modules.py
python test_measurement_module.py
python test_pipeline_module.py
python test_training_module.py
python test_monitoring_module.py

# Test with real data
python test_with_real_dataset.py --dataset adult
```

---

## Testing Each Module

### 1. Shared Modules

**Purpose**: Test core utilities used by all other modules (schemas, validation, logging, constants).

#### Quick Test

```bash
python test_shared_modules.py
```

#### Manual Testing

```python
"""Test shared modules manually."""

# Test 1: Schemas
from shared.schemas import FairnessMetricResult, PipelineConfig

result = FairnessMetricResult(
    metric_name="demographic_parity",
    value=0.15,
    confidence_interval=(0.10, 0.20),
    group_metrics={"Group_0": 0.6, "Group_1": 0.45},
    group_sizes={"Group_0": 500, "Group_1": 500},
    interpretation="Bias detected",
    is_fair=False,
    threshold=0.1
)
print(f"✅ Created metric result: {result.metric_name} = {result.value}")

# Test 2: Validation
from shared.validation import validate_dataframe, ValidationError
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'group': np.random.choice(['A', 'B'], 100),
    'outcome': np.random.choice([0, 1], 100)
})

try:
    validate_dataframe(df, required_columns=['group', 'outcome'])
    print("✅ DataFrame validation passed")
except ValidationError as e:
    print(f"❌ Validation failed: {e}")

# Test 3: Logging
from shared.logging import get_logger, log_metric

logger = get_logger("test")
log_metric(logger, "accuracy", 0.85, {"model": "test"})
print("✅ Logging works")

# Test 4: Constants
from shared.constants import FAIRNESS_METRICS, DEFAULT_CONFIDENCE_LEVEL

print(f"✅ Available metrics: {list(FAIRNESS_METRICS.keys())}")
print(f"✅ Default confidence: {DEFAULT_CONFIDENCE_LEVEL}")
```

**What to check:**
- ✅ All schemas create objects correctly
- ✅ Validation catches invalid data
- ✅ Logging functions work
- ✅ Constants are accessible

---

### 2. Measurement Module

**Purpose**: Test fairness metric computation (demographic parity, equalized odds, etc.).

#### Quick Test

```bash
python test_measurement_module.py
```

#### Detailed Testing

```python
"""Test measurement module in detail."""
import numpy as np
from measurement_module.src import FairnessAnalyzer

# Generate biased test data
np.random.seed(42)
n_samples = 500

sensitive_features = np.random.choice([0, 1], n_samples)
y_true = np.random.choice([0, 1], n_samples)

# Add bias: group 1 gets higher approval rate
y_pred = y_true.copy()
for i in range(n_samples):
    if sensitive_features[i] == 1:
        if np.random.random() < 0.15:
            y_pred[i] = 1

# Test 1: Single metric computation
analyzer = FairnessAnalyzer(bootstrap_samples=100)

result = analyzer.compute_metric(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_features=sensitive_features,
    metric='demographic_parity',
    threshold=0.1
)

print(f"Metric: {result.metric_name}")
print(f"Value: {result.value:.4f}")
print(f"Fair: {result.is_fair}")
print(f"Group rates: {result.group_metrics}")
print(f"CI: {result.confidence_interval}")

# Test 2: Multiple metrics
all_results = analyzer.compute_all_metrics(
    y_true, y_pred, sensitive_features
)

for metric_name, result in all_results.items():
    status = "✅" if result.is_fair else "❌"
    print(f"{status} {metric_name}: {result.value:.4f}")

# Test 3: Custom threshold
result_strict = analyzer.compute_metric(
    y_true, y_pred, sensitive_features,
    metric='demographic_parity',
    threshold=0.05  # Stricter
)

print(f"\nWith strict threshold (0.05):")
print(f"Fair: {result_strict.is_fair}")
```

**What to check:**
- ✅ FairnessAnalyzer imports correctly
- ✅ Computes demographic parity, equalized odds, etc.
- ✅ Confidence intervals are reasonable
- ✅ Fair/unfair classification works
- ✅ Group metrics are computed correctly

---

### 3. Pipeline Module

**Purpose**: Test bias detection and mitigation transformers.

#### Quick Test

```bash
python test_pipeline_module.py
```

#### Detailed Testing

```python
"""Test pipeline module in detail."""
import numpy as np
import pandas as pd
from pipeline_module.src import BiasDetector, InstanceReweighting, GroupBalancer

# Create biased dataset
np.random.seed(42)
n_female = 150
n_male = 350

df = pd.DataFrame({
    'age': np.concatenate([
        np.random.normal(35, 10, n_female),
        np.random.normal(38, 10, n_male)
    ]),
    'income': np.concatenate([
        np.random.normal(50000, 15000, n_female),
        np.random.normal(65000, 15000, n_male)
    ]),
    'gender': [0] * n_female + [1] * n_male,
    'outcome': np.concatenate([
        np.random.choice([0, 1], n_female, p=[0.6, 0.4]),
        np.random.choice([0, 1], n_male, p=[0.4, 0.6])
    ])
})

# Test 1: Bias Detection
print("=" * 60)
print("BIAS DETECTION")
print("=" * 60)

detector = BiasDetector()

# Representation bias
repr_result = detector.detect_representation_bias(
    df,
    protected_attribute='gender',
    reference_distribution={0: 0.5, 1: 0.5}
)

print(f"\nRepresentation Bias:")
print(f"  Detected: {repr_result.detected}")
print(f"  Severity: {repr_result.severity}")
print(f"  Evidence: {repr_result.evidence}")

# Proxy detection
proxy_result = detector.detect_proxy_variables(
    df,
    protected_attribute='gender',
    feature_columns=['age', 'income']
)

print(f"\nProxy Variables:")
print(f"  Detected: {proxy_result.detected}")
print(f"  Proxies: {proxy_result.evidence.get('proxy_features', [])}")

# Test 2: Instance Reweighting
print("\n" + "=" * 60)
print("INSTANCE REWEIGHTING")
print("=" * 60)

X = df[['age', 'income']].values
y = df['outcome'].values
sensitive = df['gender'].values

reweighter = InstanceReweighting(method='inverse_propensity')
reweighter.fit(X, y, sensitive_features=sensitive)

weights = reweighter.get_sample_weights(sensitive)

print(f"Group weights: {reweighter.group_weights_}")
print(f"Sample weight stats:")
print(f"  Mean: {weights.mean():.3f}")
print(f"  Female avg: {weights[sensitive == 0].mean():.3f}")
print(f"  Male avg: {weights[sensitive == 1].mean():.3f}")

# Check balance
female_weight = weights[sensitive == 0].sum()
male_weight = weights[sensitive == 1].sum()
print(f"\nEffective sizes:")
print(f"  Female: {female_weight:.0f}")
print(f"  Male: {male_weight:.0f}")
print(f"  Balance ratio: {female_weight/male_weight:.3f}")

# Test 3: Group Balancing
print("\n" + "=" * 60)
print("GROUP BALANCING")
print("=" * 60)

balancer = GroupBalancer(strategy='oversample')
X_resampled, y_resampled = balancer.fit_resample(
    X, y, sensitive_features=sensitive
)

print(f"Original size: {len(X)}")
print(f"Resampled size: {len(X_resampled)}")
print(f"Increase: {(len(X_resampled) - len(X)) / len(X) * 100:.1f}%")
```

**What to check:**
- ✅ BiasDetector identifies different bias types
- ✅ InstanceReweighting balances groups
- ✅ GroupBalancer oversamples/undersamples correctly
- ✅ sklearn integration works

---

### 4. Training Module

**Purpose**: Test fairness-aware training methods.

#### Quick Test

```bash
python test_training_module.py
```

#### Detailed Testing

```python
"""Test training module in detail."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from training_module.src import ReductionsWrapper, GroupFairnessCalibrator

# Generate data
np.random.seed(42)
n_samples = 500

X = np.random.randn(n_samples, 5)
y = np.random.choice([0, 1], n_samples)
sensitive = np.random.choice([0, 1], n_samples)

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, random_state=42
)

# Test 1: Baseline model
print("=" * 60)
print("BASELINE MODEL")
print("=" * 60)

baseline = LogisticRegression(random_state=42, max_iter=1000)
baseline.fit(X_train, y_train)
y_pred_baseline = baseline.predict(X_test)

acc_baseline = (y_pred_baseline == y_test).mean()
print(f"Accuracy: {acc_baseline:.3f}")

# Compute fairness
from measurement_module.src.metrics_engine import demographic_parity_difference
dp_baseline, _, _ = demographic_parity_difference(y_test, y_pred_baseline, s_test)
print(f"Demographic Parity: {dp_baseline:.3f}")

# Test 2: Fair model with constraints
print("\n" + "=" * 60)
print("FAIR MODEL (REDUCTIONS)")
print("=" * 60)

fair_model = ReductionsWrapper(
    base_estimator=LogisticRegression(random_state=42, max_iter=1000),
    constraint='demographic_parity',
    eps=0.05
)

fair_model.fit(X_train, y_train, sensitive_features=s_train)
y_pred_fair = fair_model.predict(X_test)

acc_fair = fair_model.score(X_test, y_test)
dp_fair, _, _ = demographic_parity_difference(y_test, y_pred_fair, s_test)

print(f"Accuracy: {acc_fair:.3f} (baseline: {acc_baseline:.3f})")
print(f"Demographic Parity: {dp_fair:.3f} (baseline: {dp_baseline:.3f})")
print(f"Fairness improvement: {((dp_baseline - dp_fair) / dp_baseline * 100):.1f}%")

# Test 3: Group calibration
print("\n" + "=" * 60)
print("GROUP CALIBRATION")
print("=" * 60)

X_train_sub, X_cal, y_train_sub, y_cal, s_train_sub, s_cal = train_test_split(
    X_train, y_train, s_train, test_size=0.3, random_state=42
)

base_model = LogisticRegression(random_state=42, max_iter=1000)
base_model.fit(X_train_sub, y_train_sub)

calibrator = GroupFairnessCalibrator(base_estimator=base_model)
calibrator.fit(X_cal, y_cal, sensitive_features=s_cal)

y_pred_cal = calibrator.predict(X_test, sensitive_features=s_test)
acc_cal = (y_pred_cal == y_test).mean()

print(f"Calibrated accuracy: {acc_cal:.3f}")
print(f"Number of group calibrators: {len(calibrator.group_calibrators_)}")

# Test 4: Visualization
print("\n" + "=" * 60)
print("VISUALIZATION")
print("=" * 60)

from training_module.src import plot_pareto_frontier, plot_fairness_comparison

# Pareto frontier
results = [
    {'accuracy': 0.75, 'fairness': 0.20, 'param': 0.0},
    {'accuracy': 0.73, 'fairness': 0.15, 'param': 0.5},
    {'accuracy': 0.71, 'fairness': 0.08, 'param': 1.0},
]

fig = plot_pareto_frontier(results)
print("✅ Pareto frontier plot created")

# Fairness comparison
models = {
    'Baseline': {'demographic_parity': dp_baseline, 'equalized_odds': 0.18},
    'Fair Model': {'demographic_parity': dp_fair, 'equalized_odds': 0.10},
}

fig2 = plot_fairness_comparison(models)
print("✅ Fairness comparison plot created")
```

**What to check:**
- ✅ ReductionsWrapper trains with fairness constraints
- ✅ Fairness improves without major accuracy loss
- ✅ GroupFairnessCalibrator calibrates per group
- ✅ Visualization functions work

---

### 5. Monitoring Module

**Purpose**: Test real-time monitoring, drift detection, and alerting.

#### Quick Test

```bash
python test_monitoring_module.py
```

#### Detailed Testing

```python
"""Test monitoring module in detail."""
import numpy as np
from datetime import datetime
from monitoring_module.src import (
    RealTimeFairnessTracker,
    FairnessDriftDetector,
    ThresholdAlertSystem
)

# Test 1: Real-time tracking
print("=" * 60)
print("REAL-TIME TRACKING")
print("=" * 60)

tracker = RealTimeFairnessTracker(
    window_size=500,
    metrics=['demographic_parity'],
    min_samples=50
)

# Simulate streaming data
np.random.seed(42)
for batch_num in range(10):
    n = 100
    y_pred = np.random.choice([0, 1], n)
    y_true = np.random.choice([0, 1], n)
    sensitive = np.random.choice([0, 1], n)
    
    metrics = tracker.add_batch(y_pred, y_true, sensitive)
    
    if metrics:
        print(f"Batch {batch_num + 1}: DP = {metrics.get('demographic_parity', 0):.4f}")

# Get summary
summary = tracker.get_summary_statistics()
print(f"\nSummary:")
for metric, stats in summary.items():
    print(f"  {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

# Test 2: Drift detection
print("\n" + "=" * 60)
print("DRIFT DETECTION")
print("=" * 60)

detector = FairnessDriftDetector(alpha=0.05)

# Reference period
y_true_ref = np.random.choice([0, 1], 500)
y_pred_ref = np.random.choice([0, 1], 500)
s_ref = np.random.choice([0, 1], 500)

detector.set_reference(y_true_ref, y_pred_ref, s_ref)
print("✅ Reference period set")

# Test with similar data (no drift)
y_true_1 = np.random.choice([0, 1], 500)
y_pred_1 = np.random.choice([0, 1], 500)
s_1 = np.random.choice([0, 1], 500)

drift_result = detector.detect_drift(y_true_1, y_pred_1, s_1)
print(f"\nDrift detected: {drift_result['drift_detected']}")

if drift_result['drift_detected']:
    print(f"Drifted metrics: {drift_result['drifted_metrics']}")
    print(f"Severity: {drift_result['severity']}")

# Test 3: Alerting
print("\n" + "=" * 60)
print("ALERTING")
print("=" * 60)

alerter = ThresholdAlertSystem(
    thresholds={'demographic_parity': 0.10}
)

# Test with violation
metrics_bad = {'demographic_parity': 0.15}
alert = alerter.check_thresholds(metrics_bad)

if alert:
    print(f"🚨 Alert triggered:")
    print(f"  Severity: {alert.severity}")
    print(f"  Metric: {alert.metric_name}")
    print(f"  Value: {alert.current_value:.3f}")
    print(f"  Threshold: {alert.threshold:.3f}")
    print(f"  Message: {alert.message}")

# Test with no violation
metrics_good = {'demographic_parity': 0.08}
alert2 = alerter.check_thresholds(metrics_good)

if alert2:
    print("\n Alert when shouldn't")
else:
    print("\n✅ No alert for compliant metrics")
```

**What to check:**
- ✅ RealTimeFairnessTracker processes streaming data
- ✅ FairnessDriftDetector identifies drift
- ✅ ThresholdAlertSystem generates alerts
- ✅ Time series data is collected

---

## Integration Testing

### End-to-End Workflow

```python
"""
Complete workflow: Load data → Detect bias → Mitigate → Train → Monitor
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("=" * 70)
print("END-TO-END INTEGRATION TEST")
print("=" * 70)

# Step 1: Load and validate data
print("\n[1/6] Loading data...")
from shared.validation import validate_dataframe, validate_protected_attribute

np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.randn(500),
    'feature2': np.random.randn(500),
    'gender': np.random.choice([0, 1], 500),
    'outcome': np.random.choice([0, 1], 500)
})

validate_dataframe(df, required_columns=['gender', 'outcome'])
validate_protected_attribute(df, 'gender')
print("✅ Data validated")

# Step 2: Detect bias in data
print("\n[2/6] Detecting bias...")
from pipeline_module.src import BiasDetector

detector = BiasDetector()
bias_results = detector.detect_all_bias_types(
    df,
    protected_attribute='gender',
    reference_distribution={0: 0.5, 1: 0.5}
)

bias_detected = any(r.detected for r in bias_results.values())
print(f"Bias detected: {bias_detected}")

# Step 3: Apply mitigation
print("\n[3/6] Applying mitigation...")
from pipeline_module.src import InstanceReweighting

X = df[['feature1', 'feature2']].values
y = df['outcome'].values
s = df['gender'].values

X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, s, test_size=0.3, random_state=42
)

reweighter = InstanceReweighting()
reweighter.fit(X_train, y_train, sensitive_features=s_train)
weights = reweighter.get_sample_weights(s_train)
print("✅ Sample weights computed")

# Step 4: Train fair model
print("\n[4/6] Training fair model...")
from training_module.src import ReductionsWrapper

model = ReductionsWrapper(
    base_estimator=LogisticRegression(random_state=42, max_iter=1000),
    constraint='demographic_parity',
    eps=0.05
)

model.fit(X_train, y_train, sensitive_features=s_train)
y_pred = model.predict(X_test)
print("✅ Model trained")

# Step 5: Measure fairness
print("\n[5/6] Measuring fairness...")
from measurement_module.src import FairnessAnalyzer

analyzer = FairnessAnalyzer(bootstrap_samples=100)
result = analyzer.compute_metric(
    y_test, y_pred, s_test,
    metric='demographic_parity',
    threshold=0.1
)

print(f"Demographic Parity: {result.value:.4f}")
print(f"Fair: {result.is_fair}")
print(f"Group metrics: {result.group_metrics}")

# Step 6: Setup monitoring
print("\n[6/6] Setting up monitoring...")
from monitoring_module.src import RealTimeFairnessTracker, ThresholdAlertSystem

tracker = RealTimeFairnessTracker(
    window_size=200,
    metrics=['demographic_parity']
)

alerter = ThresholdAlertSystem(
    thresholds={'demographic_parity': 0.10}
)

# Simulate monitoring
tracker.add_batch(y_pred, y_test, s_test)
summary = tracker.get_summary_statistics()

print("✅ Monitoring configured")
print(f"Current DP: {summary['demographic_parity']['current']:.4f}")

alert = alerter.check_thresholds({'demographic_parity': result.value})
if alert:
    print(f"🚨 Alert: {alert.severity}")
else:
    print("✅ No alerts")

print("\n" + "=" * 70)
print("✅ INTEGRATION TEST COMPLETE!")
print("=" * 70)
print("\nWorkflow successful:")
print("  1. Data validation ✅")
print("  2. Bias detection ✅")
print("  3. Bias mitigation ✅")
print("  4. Fair model training ✅")
print("  5. Fairness measurement ✅")
print("  6. Production monitoring ✅")
```

---

## Real Dataset Testing

### Test with UCI Adult Census Dataset

```bash
# Download and test with Adult dataset
python test_with_real_dataset.py --dataset adult

# Test with COMPAS dataset
python test_with_real_dataset.py --dataset compas

# Test with your own data
python test_with_real_dataset.py --data mydata.csv --protected-attr gender --target outcome
```

### Manual Real Dataset Test

```python
"""Test with real-world fairness benchmark."""
import pandas as pd
from shared.validation import validate_dataframe, validate_protected_attribute
from shared.schemas import DatasetMetadata

# Load Adult Census dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
df = df.dropna()

df['income_binary'] = (df['income'] == '>50K').astype(int)
df['sex_binary'] = (df['sex'] == 'Male').astype(int)

print(f"Loaded {len(df)} samples")

# Validate
validate_dataframe(df, required_columns=['sex_binary', 'income_binary'])
validate_protected_attribute(df, 'sex_binary')

# Create metadata
group_dist = df['sex_binary'].value_counts().to_dict()
metadata = DatasetMetadata(
    name="adult_census",
    n_samples=len(df),
    n_features=len(df.columns) - 2,
    task_type="binary_classification",
    protected_attribute="sex_binary",
    protected_groups=list(group_dist.keys()),
    group_distribution=group_dist
)

print(f"\nDataset Info:")
print(f"  Samples: {metadata.n_samples}")
print(f"  Groups: {metadata.protected_groups}")
print(f"  Distribution: {metadata.group_distribution}")
print(f"  Imbalance ratio: {metadata.imbalance_ratio:.2f}")

# Compute baseline fairness
groups = df['sex_binary'].unique()
group_rates = {}

for group in groups:
    mask = df['sex_binary'] == group
    rate = df[mask]['income_binary'].mean()
    group_rates[f"Group_{group}"] = rate

dp_diff = max(group_rates.values()) - min(group_rates.values())

print(f"\nBaseline Fairness:")
print(f"  Demographic Parity: {dp_diff:.4f}")
for group, rate in group_rates.items():
    print(f"    {group}: {rate:.4f}")
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

**Problem**: `ModuleNotFoundError: No module named 'measurement_module'`

**Solution**:
```bash
# Check directory structure
ls -R | grep -E "^(measurement|pipeline|training|monitoring|shared)_module"

# Ensure __init__.py files exist
find . -name "__init__.py" -type f

# Add to Python path if needed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### Issue 2: Missing Dependencies

**Problem**: `ModuleNotFoundError: No module named 'scipy'`

**Solution**:
```bash
# Install required packages
pip install numpy pandas scikit-learn scipy matplotlib seaborn

# Optional dependencies
pip install fairlearn torch plotly  # For advanced features
```

#### Issue 3: Test Timeout

**Problem**: Tests take too long or hang

**Solution**:
```python
# Reduce bootstrap samples for testing
analyzer = FairnessAnalyzer(bootstrap_samples=100)  # Instead of 1000

# Use smaller datasets
n_samples = 500  # Instead of 10000

# Skip long-running tests
python run_all_tests.py --quick
```

#### Issue 4: Numerical Instability

**Problem**: `RuntimeWarning: invalid value encountered`

**Solution**:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Ensure sufficient sample sizes
MIN_GROUP_SIZE = 30  # At least 30 samples per group

# Check for edge cases
assert len(np.unique(y_true)) > 1, "Need both classes"
assert len(np.unique(sensitive)) > 1, "Need multiple groups"
```

#### Issue 5: Fairlearn Not Available

**Problem**: `ReductionsWrapper requires fairlearn`

**Solution**:
```bash
# Install fairlearn
pip install fairlearn

# Or skip fairlearn-dependent tests
# They will be marked as skipped in test output
```

---

## Best Practices

### 1. Test Isolation

```python
"""Each test should be independent."""
import numpy as np

def test_metric_computation():
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create fresh data
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)
    
    # Test
    result = compute_metric(y_true, y_pred)
    
    # Assert
    assert result is not None
    assert 0 <= result <= 1
```

### 2. Use Realistic Data

```python
"""Test with realistic distributions."""

def create_realistic_test_data():
    """Mimic real-world data characteristics."""
    np.random.seed(42)
    
    # Realistic sample sizes
    n_samples = 1000
    
    # Realistic class imbalance (70/30)
    y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Realistic group sizes (60/40)
    sensitive = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    
    # Correlated features
    X = np.random.randn(n_samples, 10)
    X[:, 0] += sensitive * 0.5  # Correlation with protected attribute
    
    return X, y, sensitive
```

### 3. Test Edge Cases

```python
"""Always test boundary conditions."""

def test_edge_cases():
    # Perfect fairness
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    sensitive = np.array([0, 0, 1, 1])
    
    result = compute_fairness(y_true, y_pred, sensitive)
    assert result.value == 0.0
    
    # Complete bias
    y_pred_biased = np.array([0, 0, 1, 1])
    result_biased = compute_fairness(y_true, y_pred_biased, sensitive)
    assert result_biased.value > 0.3
    
    # Small sample size
    with pytest.raises(ValidationError):
        compute_fairness(y_true[:5], y_pred[:5], sensitive[:5])
```

### 4. Validate Outputs

```python
"""Check output types and ranges."""

def test_output_validation():
    result = analyzer.compute_metric(...)
    
    # Type checks
    assert isinstance(result, FairnessMetricResult)
    assert isinstance(result.value, float)
    assert isinstance(result.group_metrics, dict)
    
    # Range checks
    assert 0 <= result.value <= 1
    assert len(result.confidence_interval) == 2
    assert result.confidence_interval[0] < result.confidence_interval[1]
    
    # Completeness checks
    assert result.metric_name is not None
    assert result.interpretation is not None
```

### 5. Document Test Results

```python
"""Save test outputs for review."""

def save_test_results():
    results = run_all_tests()
    
    # Save summary
    with open('test_results/summary.txt', 'w') as f:
        f.write(f"Tests run: {len(results)}\n")
        f.write(f"Passed: {sum(r['passed'] for r in results)}\n")
        f.write(f"Failed: {sum(not r['passed'] for r in results)}\n")
    
    # Save detailed results
    pd.DataFrame(results).to_csv('test_results/detailed.csv')
    
    # Generate plots
    plot_test_coverage(results)
```

---

## Quick Reference

### Test Commands

| Command | Purpose |
|---------|---------|
| `python run_all_tests.py` | Run all tests |
| `python run_all_tests.py --module shared` | Test specific module |
| `python test_shared_modules.py` | Test shared utilities |
| `python test_measurement_module.py` | Test fairness metrics |
| `python test_pipeline_module.py` | Test bias detection/mitigation |
| `python test_training_module.py` | Test fair training methods |
| `python test_monitoring_module.py` | Test monitoring system |
| `python test_with_real_dataset.py --dataset adult` | Test with real data |

### Module Import Paths

```python
# Shared
from shared.schemas import FairnessMetricResult, PipelineConfig
from shared.validation import validate_dataframe, ValidationError
from shared.logging import get_logger, log_metric

# Measurement
from measurement_module.src import FairnessAnalyzer

# Pipeline
from pipeline_module.src import BiasDetector, InstanceReweighting, GroupBalancer

# Training
from training_module.src import ReductionsWrapper, GroupFairnessCalibrator

# Monitoring
from monitoring_module.src import (
    RealTimeFairnessTracker,
    FairnessDriftDetector,
    ThresholdAlertSystem
)
```

---

## Next Steps

1. ✅ Run `python run_all_tests.py` to verify all modules work
2. ✅ Test with your own data using `test_with_real_dataset.py`
3. ✅ Run integration test to verify end-to-end workflow
4. ✅ Set up continuous testing in your CI/CD pipeline
5. ✅ Create custom tests for your specific use cases

---

