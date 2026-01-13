# Fairness Development Toolkit


## 🎯 Overview

The Pipeline Module provides systematic bias detection, mitigation transformers and monitoring bias in machine learning pipelines. It implements all five functionalities required for enterprise-grade fairness systems:

1. ✅ **Comprehensive Bias Detection Engine**
2. ✅ **sklearn-Compatible Transformer Library**
3. ✅ **Automated CI/CD Validation System**
4. ✅ **Configurable Pipeline Framework**
5. ✅ **Post-Processing Calibration Tools**

---

## 📚 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [1. Bias Detection](#1-bias-detection)
  - [2. Bias Mitigation](#2-bias-mitigation)
  - [3. Fair Model Training](#3-fair-model-training)
  - [4. Calibration](#4-calibration)
  - [5. Monitoring](#5-monitoring)
  - [6. CI/CD Integration](#6-cicd-integration)
- [Complete Pipeline Example](#-complete-pipeline-example)
- [API Reference](#-api-reference)
- [Testing](#-testing)
- [Contributing](#-contributing)

---

## 🚀 Quick Start

### 30-Second Example

```python
from bias_detection import BiasDetector
from reweighting import InstanceReweighting
from calibration import GroupCalibrator

# 1. Detect bias
detector = BiasDetector()
bias_results = detector.detect_all_bias_types(df, protected_attribute='gender')

# 2. Mitigate bias
reweighter = InstanceReweighting()
X, y, weights = reweighter.fit_transform(X_train, y_train, sensitive_features=gender)

# 3. Train fair model
model.fit(X, y, sample_weight=weights)

# 4. Calibrate probabilities
calibrator = GroupCalibrator(method='platt')
calibrator.fit(y_proba, y_true, sensitive_features=gender)
y_proba_calibrated = calibrator.transform(y_proba, sensitive_features=gender)
```

---

## ✨ Features

### 1. Comprehensive Bias Detection 🔍

Automatically detects three types of bias in your data:

| Bias Type | What It Detects | Example |
|-----------|----------------|---------|
| **Representation Bias** | Demographic distribution mismatches | Dataset 30% female vs 51% population |
| **Proxy Variables** | Features correlated with protected attributes | ZIP code correlated 0.7 with race |
| **Statistical Disparity** | Features distributed differently across groups | Income differs significantly by gender |

**Key Components:**
- `BiasDetector` - Automated bias scanning engine
- `BiasReportGenerator` - JSON and Markdown reporting
- Configurable thresholds and statistical tests

### 2. sklearn-Compatible Transformers 🔧

Modular transformers that plug directly into sklearn pipelines:

- **InstanceReweighting** - Balance groups through sample weighting
- **GroupBalancer** - Balance through oversampling/undersampling
- **FeatureScalerByGroup** - Group-specific feature normalization
- **DisparateImpactRemover** - Feature transformation (stub)

All transformers:
- ✅ Inherit from `sklearn.base.TransformerMixin`
- ✅ Compatible with `sklearn.pipeline.Pipeline`
- ✅ Support `fit()`, `transform()`, `fit_transform()`
- ✅ Include comprehensive docstrings and examples

### 3. Automated CI/CD Validation 🛡️

Enforces fairness as a deployment gate:

- **Pytest Fairness Gates** - Block deployment if thresholds breached
- **GitHub Actions Workflow** - Auto-run on every PR
- **Deployment Criteria** - Configurable severity thresholds
- **Automated Reporting** - PR comments with bias reports

**Example Test:**
```python
def test_representation_bias_gate(sample_dataset):
    """Block deployment if representation bias detected."""
    FairnessAssertion.assert_representation_fairness(
        sample_dataset,
        protected_attribute='gender',
        reference_distribution={'M': 0.5, 'F': 0.5},
        threshold=0.15  # 15% max deviation
    )
```

### 4. Configurable Pipeline Framework ⚙️

YAML-driven configuration for end-to-end workflows:

```yaml
bias_detection:
  protected_attribute: "gender"
  reference_distribution:
    Male: 0.49
    Female: 0.51
  thresholds:
    representation_threshold: 0.2
    proxy_threshold: 0.5

bias_mitigation:
  method: "reweighting"
  params:
    alpha: 1.0

post_processing:
  calibration:
    enabled: true
    method: "platt"
```

**Features:**
- Dynamic pipeline construction from config
- Validation before execution
- Parameter tuning without code changes
- Version-controlled fairness policies

### 5. Post-Processing Calibration 📊

Ensure probability scores mean the same across all groups:

- **Group-Specific Calibration** - Per-group Platt scaling
- **Multiple Methods** - Platt, Isotonic, Temperature scaling
- **ECE Tracking** - Expected Calibration Error monitoring
- **Reliability Diagrams** - Visualization of calibration quality

---

## 📦 Installation

### Requirements

- Python 3.8+
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- scipy >= 1.7.0
- PyYAML >= 5.4.0
- matplotlib >= 3.4.0 (for visualizations)

### Install

```bash
# Clone repository
git clone https://github.com/your-org/fairness-toolkit.git
cd fairness-toolkit

# Install dependencies
pip install -r requirements.txt

# Run tests to verify installation
pytest pipeline_module/tests/ -v
```

---

## 📖 Usage Guide

### 1. Bias Detection

#### Basic Detection

```python
from bias_detection import BiasDetector

detector = BiasDetector(
    representation_threshold=0.2,  # 20% max deviation
    proxy_threshold=0.5,           # 50% correlation
    statistical_alpha=0.05         # 5% significance
)

# Detect representation bias
repr_result = detector.detect_representation_bias(
    df,
    protected_attribute='gender',
    reference_distribution={'Female': 0.51, 'Male': 0.49}
)

if repr_result.detected:
    print(f"Severity: {repr_result.severity}")
    print(f"Affected groups: {repr_result.affected_groups}")
    print(f"Recommendations: {repr_result.recommendations}")
```

#### Comprehensive Detection

```python
# Run all bias checks at once
all_results = detector.detect_all_bias_types(
    df,
    protected_attribute='gender',
    reference_distribution={'Female': 0.5, 'Male': 0.5},
    feature_columns=['age', 'income', 'credit_score']
)

# Generate report
from bias_report import BiasReportGenerator

reporter = BiasReportGenerator()
for name, result in all_results.items():
    reporter.add_result(name, result)

reporter.save_json('reports/bias_report.json')
reporter.save_markdown('reports/bias_report.md')
reporter.print_summary()
```

### 2. Bias Mitigation

#### Instance Reweighting

```python
from reweighting import InstanceReweighting

# Initialize reweighter
reweighter = InstanceReweighting(
    method='inverse_propensity',
    alpha=1.0  # 1.0 = full reweighting
)

# Fit and get weights
reweighter.fit(X_train, y_train, sensitive_features=gender_train)
weights = reweighter.get_sample_weights(gender_train)

# Or use fit_transform
X, y, weights = reweighter.fit_transform(
    X_train, y_train, sensitive_features=gender_train
)

# Train model with weights
model = LogisticRegression()
model.fit(X, y, sample_weight=weights)
```

#### Group Balancing (Resampling)

```python
from resampling import GroupBalancer

# Oversample minority group
balancer = GroupBalancer(strategy='oversample', random_state=42)
X_balanced, y_balanced = balancer.fit_resample(
    X_train, y_train, sensitive_features=gender_train
)

# Train on balanced data
model.fit(X_balanced, y_balanced)
```

### 3. Fair Model Training

#### With Fairness Constraints

```python
from training_module import ReductionsWrapper

# Train with demographic parity constraint
fair_model = ReductionsWrapper(
    base_estimator=LogisticRegression(),
    constraint='demographic_parity',
    eps=0.05  # 5% tolerance
)

fair_model.fit(X_train, y_train, sensitive_features=gender_train)
```

#### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Build fair pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

# Get reweighting weights
reweighter = InstanceReweighting()
reweighter.fit(X_train, y_train, sensitive_features=gender_train)
weights = reweighter.get_sample_weights(gender_train)

# Train with weights
pipeline.fit(X_train, y_train, model__sample_weight=weights)
```

### 4. Calibration

#### Group-Specific Calibration

```python
from calibration import GroupCalibrator, CalibrationEvaluator

# Get uncalibrated probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Calibrate per group
calibrator = GroupCalibrator(method='platt', global_calibration=False)
calibrator.fit(y_proba, y_true, sensitive_features=gender)
y_proba_calibrated = calibrator.transform(y_proba, sensitive_features=gender)

# Evaluate calibration
ece_before = CalibrationEvaluator.compute_ece(y_true, y_proba)
ece_after = CalibrationEvaluator.compute_ece(y_true, y_proba_calibrated)

print(f"ECE improvement: {ece_before - ece_after:.4f}")
```

#### Calibration Report

```python
report = CalibrationEvaluator.generate_calibration_report(
    y_true, y_proba_before, y_proba_after, sensitive_features
)

print(f"Overall ECE: {report['overall']['ece_after']:.4f}")
print(f"Group ECE disparity: {report['max_group_disparity']['after']:.4f}")
```

### 5. Monitoring

#### Real-Time Tracking

```python
from monitoring import FairnessMonitor, FairnessMetricsCalculator

# Initialize monitor
monitor = FairnessMonitor(log_file='logs/fairness_metrics.json')

# Calculate and log metrics
metrics = FairnessMetricsCalculator.calculate_all_metrics(
    y_true, y_pred, y_proba, sensitive_features
)

monitor.log_metrics(
    model_version='v1.0',
    metrics=metrics,
    sensitive_attribute='gender'
)

# Check for drift
drift_report = monitor.check_drift(
    baseline_version='v1.0',
    current_version='v1.1',
    threshold=0.1
)

if drift_report['alerts']:
    print(f"⚠️  {len(drift_report['alerts'])} fairness drift alerts!")
```

#### Alert System

```python
from monitoring import FairnessAlertSystem

alert_system = FairnessAlertSystem(thresholds={
    'demographic_parity': 0.1,
    'ece': 0.05
})

alerts = alert_system.check_metrics(metrics, model_version='v1.0')

if alerts:
    for alert in alerts:
        print(f"{alert['severity'].upper()}: {alert['message']}")
```

### 6. CI/CD Integration

#### Run Fairness Tests

```bash
# Run all fairness gate tests
pytest pipeline_module/tests/test_fairness_gates.py -v -m fairness

# Run with coverage
pytest pipeline_module/tests/ --cov=pipeline_module/src --cov-report=html
```

#### GitHub Actions Workflow

The toolkit includes a complete GitHub Actions workflow (`.github/workflows/fairness_validation.yml`) that:

1. Runs on every PR and push
2. Executes all fairness tests
3. Generates bias detection reports
4. Blocks merge if criteria not met
5. Comments PR with results

**Deployment Criteria:**
```yaml
deployment_criteria:
  max_high_severity_bias: 0      # Zero tolerance for high severity
  max_medium_severity_bias: 2    # Max 2 medium severity issues
  max_demographic_parity: 0.15   # 15% max disparity
  max_ece: 0.08                  # 8% max calibration error
```

---

## 🔄 Complete Pipeline Example

### End-to-End Workflow

```python
# 1. Load and explore data
df = pd.read_csv('data/loan_applications.csv')

# 2. Detect bias
detector = BiasDetector()
bias_results = detector.detect_all_bias_types(
    df, protected_attribute='gender'
)

# 3. Split data
X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
    X, y, sensitive, test_size=0.3, stratify=y
)

# 4. Mitigate bias
reweighter = InstanceReweighting()
X_train, y_train, weights = reweighter.fit_transform(
    X_train, y_train, sensitive_features=s_train
)

# 5. Train fair model
model = LogisticRegression()
model.fit(X_train, y_train, sample_weight=weights)

# 6. Calibrate probabilities
y_proba = model.predict_proba(X_test)[:, 1]
calibrator = GroupCalibrator(method='platt')
calibrator.fit(y_proba, y_test, sensitive_features=s_test)
y_proba_calibrated = calibrator.transform(y_proba, sensitive_features=s_test)

# 7. Validate fairness
analyzer = FairnessAnalyzer()
result = analyzer.compute_metric(
    y_test, model.predict(X_test), s_test,
    metric='demographic_parity'
)

print(f"Fair: {result.is_fair}")
print(f"Demographic Parity: {result.value:.4f}")

# 8. Setup monitoring
monitor = FairnessMonitor()
metrics = FairnessMetricsCalculator.calculate_all_metrics(
    y_test, model.predict(X_test), y_proba_calibrated, s_test
)
monitor.log_metrics('v1.0', metrics)
```

### Using Configuration File

```python
from config_loader import load_config
from pipeline_builder import build_fairness_pipeline

# Load config
config = load_config('config.yml')

# Build pipeline dynamically
pipeline = build_fairness_pipeline(config, model=LogisticRegression())

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

### CLI Usage

```bash
# Run bias detection from command line
python run_bias_detection.py \
    --data data/loans.csv \
    --protected-attr gender \
    --output reports/bias_report.json \
    --markdown reports/bias_report.md \
    --fail-on-high-severity

# Run complete pipeline
python run_pipeline.py --config config.yml --data data/loans.csv
```

---

## 📚 API Reference

### Core Modules

#### BiasDetector

```python
BiasDetector(
    representation_threshold: float = 0.2,
    proxy_threshold: float = 0.5,
    statistical_alpha: float = 0.05
)
```

**Methods:**
- `detect_representation_bias(df, protected_attribute, reference_distribution)` → `BiasDetectionResult`
- `detect_proxy_variables(df, protected_attribute, feature_columns)` → `BiasDetectionResult`
- `detect_statistical_disparity(df, protected_attribute, feature_columns)` → `BiasDetectionResult`
- `detect_all_bias_types(df, protected_attribute, ...)` → `Dict[str, BiasDetectionResult]`

#### InstanceReweighting

```python
InstanceReweighting(
    method: str = 'inverse_propensity',
    alpha: float = 1.0
)
```

**Methods:**
- `fit(X, y, sensitive_features)` → `self`
- `transform(X)` → `np.ndarray`
- `get_sample_weights(sensitive_features)` → `np.ndarray`
- `fit_transform(X, y, sensitive_features)` → `Tuple[np.ndarray, np.ndarray, np.ndarray]`

#### GroupCalibrator

```python
GroupCalibrator(
    method: str = 'platt',  # 'platt', 'isotonic', or 'temperature'
    global_calibration: bool = False
)
```

**Methods:**
- `fit(y_proba, y_true, sensitive_features)` → `self`
- `transform(y_proba, sensitive_features)` → `np.ndarray`

#### FairnessMonitor

```python
FairnessMonitor(log_file: Optional[str] = None)
```

**Methods:**
- `log_metrics(model_version, metrics, sensitive_attribute, metadata)` → `None`
- `get_metrics(model_version)` → `List[Dict]`
- `check_drift(baseline_version, current_version, threshold)` → `Dict`

### Data Schemas

#### BiasDetectionResult

```python
@dataclass
class BiasDetectionResult:
    bias_type: str              # Type of bias detected
    detected: bool              # Whether bias was found
    severity: str               # 'low', 'medium', or 'high'
    affected_groups: List[str]  # Groups/features affected
    evidence: Dict              # Statistical evidence
    recommendations: List[str]  # Actionable suggestions
    timestamp: datetime
```

---

## 🧪 Testing

### Run All Tests

```bash
# Run all tests with coverage
pytest pipeline_module/tests/ -v --cov=pipeline_module/src --cov-report=html

# Run specific test modules
pytest pipeline_module/tests/test_bias_detection.py -v
pytest pipeline_module/tests/test_transformers.py -v
pytest pipeline_module/tests/test_fairness_gates.py -v -m fairness

# Run integration tests
pytest pipeline_module/tests/test_pipeline_fairness.py -v
```

### Test Coverage

Expected coverage: **>90%**

```
pipeline_module/src/bias_detection.py        95%
pipeline_module/src/transformers/reweighting.py   93%
pipeline_module/src/calibration.py           91%
pipeline_module/src/monitoring.py            89%
```

### Quick Validation

```bash
# Run quick validation script
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
✅ Comprehensive bias detection completed

[4/5] Testing InstanceReweighting...
✅ fit_transform works correctly

[5/5] Testing GroupBalancer...
✅ Resampling completed

============================================================
✅ ALL PIPELINE MODULE TESTS COMPLETED!
============================================================
```

---

## 📊 Demo Notebook

See `demo.ipynb` for a complete interactive demonstration covering:

1. Data generation with known biases
2. Bias detection and visualization
3. Baseline model training
4. Bias mitigation with reweighting
5. Fair model training and evaluation
6. Calibration and ECE improvement
7. Monitoring setup

Run the demo:
```bash
jupyter notebook demo.ipynb
```

---

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone repo
git clone https://github.com/your-org/fairness-toolkit.git

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linting
black pipeline_module/
flake8 pipeline_module/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This toolkit implements industry best practices for fairness in ML systems, drawing inspiration from:

- [IBM AI Fairness 360 (AIF360)](https://aif360.mybluemix.net/)
- [Microsoft Fairlearn](https://fairlearn.org/)
- [Google's What-If Tool](https://pair-code.github.io/what-if-tool/)
- Academic research on algorithmic fairness

---

## 📞 Support

- **Documentation**: See module READMEs and docstrings
- **Issues**: [GitHub Issues](https://github.com/your-org/fairness-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/fairness-toolkit/discussions)

---

## 🗺️ Roadmap

### Current (v1.0)
- ✅ Bias detection (3 types)
- ✅ Reweighting and resampling
- ✅ Group calibration
- ✅ CI/CD gates
- ✅ Monitoring framework

### Future (v2.0)
- ⬜ SMOTE integration
- ⬜ Multi-attribute fairness
- ⬜ Causal fairness metrics
- ⬜ Explainability integration
- ⬜ AutoML fairness optimization

---

**Built with ❤️ for fair and responsible AI**