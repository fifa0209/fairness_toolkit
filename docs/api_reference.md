# API Reference

## Overview

This document provides complete API reference for all modules in the Fairness Pipeline Toolkit. Each module exposes programmatic interfaces for integration into custom ML workflows.

---

## Table of Contents

1. [Measurement Module](#measurement-module)
2. [Pipeline Module](#pipeline-module)
3. [Training Module](#training-module)
4. [Monitoring Module](#monitoring-module)
5. [Shared Utilities](#shared-utilities)
6. [Orchestrator](#orchestrator)

---

## Measurement Module

### FairnessAnalyzer

Main class for computing fairness metrics with statistical validation.

```python
from measurement_module import FairnessAnalyzer
```

#### Constructor

```python
FairnessAnalyzer(
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    min_group_size: int = 30,
    random_state: int = 42
)
```

**Parameters:**
- `confidence_level` (float): Confidence level for intervals (default: 0.95)
- `bootstrap_samples` (int): Number of bootstrap resamples (default: 1000)
- `min_group_size` (int): Minimum samples per group (default: 30)
- `random_state` (int): Random seed for reproducibility (default: 42)

#### Methods

##### compute_metric()

```python
compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric: str = 'demographic_parity',
    threshold: float = 0.1
) -> FairnessResult
```

Compute single fairness metric with confidence interval.

**Parameters:**
- `y_true`: Ground truth labels (binary)
- `y_pred`: Predicted labels (binary)
- `sensitive_features`: Protected attribute values
- `metric`: Metric name (`demographic_parity`, `equalized_odds`, `equal_opportunity`)
- `threshold`: Fairness violation threshold

**Returns:** `FairnessResult` object with:
- `value` (float): Metric value
- `is_fair` (bool): Whether metric meets threshold
- `confidence_interval` (tuple): (lower, upper) bounds
- `effect_size` (float): Cohen's d effect size
- `group_metrics` (dict): Per-group breakdown

**Example:**
```python
analyzer = FairnessAnalyzer(bootstrap_samples=1000)
result = analyzer.compute_metric(
    y_true=y_test,
    y_pred=predictions,
    sensitive_features=gender,
    metric='demographic_parity',
    threshold=0.1
)

print(f"Bias: {result.value:.3f}")
print(f"95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
print(f"Fair: {result.is_fair}")
```

##### compute_all_metrics()

```python
compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    threshold: float = 0.1
) -> Dict[str, FairnessResult]
```

Compute all available fairness metrics.

**Returns:** Dictionary mapping metric names to `FairnessResult` objects.

**Example:**
```python
results = analyzer.compute_all_metrics(y_test, predictions, gender)

for metric_name, result in results.items():
    print(f"{metric_name}: {result.value:.3f} (Fair: {result.is_fair})")
```

---

## Pipeline Module

### BiasDetector

Detects various types of bias in datasets.

```python
from pipeline_module import BiasDetector
```

#### Constructor

```python
BiasDetector(
    representation_threshold: float = 0.2,
    proxy_threshold: float = 0.5,
    statistical_alpha: float = 0.05
)
```

**Parameters:**
- `representation_threshold`: Max deviation from expected representation
- `proxy_threshold`: Correlation threshold for proxy detection
- `statistical_alpha`: Significance level for statistical tests

#### Methods

##### detect_representation_bias()

```python
detect_representation_bias(
    X: pd.DataFrame,
    protected_attribute: str,
    reference_distribution: Optional[Dict] = None
) -> BiasResult
```

Detect underrepresentation or overrepresentation of protected groups.

**Parameters:**
- `X`: Input dataframe
- `protected_attribute`: Column name of protected attribute
- `reference_distribution`: Expected proportions (if known)

**Returns:** `BiasResult` with:
- `detected` (bool): Whether bias detected
- `severity` (str): `low`, `medium`, or `high`
- `details` (dict): Additional information

**Example:**
```python
detector = BiasDetector(representation_threshold=0.2)
result = detector.detect_representation_bias(
    df,
    protected_attribute='gender',
    reference_distribution={0: 0.5, 1: 0.5}  # Expected 50/50 split
)

if result.detected:
    print(f"Representation bias detected: {result.severity}")
```

##### detect_proxy_features()

```python
detect_proxy_features(
    X: pd.DataFrame,
    protected_attribute: str
) -> BiasResult
```

Identify features highly correlated with protected attribute.

**Returns:** `BiasResult` with proxy features listed in `details['proxy_features']`.

##### detect_statistical_disparity()

```python
detect_statistical_disparity(
    X: pd.DataFrame,
    y: np.ndarray,
    protected_attribute: str
) -> BiasResult
```

Test for statistical differences in outcomes across groups.

##### detect_all_bias_types()

```python
detect_all_bias_types(
    X: pd.DataFrame,
    protected_attribute: str,
    y: Optional[np.ndarray] = None,
    reference_distribution: Optional[Dict] = None
) -> Dict[str, BiasResult]
```

Run all bias detection checks.

**Returns:** Dictionary of bias check results.

---

### InstanceReweighting

Pre-processing transformer that assigns sample weights to mitigate bias.

```python
from pipeline_module import InstanceReweighting
```

#### Constructor

```python
InstanceReweighting(
    method: str = 'inverse_propensity',
    alpha: float = 1.0
)
```

**Parameters:**
- `method`: Reweighting method (`inverse_propensity`, `balanced`)
- `alpha`: Reweighting strength (0.0 = no reweighting, 1.0 = full reweighting)

#### Methods

##### fit()

```python
fit(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray
) -> InstanceReweighting
```

Learn reweighting scheme from training data.

##### transform()

```python
transform(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Apply reweighting and return (X, y, weights).

##### fit_transform()

```python
fit_transform(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
```

Fit and transform in one step.

**Example:**
```python
reweighter = InstanceReweighting(method='inverse_propensity', alpha=1.0)
X_train, y_train, weights = reweighter.fit_transform(
    X_train, y_train, sensitive_features=gender_train
)

# Train with weights
model.fit(X_train, y_train, sample_weight=weights)
```

---

## Training Module

### ReductionsWrapper

Wrapper for training models with fairness constraints using Fairlearn's reduction approach.

```python
from training_module import ReductionsWrapper
```

#### Constructor

```python
ReductionsWrapper(
    base_estimator,
    constraint: str = 'demographic_parity',
    eps: float = 0.05,
    **kwargs
)
```

**Parameters:**
- `base_estimator`: sklearn-compatible estimator
- `constraint`: Fairness constraint type
  - `demographic_parity`: Equal positive prediction rates
  - `equalized_odds`: Equal TPR and FPR
  - `equal_opportunity`: Equal TPR
  - `bounded_group_loss`: Maximum loss per group
- `eps`: Constraint violation tolerance
- `**kwargs`: Additional arguments passed to underlying reduction

#### Methods

##### fit()

```python
fit(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray
) -> ReductionsWrapper
```

Train model with fairness constraints.

**Example:**
```python
from sklearn.linear_model import LogisticRegression

model = ReductionsWrapper(
    base_estimator=LogisticRegression(max_iter=1000),
    constraint='equalized_odds',
    eps=0.01
)

model.fit(X_train, y_train, sensitive_features=gender_train)
predictions = model.predict(X_test)
```

##### predict()

```python
predict(X: np.ndarray) -> np.ndarray
```

Generate predictions.

##### predict_proba()

```python
predict_proba(X: np.ndarray) -> np.ndarray
```

Generate probability estimates (if supported by base estimator).

---

### GroupCalibrator

Post-processing calibration for group fairness.

```python
from training_module import GroupCalibrator
```

#### Constructor

```python
GroupCalibrator(
    method: str = 'isotonic',
    n_bins: int = 10
)
```

**Parameters:**
- `method`: Calibration method (`isotonic`, `sigmoid`)
- `n_bins`: Number of bins for calibration curves

#### Methods

##### fit()

```python
fit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_features: np.ndarray
) -> GroupCalibrator
```

Learn group-specific calibration.

##### transform()

```python
transform(
    y_pred_proba: np.ndarray,
    sensitive_features: np.ndarray
) -> np.ndarray
```

Apply calibration to probabilities.

**Example:**
```python
calibrator = GroupCalibrator(method='isotonic')
calibrator.fit(y_val, model.predict_proba(X_val), gender_val)

# Calibrate test predictions
calibrated_proba = calibrator.transform(
    model.predict_proba(X_test),
    gender_test
)
```

---

## Monitoring Module

### RealTimeFairnessTracker

Real-time fairness monitoring with sliding window.

```python
from monitoring_module import RealTimeFairnessTracker
```

#### Constructor

```python
RealTimeFairnessTracker(
    window_size: int = 1000,
    metrics: List[str] = None,
    thresholds: Dict[str, float] = None
)
```

**Parameters:**
- `window_size`: Sliding window size for metric computation
- `metrics`: List of fairness metrics to track
- `thresholds`: Alert thresholds per metric

#### Methods

##### add_batch()

```python
add_batch(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensitive_features: np.ndarray
) -> Dict[str, float]
```

Process new prediction batch and update metrics.

**Returns:** Dictionary of current metric values.

**Example:**
```python
tracker = RealTimeFairnessTracker(
    window_size=1000,
    metrics=['demographic_parity', 'equalized_odds'],
    thresholds={'demographic_parity': 0.1}
)

# Production loop
for batch in prediction_stream:
    metrics = tracker.add_batch(
        y_pred=batch['predictions'],
        y_true=batch['labels'],
        sensitive_features=batch['gender']
    )
    
    # Check for violations
    if metrics.get('demographic_parity', 0) > 0.1:
        send_alert("Fairness violation detected!")
```

##### get_current_metrics()

```python
get_current_metrics() -> Dict[str, float]
```

Get current fairness metric values.

##### get_alerts()

```python
get_alerts() -> List[Dict]
```

Retrieve triggered alerts.

##### reset()

```python
reset() -> None
```

Clear tracking history.

---

### DriftDetector

Statistical drift detection for model monitoring.

```python
from monitoring_module import DriftDetector
```

#### Constructor

```python
DriftDetector(
    alpha: float = 0.05,
    window_size: int = 1000
)
```

**Parameters:**
- `alpha`: Significance level for KS test
- `window_size`: Reference window size

#### Methods

##### fit()

```python
fit(reference_data: np.ndarray) -> DriftDetector
```

Establish reference distribution.

##### detect()

```python
detect(current_data: np.ndarray) -> DriftResult
```

Test for distribution shift.

**Returns:** `DriftResult` with:
- `drift_detected` (bool)
- `p_value` (float)
- `statistic` (float)

**Example:**
```python
detector = DriftDetector(alpha=0.05)
detector.fit(X_train)

# Check production data
result = detector.detect(X_production)
if result.drift_detected:
    print(f"Drift detected (p={result.p_value:.4f})")
```

---

### generate_monitoring_report()

Generate comprehensive monitoring report.

```python
from monitoring_module import generate_monitoring_report
```

#### Function Signature

```python
generate_monitoring_report(
    tracker: RealTimeFairnessTracker,
    output_path: str = 'reports/monitoring_report.md',
    include_plots: bool = True
) -> str
```

**Parameters:**
- `tracker`: Configured tracker instance
- `output_path`: Output file path
- `include_plots`: Include visualization plots

**Returns:** Path to generated report.

**Example:**
```python
report_path = generate_monitoring_report(
    tracker,
    output_path='reports/daily_fairness.md',
    include_plots=True
)
print(f"Report saved to {report_path}")
```

---

## Shared Utilities

### Logger

```python
from shared.logging import get_logger, PipelineLogger
```

#### get_logger()

```python
get_logger(name: str) -> logging.Logger
```

Get configured logger instance.

**Example:**
```python
logger = get_logger(__name__)
logger.info("Processing started")
```

#### PipelineLogger

Context manager for timing pipeline stages.

```python
with PipelineLogger(logger, "data_loading"):
    df = pd.read_csv('data.csv')
    # Automatically logs elapsed time
```

---

### Validation

```python
from shared.validation import validate_inputs, validate_config
```

#### validate_inputs()

```python
validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
    min_samples: int = 30
) -> None
```

Validate input arrays for consistency.

**Raises:** `ValueError` if validation fails.

#### validate_config()

```python
validate_config(config: Dict) -> None
```

Validate configuration dictionary structure.

---

## Orchestrator

### FairnessPipelineOrchestrator

End-to-end pipeline orchestration.

```python
from run_pipeline import FairnessPipelineOrchestrator
```

#### Constructor

```python
FairnessPipelineOrchestrator(config_path: str)
```

**Parameters:**
- `config_path`: Path to YAML configuration file

#### Methods

##### run()

```python
run(data_path: Optional[str] = None) -> Dict[str, Any]
```

Execute complete fairness pipeline.

**Parameters:**
- `data_path`: Optional override for data path in config

**Returns:** Dictionary containing:
- `baseline`: Baseline fairness measurements
- `bias_detection`: Bias detection results
- `validation`: Final validation metrics
- `tracker`: Configured monitoring tracker

**Example:**
```python
orchestrator = FairnessPipelineOrchestrator('config.yml')
results = orchestrator.run(data_path='data/my_data.csv')

# Access results
print("Baseline metrics:", results['baseline'])
print("Final accuracy:", results['validation']['accuracy'])
```

##### run_measurement()

```python
run_measurement(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray
) -> Dict[str, FairnessResult]
```

Execute measurement step only.

##### run_bias_detection()

```python
run_bias_detection(df: pd.DataFrame) -> Dict[str, BiasResult]
```

Execute bias detection step only.

##### run_mitigation()

```python
run_mitigation(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
```

Execute mitigation step only.

##### run_training()

```python
run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None
)
```

Execute training step only.

##### run_validation()

```python
run_validation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sensitive_test: np.ndarray
) -> Dict[str, Any]
```

Execute validation step only.

##### setup_monitoring()

```python
setup_monitoring() -> RealTimeFairnessTracker
```

Initialize monitoring tracker.

---

## Data Schemas

### FairnessResult

```python
@dataclass
class FairnessResult:
    value: float
    is_fair: bool
    confidence_interval: Tuple[float, float]
    effect_size: float
    group_metrics: Dict[str, float]
    metadata: Dict[str, Any]
```

### BiasResult

```python
@dataclass
class BiasResult:
    detected: bool
    severity: str  # 'low', 'medium', 'high'
    details: Dict[str, Any]
    metric_value: float
```

### DriftResult

```python
@dataclass
class DriftResult:
    drift_detected: bool
    p_value: float
    statistic: float
    timestamp: datetime
```

---

## Error Handling

All modules raise standard Python exceptions:

- `ValueError`: Invalid input parameters
- `FileNotFoundError`: Missing configuration or data files
- `RuntimeError`: Pipeline execution failures

**Example:**
```python
try:
    orchestrator = FairnessPipelineOrchestrator('config.yml')
    results = orchestrator.run()
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
except ValueError as e:
    print(f"Invalid configuration: {e}")
except RuntimeError as e:
    print(f"Pipeline failed: {e}")
```

---

## Type Hints

All functions include full type hints for IDE support:

```python
def compute_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric: str = 'demographic_parity',
    threshold: float = 0.1
) -> FairnessResult:
    ...
```

---

## See Also

- [Configuration Guide](configuration.md) - YAML configuration reference
- [Best Practices](best_practices.md) - Usage recommendations
- Module READMEs for detailed examples