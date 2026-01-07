# Monitoring Module

**Production fairness tracking, drift detection, and alerting**

## Overview

The Monitoring Module ensures fairness doesn't degrade in production through:

- ✅ **RealTimeFairnessTracker** - Sliding window metrics for streaming predictions
- ✅ **FairnessDriftDetector** - Statistical drift detection (KS test)
- ✅ **ThresholdAlertSystem** - Simple threshold-based alerts
- ✅ **Dashboard** - Interactive visualizations (Plotly)

## Quick Start

```python
from monitoring_module import RealTimeFairnessTracker

# Initialize tracker
tracker = RealTimeFairnessTracker(window_size=1000)

# Process streaming predictions
for batch in prediction_stream:
    metrics = tracker.add_batch(
        batch['y_pred'],
        batch['y_true'],
        batch['sensitive']
    )
    
    # Check for violations
    if metrics.get('demographic_parity', 0) > 0.1:
        send_alert("Fairness violation detected!")
```

## Components

### 1. Real-Time Tracker

**RealTimeFairnessTracker** maintains sliding windows of recent predictions and computes fairness metrics continuously.

#### Features
- Sliding window computation (configurable size)
- Time-series storage
- Summary statistics
- Export to CSV

#### Usage

```python
from monitoring_module import RealTimeFairnessTracker

tracker = RealTimeFairnessTracker(
    window_size=1000,  # Number of samples in window
    metrics=['demographic_parity', 'equalized_odds'],
    min_samples=100     # Minimum before computing
)

# Add predictions in batches
metrics = tracker.add_batch(y_pred, y_true, sensitive_features)

# Get current state
current = tracker.get_current_metrics()
print(f"Current DP: {current['demographic_parity']:.3f}")

# Get time series for analysis
time_series = tracker.get_time_series()

# Get summary statistics
summary = tracker.get_summary_statistics()
print(f"Mean DP: {summary['demographic_parity']['mean']:.3f}")

# Export history
tracker.export_history('fairness_history.csv')
```

**When to use**: Streaming predictions (online learning, real-time scoring)

### 2. Drift Detection

**FairnessDriftDetector** uses statistical tests to identify when fairness degrades compared to a reference period.

#### How it Works
1. Set reference period (e.g., first week of production)
2. Compare recent predictions to reference using KS test
3. Alert if distribution has changed significantly

#### Usage

```python
from monitoring_module import FairnessDriftDetector

detector = FairnessDriftDetector(
    alpha=0.05,  # Significance level
    test_method='ks'  # Kolmogorov-Smirnov test
)

# Set reference period
detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)

# Check for drift periodically
drift_result = detector.detect_drift(
    y_true_recent, y_pred_recent, sensitive_recent
)

if drift_result['drift_detected']:
    print(f"Drift in: {drift_result['drifted_metrics']}")
    
    # Create alert
    alert = detector.create_alert(drift_result, severity='HIGH')
    send_to_monitoring_system(alert)
```

**When to use**: Detect gradual fairness degradation over days/weeks

### 3. Threshold Alerts

**ThresholdAlertSystem** triggers alerts when metrics exceed absolute thresholds.

#### Features
- Configurable thresholds per metric
- Severity levels (LOW, HIGH, CRITICAL)
- Simple and fast

#### Usage

```python
from monitoring_module import ThresholdAlertSystem

alerter = ThresholdAlertSystem(
    thresholds={
        'demographic_parity': 0.1,
        'equalized_odds': 0.1
    },
    severity_levels={
        'LOW': 1.0,      # At threshold
        'HIGH': 1.5,     # 50% over
        'CRITICAL': 2.0  # 100% over
    }
)

# Check current metrics
alert = alerter.check_thresholds({
    'demographic_parity': 0.15,
    'equalized_odds': 0.08
})

if alert:
    print(f"Severity: {alert.severity}")
    print(f"Message: {alert.message}")
```

**When to use**: Simple absolute fairness requirements

### 4. Dashboard

**FairnessMonitoringDashboard** creates interactive Plotly visualizations.

#### Features
- Time series plots
- Per-group comparisons
- Alert timeline
- HTML export

#### Usage

```python
from monitoring_module import FairnessMonitoringDashboard

dashboard = FairnessMonitoringDashboard()

# Plot metrics over time
fig = dashboard.plot_metrics_over_time(
    time_series_df,
    metrics=['demographic_parity', 'equalized_odds'],
    threshold=0.1
)
fig.write_html('fairness_dashboard.html')

# Plot group comparison
fig2 = dashboard.plot_group_comparison({
    'Group_0': {'accuracy': 0.85, 'precision': 0.82},
    'Group_1': {'accuracy': 0.87, 'precision': 0.84}
})
fig2.show()

# Create comprehensive dashboard
dashboard.create_dashboard(
    time_series=time_series_df,
    group_metrics=group_metrics_dict,
    alerts=alerts_list,
    output_path='dashboard.html'
)
```

## Complete Example

### End-to-End Monitoring Pipeline

```python
from monitoring_module import (
    RealTimeFairnessTracker,
    FairnessDriftDetector,
    ThresholdAlertSystem,
    FairnessMonitoringDashboard
)

# 1. Initialize components
tracker = RealTimeFairnessTracker(window_size=1000)
drift_detector = FairnessDriftDetector(alpha=0.05)
alerter = ThresholdAlertSystem()
dashboard = FairnessMonitoringDashboard()

# 2. Set reference period (first week)
drift_detector.set_reference(
    y_true_week1, y_pred_week1, sensitive_week1
)

# 3. Monitor production predictions
alerts = []

for day in range(30):  # Monitor for 30 days
    # Get day's predictions
    y_true, y_pred, sensitive = get_daily_predictions(day)
    
    # Track real-time metrics
    metrics = tracker.add_batch(y_pred, y_true, sensitive)
    
    # Check thresholds
    threshold_alert = alerter.check_thresholds(metrics)
    if threshold_alert:
        alerts.append(threshold_alert)
        send_alert(threshold_alert)
    
    # Check drift (weekly)
    if day % 7 == 0:
        drift_result = drift_detector.detect_drift(y_true, y_pred, sensitive)
        if drift_result['drift_detected']:
            drift_alert = drift_detector.create_alert(drift_result)
            alerts.append(drift_alert)
            send_alert(drift_alert)

# 4. Generate final dashboard
time_series = tracker.get_time_series()
summary = tracker.get_summary_statistics()

dashboard.create_dashboard(
    time_series=time_series,
    group_metrics=get_latest_group_metrics(),
    alerts=[a.to_dict() for a in alerts],
    output_path='production_dashboard.html'
)
```

## Integration with Other Modules

### With Measurement Module

```python
from measurement_module import FairnessAnalyzer
from monitoring_module import RealTimeFairnessTracker

analyzer = FairnessAnalyzer()
tracker = RealTimeFairnessTracker()

# Detailed analysis on-demand
for batch in batches:
    # Quick monitoring
    tracker.add_batch(batch['y_pred'], batch['y_true'], batch['sensitive'])
    
    # Detailed analysis when needed
    if tracker.get_current_metrics()['demographic_parity'] > 0.1:
        result = analyzer.compute_metric(
            batch['y_true'], batch['y_pred'], batch['sensitive'],
            metric='demographic_parity',
            compute_ci=True  # Get confidence interval
        )
        print(f"Detailed: {result.value:.3f}, CI: {result.confidence_interval}")
```

### With Training Module

```python
from training_module import ReductionsWrapper
from monitoring_module import RealTimeFairnessTracker

# Train model
model = ReductionsWrapper(...)
model.fit(X_train, y_train, sensitive_features=s_train)

# Monitor in production
tracker = RealTimeFairnessTracker()

for batch in production_batches:
    y_pred = model.predict(batch['X'])
    tracker.add_batch(y_pred, batch['y_true'], batch['sensitive'])
    
    # Retrain if fairness degrades
    if tracker.get_current_metrics()['demographic_parity'] > 0.15:
        model.fit(X_recent, y_recent, sensitive_features=s_recent)
```

## Reporting

### Generate Markdown Report

```python
from monitoring_module import generate_monitoring_report

summary = tracker.get_summary_statistics()
alerts = [alert.to_dict() for alert in alert_list]

generate_monitoring_report(
    summary_stats=summary,
    alerts=alerts,
    output_path='weekly_report.md'
)
```

**Sample Output**:
```markdown
# Fairness Monitoring Report

**Generated:** 2024-12-30 14:30:00

## Summary Statistics

### Demographic Parity
- **Current:** 0.0850
- **Mean:** 0.0920
- **Std:** 0.0150
- **Range:** [0.0650, 0.1200]

## Alerts (3 total)

### 🟠 HIGH - threshold_violation
- **Time:** 2024-12-25 10:15:00
- **Metric:** demographic_parity
- **Message:** Threshold violations detected: demographic_parity=0.15 (threshold=0.10)
```

## API Reference

### RealTimeFairnessTracker

```python
RealTimeFairnessTracker(
    window_size=1000,
    metrics=['demographic_parity', 'equalized_odds'],
    min_samples=100
)
```

**Methods**:
- `add_batch(y_pred, y_true, sensitive_features, timestamps=None)` → dict
- `get_current_metrics()` → dict
- `get_time_series(metric=None, start_time=None, end_time=None)` → DataFrame
- `get_summary_statistics()` → dict
- `export_history(filepath)` → None
- `reset()` → None

### FairnessDriftDetector

```python
FairnessDriftDetector(
    alpha=0.05,
    test_method='ks',
    min_samples=100
)
```

**Methods**:
- `set_reference(y_true, y_pred, sensitive_features)` → None
- `detect_drift(y_true, y_pred, sensitive_features)` → dict
- `create_alert(drift_result, severity='HIGH')` → MonitoringAlert

### ThresholdAlertSystem

```python
ThresholdAlertSystem(
    thresholds={'demographic_parity': 0.1},
    severity_levels={'LOW': 1.0, 'HIGH': 1.5, 'CRITICAL': 2.0}
)
```

**Methods**:
- `check_thresholds(metrics, group_sizes=None)` → MonitoringAlert or None

## Testing

```bash
python test_monitoring_module.py
```

**Expected Output**:
```
============================================================
Testing Monitoring Module
============================================================

[1/5] Testing imports...
✅ Core imports successful

[2/5] Generating streaming data...
✅ Data generator ready

[3/5] Testing RealTimeFairnessTracker...
✅ Tracker initialized
   Collected 10 time points

[4/5] Testing FairnessDriftDetector...
✅ Drift detector initialized
   Test 2 (high bias):
   Drift detected: True

[5/5] Testing ThresholdAlertSystem...
✅ Alert system initialized
   Alert: Severity: HIGH

============================================================
✅ MONITORING MODULE TESTS COMPLETED!
============================================================
```

## Limitations (48-Hour Scope)

**Implemented**:
- ✅ Sliding window tracking
- ✅ KS test for drift detection
- ✅ Threshold-based alerts
- ✅ Plotly dashboards

**Not Implemented** (documented for future):
- ❌ Wavelet decomposition (multi-scale drift)
- ❌ Adaptive thresholds
- ❌ Causal analysis of drift
- ❌ Live Dash/Streamlit dashboard

## Dependencies

```bash
pip install pandas numpy scipy plotly
```

## Next Steps

All 4 core modules complete! Now:
1. **Integration** → Orchestration script (`run_pipeline.py`)
2. **Demo Notebook** → End-to-end walkthrough
3. **Documentation** → Final polish

---

**Questions?** See test file for examples or main README.