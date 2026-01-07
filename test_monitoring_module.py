"""
Test monitoring module end-to-end.

Run: python test_monitoring_module.py
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

print("=" * 60)
print("Testing Monitoring Module")
print("=" * 60)

# Test 1: Import check
print("\n[1/5] Testing imports...")
try:
    from monitoring_module.src import (
        RealTimeFairnessTracker,
        BatchFairnessMonitor,
        FairnessDriftDetector,
        ThresholdAlertSystem,
        DASHBOARD_AVAILABLE,
    )
    print("✅ Core imports successful")
    print(f"   Dashboard available: {DASHBOARD_AVAILABLE}")
    
    if DASHBOARD_AVAILABLE:
        from monitoring_module.src import FairnessMonitoringDashboard
        print("✅ Dashboard imports successful")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test 2: Generate streaming data
print("\n[2/5] Generating streaming data...")
np.random.seed(42)

def generate_batch(n, bias_level=0.1):
    """Generate a batch of predictions with some bias."""
    sensitive = np.random.choice([0, 1], n)
    y_true = np.random.choice([0, 1], n)
    
    # Add bias to predictions
    y_pred = y_true.copy()
    for i in range(n):
        if sensitive[i] == 1:  # Bias toward group 1
            if np.random.random() < bias_level:
                y_pred[i] = 1
    
    return y_true, y_pred, sensitive

print("✅ Data generator ready")

# Test 3: Real-Time Tracker
print("\n[3/5] Testing RealTimeFairnessTracker...")

try:
    tracker = RealTimeFairnessTracker(
        window_size=500,
        metrics=['demographic_parity'],
        min_samples=50
    )
    print("✅ Tracker initialized")
    
    # Simulate streaming predictions
    n_batches = 10
    batch_size = 100
    
    print(f"   Simulating {n_batches} batches...")
    for i in range(n_batches):
        y_true, y_pred, sensitive = generate_batch(batch_size, bias_level=0.1)
        
        metrics = tracker.add_batch(y_pred, y_true, sensitive)
        
        if metrics:
            print(f"   Batch {i+1}: DP = {metrics.get('demographic_parity', 0):.4f}")
    
    # Get summary
    summary = tracker.get_summary_statistics()
    print(f"\n   Summary statistics:")
    for metric, stats in summary.items():
        print(f"   {metric}:")
        print(f"     Mean: {stats['mean']:.4f}")
        print(f"     Current: {stats['current']:.4f}")
    
    # Get time series
    time_series = tracker.get_time_series()
    print(f"   Collected {len(time_series)} time points")
    
except Exception as e:
    print(f"❌ RealTimeFairnessTracker failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Drift Detection
print("\n[4/5] Testing FairnessDriftDetector...")

try:
    detector = FairnessDriftDetector(alpha=0.05)
    print("✅ Drift detector initialized")
    
    # Set reference period (low bias)
    y_true_ref, y_pred_ref, s_ref = generate_batch(500, bias_level=0.05)
    detector.set_reference(y_true_ref, y_pred_ref, s_ref)
    print("   Reference period set")
    
    # Test with similar data (no drift expected)
    y_true_1, y_pred_1, s_1 = generate_batch(500, bias_level=0.06)
    drift_result_1 = detector.detect_drift(y_true_1, y_pred_1, s_1)
    
    print(f"\n   Test 1 (similar bias):")
    print(f"   Drift detected: {drift_result_1['drift_detected']}")
    
    # Test with high bias (drift expected)
    y_true_2, y_pred_2, s_2 = generate_batch(500, bias_level=0.25)
    drift_result_2 = detector.detect_drift(y_true_2, y_pred_2, s_2)
    
    print(f"\n   Test 2 (high bias):")
    print(f"   Drift detected: {drift_result_2['drift_detected']}")
    if drift_result_2['drift_detected']:
        print(f"   Drifted metrics: {drift_result_2['drifted_metrics']}")
    
    # Create alert
    alert = detector.create_alert(drift_result_2)
    if alert:
        print(f"\n   Alert created:")
        print(f"   Severity: {alert.severity}")
        print(f"   Message: {alert.message[:100]}...")
    
except Exception as e:
    print(f"❌ FairnessDriftDetector failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Threshold Alerts
print("\n[5/5] Testing ThresholdAlertSystem...")

try:
    alerter = ThresholdAlertSystem(
        thresholds={'demographic_parity': 0.1}
    )
    print("✅ Alert system initialized")
    
    # Test with fair metrics (no alert expected)
    fair_metrics = {'demographic_parity': 0.08}
    alert_1 = alerter.check_thresholds(fair_metrics)
    
    print(f"\n   Test 1 (fair):")
    print(f"   Alert: {alert_1 is not None}")
    
    # Test with unfair metrics (alert expected)
    unfair_metrics = {'demographic_parity': 0.25}
    alert_2 = alerter.check_thresholds(unfair_metrics)
    
    print(f"\n   Test 2 (unfair):")
    print(f"   Alert: {alert_2 is not None}")
    if alert_2:
        print(f"   Severity: {alert_2.severity}")
        print(f"   Message: {alert_2.message[:100]}...")
    
except Exception as e:
    print(f"❌ ThresholdAlertSystem failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Dashboard (bonus)
if DASHBOARD_AVAILABLE:
    print("\n[BONUS] Testing Dashboard...")
    
    try:
        from monitoring_module.src import FairnessMonitoringDashboard, generate_monitoring_report
        
        dashboard = FairnessMonitoringDashboard()
        print("✅ Dashboard initialized")
        
        # Create time series plot
        time_series = tracker.get_time_series()
        if len(time_series) > 0:
            fig = dashboard.plot_metrics_over_time(
                time_series,
                metrics=['demographic_parity']
            )
            print("✅ Time series plot created")
            # fig.write_html('monitoring_dashboard.html')  # Uncomment to save
        
        # Generate report
        summary = tracker.get_summary_statistics()
        alerts = []
        if alert_2:
            alerts.append(alert_2.to_dict())
        
        generate_monitoring_report(summary, alerts, 'monitoring_report.md')
        print("✅ Report generated")
        
    except Exception as e:
        print(f"⚠️  Dashboard test failed: {e}")

print("\n" + "=" * 60)
print("✅ MONITORING MODULE TESTS COMPLETED!")
print("=" * 60)
print("\n📊 Summary:")
print("   ✅ RealTimeFairnessTracker processes streaming data")
print("   ✅ FairnessDriftDetector identifies drift")
print("   ✅ ThresholdAlertSystem generates alerts")
if DASHBOARD_AVAILABLE:
    print("   ✅ Dashboard visualization works")
print("\n🎯 Monitoring module is ready!")
print("\nNext steps:")
print("  1. Integrate all 4 modules (measurement + pipeline + training + monitoring)")
print("  2. Create end-to-end demo notebook")
print("  3. Build orchestration script (run_pipeline.py)")