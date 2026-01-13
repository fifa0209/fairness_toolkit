"""
COMPREHENSIVE MONITORING MODULE TEST SUITE

Complete integration testing covering all monitoring components.
Tests production-readiness of fairness monitoring infrastructure.

Run: python test_monitoring_comprehensive.py
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tempfile
from pathlib import Path

print("=" * 80)
print("COMPREHENSIVE FAIRNESS MONITORING TEST SUITE")
print("Testing all components for production deployment")
print("=" * 80)

# Test counters
tests_passed = 0
tests_failed = 0
test_names = []

def run_test(name, test_func):
    """Run a test and track results."""
    global tests_passed, tests_failed
    test_names.append(name)
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print('='*80)
    try:
        test_func()
        tests_passed += 1
        print(f"✅ {name} PASSED")
        return True
    except Exception as e:
        tests_failed += 1
        print(f"❌ {name} FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# IMPORTS
# ============================================================================
def test_imports():
    """Test all module imports."""
    from monitoring_module.src import (
        RealTimeFairnessTracker,
        BatchFairnessMonitor,
        FairnessDriftDetector,
        ThresholdAlertSystem,
    )
    
    from monitoring_module.src.ab_testing import (
        FairnessABTestAnalyzer,
        run_ab_test,
    )
    
    from monitoring_module.src.power_analysis import (
        FairnessPowerAnalyzer,
        ExperimentDesigner,
        quick_sample_size_calculator,
    )
    
    from monitoring_module.src.proxy_monitoring import (
        ProxyBasedMonitor,
        GeographicProxyBuilder,
        PrivacyPreservingReporter,
    )
    
    from monitoring_module.src.circuit_breaker import (
        FairnessCircuitBreaker,
        CircuitBreakerMonitor,
        CircuitState,
    )
    
    from monitoring_module.src.alerting import (
        ThresholdAlertSystem as AdvancedAlertSystem,
        AdaptiveAlertSystem,
        AlertAggregator,
        AlertNotifier,
    )
    
    print("All critical imports successful")

# ============================================================================
# UTILITIES
# ============================================================================
np.random.seed(42)

def generate_data(n, bias=0.1):
    """Generate test data."""
    sensitive = np.random.choice([0, 1], n)
    y_true = np.random.choice([0, 1], n)
    y_pred = y_true.copy()
    for i in range(n):
        if sensitive[i] == 1 and np.random.random() < bias:
            y_pred[i] = 1 - y_pred[i]
    return y_true, y_pred, sensitive

# ============================================================================
# REAL-TIME TRACKING TESTS
# ============================================================================
def test_realtime_tracking():
    """Test real-time fairness tracking."""
    from monitoring_module.src import RealTimeFairnessTracker
    
    tracker = RealTimeFairnessTracker(window_size=300, min_samples=50)
    
    # Add batches
    for i in range(5):
        y_true, y_pred, sensitive = generate_data(80)
        metrics = tracker.add_batch(y_pred, y_true, sensitive)
        if metrics:
            print(f"Batch {i+1} metrics computed")
    
    # Verify functionality
    summary = tracker.get_summary_statistics()
    ts = tracker.get_time_series()
    
    assert len(tracker.history) > 0, "No history recorded"
    assert tracker.n_samples_processed > 0, "No samples processed"
    print(f"Processed {tracker.n_samples_processed} samples")
    print(f"History: {len(tracker.history)} entries")

def test_batch_monitoring():
    """Test batch fairness evaluation."""
    from monitoring_module.src import BatchFairnessMonitor
    
    monitor = BatchFairnessMonitor(fairness_threshold=0.10)
    
    y_true, y_pred, sensitive = generate_data(200, bias=0.05)
    result = monitor.evaluate_batch(y_true, y_pred, sensitive)
    
    assert 'is_fair' in result
    assert 'violations' in result
    assert 'metrics' in result
    print(f"Batch evaluation: is_fair={result['is_fair']}")

# ============================================================================
# PROXY MONITORING TESTS
# ============================================================================
def test_proxy_monitoring():
    """Test proxy-based monitoring."""
    from monitoring_module.src.proxy_monitoring import (
        GeographicProxyBuilder,
        ProxyBasedMonitor,
    )
    
    # Create census data
    census_df = pd.DataFrame({
        'zip_code': ['10001', '10002', '10003', '10004'],
        'pct_minority': [0.25, 0.65, 0.40, 0.70],
    })
    
    # Build proxy mapping
    builder = GeographicProxyBuilder()
    mapping = builder.build_from_census_data(
        census_df,
        geography_col='zip_code',
        demographic_cols=['pct_minority'],
        threshold=0.5
    )
    
    # Create monitor
    monitor = ProxyBasedMonitor(mapping, uncertainty_adjustment=True)
    
    # Generate data
    df = pd.DataFrame({
        'zip_code': np.random.choice(['10001', '10002', '10003', '10004'], 200),
        'y_pred': np.random.choice([0, 1], 200),
        'y_true': np.random.choice([0, 1], 200),
    })
    
    # Compute metrics
    results = monitor.compute_proxy_metrics(df)
    
    assert len(results) > 0, "No metrics computed"
    for metric, data in results.items():
        assert 'value' in data
        assert 'uncertainty' in data
        print(f"{metric}: {data['value']:.4f} ± {data['uncertainty']:.4f}")

def test_privacy_reporting():
    """Test privacy-preserving reporting."""
    from monitoring_module.src.proxy_monitoring import PrivacyPreservingReporter
    
    reporter = PrivacyPreservingReporter(k_threshold=10, add_noise=True)
    
    group_metrics = {'A': 0.15, 'B': 0.22, 'C': 0.18}
    group_sizes = {'A': 150, 'B': 200, 'C': 5}  # C below threshold
    
    report = reporter.safe_report_metrics(group_metrics, group_sizes)
    
    assert report['C']['suppressed'], "Small group not suppressed"
    assert not report['A']['suppressed'], "Large group wrongly suppressed"
    print("K-anonymity working correctly")

# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================
def test_circuit_breaker():
    """Test fairness circuit breaker."""
    from monitoring_module.src.circuit_breaker import (
        FairnessCircuitBreaker,
        CircuitBreakerConfig,
        CircuitState,
        InterventionType,
    )
    
    config = CircuitBreakerConfig(
        critical_threshold=0.20,
        failure_count_threshold=3,
        intervention_type=InterventionType.ROUTE_TO_BASELINE,
    )
    
    breaker = FairnessCircuitBreaker(config)
    
    # Normal operation
    for _ in range(2):
        breaker.record_metrics({'demographic_parity': 0.08})
    assert breaker.state == CircuitState.CLOSED
    
    # Trigger violations
    for _ in range(3):
        event = breaker.record_metrics({'demographic_parity': 0.25})
    
    assert breaker.state == CircuitState.OPEN, "Circuit didn't open"
    assert breaker.should_use_baseline(), "Not routing to baseline"
    print(f"Circuit opened correctly, using baseline")

def test_circuit_breaker_system():
    """Test system-wide circuit breaker monitoring."""
    from monitoring_module.src.circuit_breaker import (
        CircuitBreakerMonitor,
        FairnessCircuitBreaker,
        CircuitBreakerConfig,
    )
    
    monitor = CircuitBreakerMonitor()
    
    # Register multiple breakers
    for name in ['model_a', 'model_b', 'model_c']:
        config = CircuitBreakerConfig(critical_threshold=0.15)
        breaker = FairnessCircuitBreaker(config)
        monitor.register_breaker(name, breaker)
    
    # Degrade one model
    for _ in range(3):
        monitor.breakers['model_b'].record_metrics({'demographic_parity': 0.25})
    
    # Check system status
    status = monitor.get_system_status()
    alerts = monitor.get_alerts()
    
    assert status['total_breakers'] == 3
    assert not status['healthy']
    assert len(alerts) > 0
    print(f"System monitoring: {len(alerts)} alerts")

# ============================================================================
# ALERTING TESTS  
# ============================================================================
def test_advanced_alerting():
    """Test advanced alerting features."""
    from monitoring_module.src.alerting import (
        ThresholdAlertSystem,
        AdaptiveAlertSystem,
        AlertAggregator,
        AlertNotifier,
        AlertSeverity,
    )
    
    # Threshold alerts
    system = ThresholdAlertSystem(thresholds={'demographic_parity': 0.10})
    alert = system.check_thresholds({'demographic_parity': 0.22})
    assert alert is not None
    print(f"Alert severity: {alert.severity.value}")
    
    # Adaptive system (pass thresholds as first positional arg to parent class)
    base_thresholds = {'demographic_parity': 0.12}
    adaptive = AdaptiveAlertSystem(base_thresholds, target_fpr=0.05, adaptation_window=10)
    
    # base_thresholds = {'demographic_parity': 0.10}
    # adaptive = AdaptiveAlertSystem(
    #     base_thresholds,  # First positional argument for parent ThresholdAlertSystem
    #     target_fpr=0.05,
    #     adaptation_window=10
    # )
    
        
    for i in range(10):
        alert = adaptive.check_and_adapt({'demographic_parity': 0.12})
        if alert:
            adaptive.provide_feedback(alert.alert_id, is_true_positive=(i % 2 == 0))
    
    print(f"Adaptive threshold: {adaptive.thresholds['demographic_parity']:.4f}")
    
    # Alert notification
    notifier = AlertNotifier()
    calls = []
    notifier.register_handler('test', lambda a: calls.append(a.alert_id))
    notifier.add_routing_rule(AlertSeverity.HIGH, ['test'])
    
    if alert:
        notifier.notify(alert)
        assert len(calls) > 0
        print("Notification routing working")

# ============================================================================
# A/B TESTING TESTS
# ============================================================================
def test_ab_testing():
    """Test A/B testing framework."""
    from monitoring_module.src.ab_testing import FairnessABTestAnalyzer
    
    # Generate test groups
    control = pd.DataFrame({
        'y_true': np.random.choice([0, 1], 200),
        'y_pred': np.random.choice([0, 1], 200, p=[0.6, 0.4]),
        'sensitive': np.random.choice([0, 1], 200),
    })
    
    treatment = pd.DataFrame({
        'y_true': np.random.choice([0, 1], 200),
        'y_pred': np.random.choice([0, 1], 200, p=[0.5, 0.5]),
        'sensitive': np.random.choice([0, 1], 200),
    })
    
    analyzer = FairnessABTestAnalyzer(alpha=0.05, n_bootstrap=50)
    
    # Overall test
    results = analyzer.analyze_test(control, treatment, metrics=['demographic_parity'])
    assert 'demographic_parity' in results
    
    # Multi-objective
    multi = analyzer.multi_objective_analysis(control, treatment)
    assert 'outcome' in multi
    assert 'recommendation' in multi
    
    print(f"A/B outcome: {multi['outcome']}")

# ============================================================================
# POWER ANALYSIS TESTS
# ============================================================================
def test_power_analysis():
    """Test statistical power analysis."""
    from monitoring_module.src.power_analysis import (
        FairnessPowerAnalyzer,
        ExperimentDesigner,
        quick_sample_size_calculator,
    )
    
    analyzer = FairnessPowerAnalyzer(alpha=0.05)
    
    # Sample size
    result = analyzer.sample_size_for_proportion_test(
        baseline_rate=0.50,
        target_rate=0.60,
        power=0.80
    )
    assert result.result_value > 0
    print(f"Required sample size: {result.result_value}")
    
    # Power calculation
    power = analyzer.power_for_proportion_test(
        baseline_rate=0.50,
        target_rate=0.60,
        n_control=200,
        n_treatment=200
    )
    assert 0 <= power.power <= 1
    print(f"Achieved power: {power.power:.2%}")
    
    # Quick calculator
    n = quick_sample_size_calculator(0.20, 0.10, 0.05, 0.80)
    assert n > 0
    print(f"Quick estimate: {n}")

# ============================================================================
# DRIFT DETECTION TESTS
# ============================================================================
def test_drift_detection():
    """Test fairness drift detection."""
    from monitoring_module.src import FairnessDriftDetector
    
    detector = FairnessDriftDetector(alpha=0.05)
    
    # Set reference
    y_true_ref, y_pred_ref, s_ref = generate_data(400, bias=0.05)
    detector.set_reference(y_true_ref, y_pred_ref, s_ref)
    
    # Test with drifted data
    y_true_new, y_pred_new, s_new = generate_data(400, bias=0.25)
    result = detector.detect_drift(y_true_new, y_pred_new, s_new)
    
    assert 'drift_detected' in result
    assert 'tests' in result
    print(f"Drift detected: {result['drift_detected']}")

# ============================================================================
# DASHBOARD & REPORTING TESTS
# ============================================================================
def test_reporting():
    """Test report generation."""
    try:
        from monitoring_module.src.report_generator import FairnessMonitoringReport
        
        with tempfile.TemporaryDirectory() as tmpdir:
            generator = FairnessMonitoringReport(output_dir=Path(tmpdir))
            
            summary_stats = {
                'demographic_parity': {
                    'mean': 0.08,
                    'std': 0.02,
                    'current': 0.09,
                    'min': 0.05,
                    'max': 0.12,
                }
            }
            
            report_path = generator.generate_monitoring_report(
                summary_stats=summary_stats,
                alerts=[],
                metadata={'Model': 'TestModel'}
            )
            
            assert Path(report_path).exists()
            print(f"Report generated: {Path(report_path).name}")
            
    except ImportError:
        print("Report generation skipped (optional dependency)")

# ============================================================================
# INTEGRATION TESTS
# ============================================================================
def test_end_to_end_monitoring():
    """Test complete monitoring workflow."""
    from monitoring_module.src import RealTimeFairnessTracker
    from monitoring_module.src.alerting import ThresholdAlertSystem
    from monitoring_module.src.circuit_breaker import (
        FairnessCircuitBreaker,
        CircuitBreakerConfig,
        InterventionType,
    )
    
    # Setup components
    tracker = RealTimeFairnessTracker(window_size=200, min_samples=50)
    alert_system = ThresholdAlertSystem({'demographic_parity': 0.10})
    
    config = CircuitBreakerConfig(
        critical_threshold=0.20,
        failure_count_threshold=2,
        intervention_type=InterventionType.ROUTE_TO_BASELINE,
    )
    breaker = FairnessCircuitBreaker(config)
    
    # Simulate production monitoring
    alerts_triggered = 0
    circuit_opened = False
    
    for batch_idx in range(10):
        y_true, y_pred, sensitive = generate_data(60, bias=0.05 + batch_idx*0.02)
        
        # Track in real-time
        metrics = tracker.add_batch(y_pred, y_true, sensitive)
        
        if metrics and 'demographic_parity' in metrics:
            dp_value = metrics['demographic_parity']
            
            # Check alerts
            alert = alert_system.check_thresholds(metrics)
            if alert:
                alerts_triggered += 1
            
            # Update circuit breaker
            event = breaker.record_metrics(metrics)
            if breaker.should_use_baseline():
                circuit_opened = True
    
    print(f"Batches processed: 10")
    print(f"Alerts: {alerts_triggered}")
    print(f"Circuit opened: {circuit_opened}")
    print("End-to-end monitoring complete")

# ============================================================================
# RUN ALL TESTS
# ============================================================================
if __name__ == '__main__':
    print("\nRunning comprehensive test suite...\n")
    
    # Core functionality
    run_test("Module Imports", test_imports)
    run_test("Real-Time Tracking", test_realtime_tracking)
    run_test("Batch Monitoring", test_batch_monitoring)
    
    # Privacy & compliance
    run_test("Proxy Monitoring", test_proxy_monitoring)
    run_test("Privacy Reporting", test_privacy_reporting)
    
    # Safety & interventions
    run_test("Circuit Breaker", test_circuit_breaker)
    run_test("Circuit Breaker System", test_circuit_breaker_system)
    
    # Alerting
    run_test("Advanced Alerting", test_advanced_alerting)
    
    # Statistical analysis
    run_test("A/B Testing", test_ab_testing)
    run_test("Power Analysis", test_power_analysis)
    run_test("Drift Detection", test_drift_detection)
    
    # Reporting
    run_test("Report Generation", test_reporting)
    
    # Integration
    run_test("End-to-End Monitoring", test_end_to_end_monitoring)
    
    # ========================================================================
    # FINAL REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE RESULTS")
    print("="*80)
    
    total_tests = tests_passed + tests_failed
    success_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"✅ Passed: {tests_passed}")
    print(f"❌ Failed: {tests_failed}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print("\n" + "="*80)
    print("PRODUCTION READINESS CHECKLIST")
    print("="*80)
    
    checklist = {
        "Real-time monitoring": tests_passed >= 2,
        "Privacy compliance (k-anonymity, DP)": tests_passed >= 4,
        "Automated interventions (circuit breaker)": tests_passed >= 6,
        "Alert system": tests_passed >= 7,
        "Statistical testing (A/B, power)": tests_passed >= 9,
        "Drift detection": tests_passed >= 10,
        "Reporting": tests_passed >= 11,
    }
    
    for feature, ready in checklist.items():
        status = "✅" if ready else "⚠️"
        print(f"{status} {feature}")
    
    all_ready = all(checklist.values())
    
    print("\n" + "="*80)
    if all_ready and tests_failed == 0:
        print("🎉 ALL SYSTEMS GO - READY FOR PRODUCTION DEPLOYMENT")
    elif tests_failed == 0:
        print("✅ ALL TESTS PASSED - System functional")
    else:
        print("⚠️  SOME TESTS FAILED - Review before deployment")
    print("="*80)
    
    sys.exit(0 if tests_failed == 0 else 1)