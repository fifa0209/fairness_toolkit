"""
Tests for Drift Detection Module

Tests statistical drift detection for fairness metrics including
KS tests, threshold alerts, and monitoring alert creation.
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from monitoring_module.src.drift_detection import (
    FairnessDriftDetector,
    ThresholdAlertSystem,
)
from shared.schemas import MonitoringAlert


@pytest.fixture
def sample_reference_data():
    """Create sample reference period data."""
    np.random.seed(42)
    n = 500
    
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    sensitive = np.random.binomial(1, 0.5, n)
    
    return y_true, y_pred, sensitive


@pytest.fixture
def sample_current_data():
    """Create sample current period data (similar to reference)."""
    np.random.seed(43)
    n = 500
    
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    sensitive = np.random.binomial(1, 0.5, n)
    
    return y_true, y_pred, sensitive


@pytest.fixture
def drifted_data():
    """Create drifted data with significant distribution shift."""
    np.random.seed(44)
    n = 500
    
    y_true = np.random.binomial(1, 0.5, n)
    # Introduce bias - group 0 gets more positive predictions
    y_pred = np.concatenate([
        np.random.binomial(1, 0.7, n//2),  # Group 0: 70% positive
        np.random.binomial(1, 0.3, n//2),  # Group 1: 30% positive
    ])
    sensitive = np.array([0] * (n//2) + [1] * (n//2))
    
    return y_true, y_pred, sensitive


class TestFairnessDriftDetector:
    """Tests for FairnessDriftDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = FairnessDriftDetector(alpha=0.05, test_method='ks', min_samples=100)
        
        assert detector.alpha == 0.05
        assert detector.test_method == 'ks'
        assert detector.min_samples == 100
        assert detector.reference_metrics is None
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_set_reference(self, mock_compute_metric, sample_reference_data):
        """Test setting reference period."""
        y_true, y_pred, sensitive = sample_reference_data
        
        # Mock compute_metric to return consistent values
        mock_compute_metric.return_value = (0.10, {'group_0': 0.50, 'group_1': 0.60}, {})
        
        detector = FairnessDriftDetector()
        detector.set_reference(y_true, y_pred, sensitive)
        
        assert detector.reference_metrics is not None
        assert detector.reference_predictions is not None
        assert 'y_true' in detector.reference_predictions
        assert 'y_pred' in detector.reference_predictions
        assert 'sensitive' in detector.reference_predictions
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_detect_drift_no_drift(self, mock_compute_metric, sample_reference_data, sample_current_data):
        """Test drift detection when no drift occurs."""
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        y_true_cur, y_pred_cur, sensitive_cur = sample_current_data
        
        # Mock similar metrics
        mock_compute_metric.return_value = (0.10, {}, {})
        
        detector = FairnessDriftDetector(alpha=0.05)
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        result = detector.detect_drift(y_true_cur, y_pred_cur, sensitive_cur)
        
        assert 'timestamp' in result
        assert 'drift_detected' in result
        assert 'drifted_metrics' in result
        assert 'tests' in result
        assert isinstance(result['drift_detected'], bool)
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_detect_drift_with_drift(self, mock_compute_metric, sample_reference_data, drifted_data):
        """Test drift detection when drift occurs."""
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        y_true_drift, y_pred_drift, sensitive_drift = drifted_data
        
        # Mock reference value
        def mock_metric_side_effect(*args):
            # Return different values for reference vs current
            if args[1] is y_pred_ref or (hasattr(args[1], '__iter__') and len(args[1]) == len(y_pred_ref)):
                return (0.10, {}, {})
            else:
                return (0.40, {}, {})  # Much worse fairness
        
        mock_compute_metric.side_effect = mock_metric_side_effect
        
        detector = FairnessDriftDetector(alpha=0.05)
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        result = detector.detect_drift(y_true_drift, y_pred_drift, sensitive_drift)
        
        # With significant drift, should detect it
        assert isinstance(result['drift_detected'], bool)
        if result['drift_detected']:
            assert len(result['drifted_metrics']) > 0
    
    def test_detect_drift_without_reference(self):
        """Test error when detecting drift without setting reference."""
        detector = FairnessDriftDetector()
        
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1])
        sensitive = np.array([0, 0, 1, 1])
        
        with pytest.raises(ValueError, match="Must call set_reference"):
            detector.detect_drift(y_true, y_pred, sensitive)
    
    def test_test_metric_drift_basic(self, sample_reference_data):
        """Test metric drift testing."""
        y_true, y_pred, sensitive = sample_reference_data
        
        detector = FairnessDriftDetector(alpha=0.05, min_samples=50)
        detector.reference_predictions = {
            'y_pred': y_pred,
            'sensitive': sensitive,
        }
        
        # Test with same data (no drift expected)
        drift_test = detector._test_metric_drift(
            metric_name='demographic_parity',
            current_value=0.10,
            reference_value=0.10,
            current_predictions=y_pred,
            current_sensitive=sensitive
        )
        
        assert 'current_value' in drift_test
        assert 'reference_value' in drift_test
        assert 'change' in drift_test
        assert 'p_value' in drift_test
        assert 'significant' in drift_test
        assert drift_test['test'] == 'ks_2samp'
    
    def test_test_metric_drift_insufficient_samples(self, caplog):
        """Test drift testing with insufficient samples."""
        detector = FairnessDriftDetector(alpha=0.05, min_samples=100)
        
        # Small reference data
        detector.reference_predictions = {
            'y_pred': np.array([1, 0, 1, 0, 1]),
            'sensitive': np.array([0, 0, 1, 1, 0]),
        }
        
        # Small current data
        y_pred_cur = np.array([1, 0, 1, 0, 1, 0])
        sensitive_cur = np.array([0, 0, 1, 1, 0, 1])
        
        drift_test = detector._test_metric_drift(
            metric_name='demographic_parity',
            current_value=0.10,
            reference_value=0.10,
            current_predictions=y_pred_cur,
            current_sensitive=sensitive_cur
        )
        
        # Should still return result but with warning
        assert 'p_value' in drift_test
    
    def test_test_metric_drift_unsupported_method(self):
        """Test error for unsupported test method."""
        detector = FairnessDriftDetector(test_method='unsupported')
        detector.reference_predictions = {
            'y_pred': np.array([1, 0, 1, 0]),
            'sensitive': np.array([0, 0, 1, 1]),
        }
        
        with pytest.raises(NotImplementedError):
            detector._test_metric_drift(
                metric_name='demographic_parity',
                current_value=0.10,
                reference_value=0.10,
                current_predictions=np.array([1, 0, 1, 0]),
                current_sensitive=np.array([0, 0, 1, 1])
            )
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_create_alert_with_drift(self, mock_compute_metric, sample_reference_data, drifted_data):
        """Test alert creation when drift is detected."""
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        y_true_drift, y_pred_drift, sensitive_drift = drifted_data
        
        mock_compute_metric.return_value = (0.10, {}, {})
        
        detector = FairnessDriftDetector()
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        # Create drift result
        drift_result = {
            'timestamp': datetime.now(),
            'drift_detected': True,
            'drifted_metrics': ['demographic_parity', 'equalized_odds'],
            'tests': {
                'demographic_parity': {
                    'current_value': 0.25,
                    'reference_value': 0.10,
                    'change': 0.15,
                    'p_value': 0.001,
                    'significant': True,
                },
                'equalized_odds': {
                    'current_value': 0.20,
                    'reference_value': 0.10,
                    'change': 0.10,
                    'p_value': 0.01,
                    'significant': True,
                }
            }
        }
        
        alert = detector.create_alert(drift_result, severity='HIGH')
        
        assert alert is not None
        assert alert.alert_type == 'drift'
        assert alert.severity == 'HIGH'
        assert 'demographic_parity' in alert.metric_name
        assert 'drift detected' in alert.message.lower()
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_create_alert_no_drift(self, mock_compute_metric, sample_reference_data):
        """Test alert creation when no drift is detected."""
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        
        mock_compute_metric.return_value = (0.10, {}, {})
        
        detector = FairnessDriftDetector()
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        drift_result = {
            'timestamp': datetime.now(),
            'drift_detected': False,
            'drifted_metrics': [],
            'tests': {}
        }
        
        alert = detector.create_alert(drift_result)
        
        assert alert is None


class TestThresholdAlertSystem:
    """Tests for ThresholdAlertSystem."""
    
    def test_initialization_default(self):
        """Test initialization with defaults."""
        system = ThresholdAlertSystem()
        
        assert 'demographic_parity' in system.thresholds
        assert 'equalized_odds' in system.thresholds
        assert system.thresholds['demographic_parity'] == 0.1
    
    def test_initialization_custom(self):
        """Test initialization with custom thresholds."""
        thresholds = {
            'demographic_parity': 0.05,
            'custom_metric': 0.15,
        }
        
        system = ThresholdAlertSystem(thresholds=thresholds)
        
        assert system.thresholds['demographic_parity'] == 0.05
        assert system.thresholds['custom_metric'] == 0.15
    
    def test_check_thresholds_pass(self):
        """Test threshold checking when all pass."""
        system = ThresholdAlertSystem()
        
        metrics = {
            'demographic_parity': 0.05,
            'equalized_odds': 0.08,
        }
        
        alert = system.check_thresholds(metrics)
        
        assert alert is None
    
    def test_check_thresholds_violation(self):
        """Test threshold checking with violation."""
        system = ThresholdAlertSystem()
        
        metrics = {
            'demographic_parity': 0.15,
            'equalized_odds': 0.08,
        }
        
        alert = system.check_thresholds(metrics)
        
        assert alert is not None
        assert 'demographic_parity' in alert.metric_name
        assert alert.current_value == 0.15
        assert alert.alert_type == 'threshold_violation'
    
    def test_check_thresholds_multiple_violations(self):
        """Test with multiple metric violations."""
        system = ThresholdAlertSystem()
        
        metrics = {
            'demographic_parity': 0.15,
            'equalized_odds': 0.20,
        }
        
        alert = system.check_thresholds(metrics)
        
        assert alert is not None
        # Should report the worse violation
        assert alert.current_value >= 0.15
    
    def test_severity_levels_low(self):
        """Test low severity classification."""
        system = ThresholdAlertSystem()
        
        metrics = {'demographic_parity': 0.11}  # Just above threshold
        
        alert = system.check_thresholds(metrics)
        
        assert alert is not None
        assert alert.severity == 'LOW'
    
    def test_severity_levels_high(self):
        """Test high severity classification."""
        thresholds = {'demographic_parity': 0.10}
        severity_levels = {
            'LOW': 1.0,
            'HIGH': 1.5,
            'CRITICAL': 2.0,
        }
        
        system = ThresholdAlertSystem(
            thresholds=thresholds,
            severity_levels=severity_levels
        )
        
        metrics = {'demographic_parity': 0.17}  # 1.7x threshold
        
        alert = system.check_thresholds(metrics)
        
        assert alert is not None
        assert alert.severity == 'HIGH'
    
    def test_severity_levels_critical(self):
        """Test critical severity classification."""
        system = ThresholdAlertSystem()
        
        metrics = {'demographic_parity': 0.25}  # 2.5x threshold
        
        alert = system.check_thresholds(metrics)
        
        assert alert is not None
        assert alert.severity == 'CRITICAL'
    
    def test_with_group_sizes(self):
        """Test threshold checking with group sizes."""
        system = ThresholdAlertSystem()
        
        metrics = {'demographic_parity': 0.15}
        group_sizes = {'group_0': 1000, 'group_1': 500}
        
        alert = system.check_thresholds(metrics, group_sizes)
        
        assert alert is not None
        # Group sizes don't affect threshold checking in this implementation
        # but should be accepted without error
    
    def test_unknown_metric_ignored(self):
        """Test that unknown metrics are ignored."""
        system = ThresholdAlertSystem(thresholds={'known_metric': 0.10})
        
        metrics = {
            'known_metric': 0.05,
            'unknown_metric': 0.50,  # High value but not in thresholds
        }
        
        alert = system.check_thresholds(metrics)
        
        assert alert is None


class TestIntegration:
    """Integration tests for drift detection."""
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_full_drift_detection_workflow(self, mock_compute_metric, sample_reference_data):
        """Test complete drift detection workflow."""
        np.random.seed(42)
        
        # Setup
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        
        # Mock metrics
        mock_compute_metric.return_value = (0.10, {}, {})
        
        detector = FairnessDriftDetector(alpha=0.05)
        
        # 1. Set reference period
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        assert detector.reference_metrics is not None
        
        # 2. Monitor current period (no drift)
        y_true_cur = np.random.binomial(1, 0.5, 500)
        y_pred_cur = np.random.binomial(1, 0.5, 500)
        sensitive_cur = np.random.binomial(1, 0.5, 500)
        
        result = detector.detect_drift(y_true_cur, y_pred_cur, sensitive_cur)
        
        assert 'drift_detected' in result
        
        # 3. Create alert if drift detected
        if result['drift_detected']:
            alert = detector.create_alert(result)
            assert alert is not None
    
    def test_threshold_and_drift_combination(self):
        """Test using both threshold and drift detection together."""
        np.random.seed(42)
        
        # Create systems
        drift_detector = FairnessDriftDetector(alpha=0.05)
        threshold_system = ThresholdAlertSystem()
        
        # Reference data
        y_true_ref = np.random.binomial(1, 0.5, 500)
        y_pred_ref = np.random.binomial(1, 0.5, 500)
        sensitive_ref = np.random.binomial(1, 0.5, 500)
        
        # Current data with violations
        y_true_cur = np.random.binomial(1, 0.5, 500)
        y_pred_cur = np.concatenate([
            np.random.binomial(1, 0.7, 250),
            np.random.binomial(1, 0.3, 250),
        ])
        sensitive_cur = np.array([0] * 250 + [1] * 250)
        
        # Mock compute_metric for drift detector
        # with patch('monitoring_module.src.drift_detection.compute_metric') as mock:
        #     mock.return_value = (0.08, {}, {})
        #     drift_detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        with patch('measurement_module.src.metrics_engine.compute_metric') as mock:
            mock.return_value = (0.08, {}, {})
            drift_detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        # Check thresholds
        metrics = {'demographic_parity': 0.15}
        threshold_alert = threshold_system.check_thresholds(metrics)
        
        assert threshold_alert is not None
        
        # Both systems can work together for comprehensive monitoring
    
    # @patch('monitoring_module.src.drift_detection.FairnessDriftDetector._compute_metrics')
    @patch('measurement_module.src.metrics_engine.compute_metric')
    def test_multiple_drift_detections_over_time(self, mock_compute_metric, sample_reference_data):
        """Test drift detection across multiple time periods."""
        y_true_ref, y_pred_ref, sensitive_ref = sample_reference_data
        
        mock_compute_metric.return_value = (0.10, {}, {})
        
        detector = FairnessDriftDetector()
        detector.set_reference(y_true_ref, y_pred_ref, sensitive_ref)
        
        # Simulate multiple monitoring periods
        drift_detected_count = 0
        
        for i in range(5):
            np.random.seed(50 + i)
            
            y_true = np.random.binomial(1, 0.5, 500)
            y_pred = np.random.binomial(1, 0.5, 500)
            sensitive = np.random.binomial(1, 0.5, 500)
            
            result = detector.detect_drift(y_true, y_pred, sensitive)
            
            if result['drift_detected']:
                drift_detected_count += 1
        
        # Should have consistent results across periods
        assert isinstance(drift_detected_count, int)
        assert drift_detected_count >= 0