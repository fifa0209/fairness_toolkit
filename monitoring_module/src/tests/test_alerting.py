"""
Tests for Alerting System

Tests the alerting infrastructure including threshold-based alerts,
adaptive thresholds, alert aggregation, and notification routing.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from monitoring_module.src.alerting import (
    AlertSeverity,
    AlertType,
    FairnessAlert,
    ThresholdAlertSystem,
    AdaptiveAlertSystem,
    AlertAggregator,
    AlertNotifier,
    log_handler,
    console_handler,
)


class TestAlertEnums:
    """Tests for alert enums."""
    
    def test_alert_severity_values(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.CRITICAL.value == "CRITICAL"
        assert AlertSeverity.HIGH.value == "HIGH"
        assert AlertSeverity.MEDIUM.value == "MEDIUM"
        assert AlertSeverity.LOW.value == "LOW"
        assert AlertSeverity.INFO.value == "INFO"
    
    def test_alert_type_values(self):
        """Test AlertType enum values."""
        assert AlertType.THRESHOLD_VIOLATION.value == "threshold_violation"
        assert AlertType.DRIFT_DETECTED.value == "drift_detected"
        assert AlertType.RAPID_DEGRADATION.value == "rapid_degradation"
        assert AlertType.CONSISTENT_BIAS.value == "consistent_bias"
        assert AlertType.INTERSECTIONAL_ISSUE.value == "intersectional_issue"


class TestFairnessAlert:
    """Tests for FairnessAlert dataclass."""
    
    def test_alert_creation(self):
        """Test creating a fairness alert."""
        timestamp = datetime.now()
        alert = FairnessAlert(
            alert_id="test_001",
            timestamp=timestamp,
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="demographic_parity",
            current_value=0.15,
            threshold=0.10,
            affected_groups=["group_A", "group_B"],
            message="Threshold exceeded",
            priority_score=75.0,
        )
        
        assert alert.alert_id == "test_001"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.current_value == 0.15
        assert len(alert.affected_groups) == 2
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        timestamp = datetime.now()
        alert = FairnessAlert(
            alert_id="test_001",
            timestamp=timestamp,
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="demographic_parity",
            current_value=0.15,
            threshold=0.10,
            affected_groups=["group_A"],
            message="Test message",
            evidence={'excess': 0.05},
            recommended_actions=["Action 1", "Action 2"],
            priority_score=75.0,
        )
        
        alert_dict = alert.to_dict()
        
        assert alert_dict['alert_id'] == "test_001"
        assert alert_dict['severity'] == "HIGH"
        assert alert_dict['current_value'] == 0.15
        assert alert_dict['evidence']['excess'] == 0.05
        assert len(alert_dict['recommended_actions']) == 2
    
    def test_alert_str(self):
        """Test string representation."""
        alert = FairnessAlert(
            alert_id="test_001",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.CRITICAL,
            metric_name="demographic_parity",
            current_value=0.25,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        alert_str = str(alert)
        
        assert "CRITICAL" in alert_str
        assert "threshold_violation" in alert_str
        assert "demographic_parity" in alert_str
        assert "0.250" in alert_str


class TestThresholdAlertSystem:
    """Tests for ThresholdAlertSystem."""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system with default thresholds."""
        thresholds = {
            'demographic_parity': 0.10,
            'equalized_odds': 0.10,
        }
        return ThresholdAlertSystem(thresholds)
    
    def test_initialization(self, alert_system):
        """Test system initialization."""
        assert alert_system.thresholds['demographic_parity'] == 0.10
        assert 'demographic_parity' in alert_system.severity_rules
    
    def test_check_thresholds_no_violation(self, alert_system):
        """Test when no thresholds are violated."""
        metrics = {
            'demographic_parity': 0.05,
            'equalized_odds': 0.08,
        }
        
        alert = alert_system.check_thresholds(metrics)
        
        assert alert is None
    
    def test_check_thresholds_violation(self, alert_system):
        """Test when threshold is violated."""
        metrics = {
            'demographic_parity': 0.15,
            'equalized_odds': 0.05,
        }
        
        alert = alert_system.check_thresholds(metrics)
        
        assert alert is not None
        assert alert.metric_name == 'demographic_parity'
        assert alert.current_value == 0.15
        assert alert.threshold == 0.10
        assert alert.alert_type == AlertType.THRESHOLD_VIOLATION
    
    def test_check_thresholds_with_group_metrics(self, alert_system):
        """Test threshold checking with group-level metrics."""
        metrics = {
            'demographic_parity': 0.12,
        }
        
        group_metrics = {
            'demographic_parity': {
                'group_0': 0.15,
                'group_1': 0.05,
            }
        }
        
        alert = alert_system.check_thresholds(metrics, group_metrics)
        
        assert alert is not None
        assert 'group_0' in alert.affected_groups
        assert 'group_1' not in alert.affected_groups
    
    def test_severity_determination_critical(self, alert_system):
        """Test critical severity determination."""
        severity = alert_system._determine_severity('demographic_parity', 0.30)
        assert severity == AlertSeverity.CRITICAL
    
    def test_severity_determination_high(self, alert_system):
        """Test high severity determination."""
        severity = alert_system._determine_severity('demographic_parity', 0.22)
        assert severity == AlertSeverity.HIGH
    
    def test_severity_determination_medium(self, alert_system):
        """Test medium severity determination."""
        severity = alert_system._determine_severity('demographic_parity', 0.17)
        assert severity == AlertSeverity.MEDIUM
    
    def test_severity_determination_low(self, alert_system):
        """Test low severity determination."""
        severity = alert_system._determine_severity('demographic_parity', 0.12)
        assert severity == AlertSeverity.LOW
    
    def test_custom_severity_rules(self):
        """Test custom severity rules."""
        thresholds = {'metric_a': 0.10}
        severity_rules = {
            'metric_a': {
                'critical': 0.30,
                'high': 0.20,
                'medium': 0.15,
                'low': 0.10,
            }
        }
        
        alert_system = ThresholdAlertSystem(thresholds, severity_rules)
        
        severity = alert_system._determine_severity('metric_a', 0.22)
        assert severity == AlertSeverity.HIGH
    
    def test_identify_affected_groups(self, alert_system):
        """Test identifying affected groups."""
        group_values = {
            'group_A': 0.15,
            'group_B': 0.08,
            'group_C': 0.12,
        }
        
        threshold = 0.10
        affected = alert_system._identify_affected_groups(group_values, threshold)
        
        assert 'group_A' in affected
        assert 'group_C' in affected
        assert 'group_B' not in affected
    
    def test_calculate_priority(self, alert_system):
        """Test priority score calculation."""
        priority = alert_system._calculate_priority(
            severity=AlertSeverity.HIGH,
            excess_pct=50.0,
            n_groups=2
        )
        
        assert 70 <= priority <= 100
        assert isinstance(priority, float)
    
    def test_calculate_priority_critical(self, alert_system):
        """Test priority calculation for critical alert."""
        priority = alert_system._calculate_priority(
            severity=AlertSeverity.CRITICAL,
            excess_pct=100.0,
            n_groups=3
        )
        
        assert priority >= 90
    
    def test_alert_message_format(self, alert_system):
        """Test alert message formatting."""
        metrics = {'demographic_parity': 0.15}
        alert = alert_system.check_thresholds(metrics)
        
        assert '0.15' in alert.message or '0.150' in alert.message
        assert '0.10' in alert.message or '0.100' in alert.message
        assert 'demographic_parity' in alert.message


class TestAdaptiveAlertSystem:
    """Tests for AdaptiveAlertSystem."""
    
    @pytest.fixture
    def adaptive_system(self):
        """Create adaptive alert system."""
        thresholds = {
            'demographic_parity': 0.10,
            'equalized_odds': 0.10,
        }
        return AdaptiveAlertSystem(thresholds, target_fpr=0.05, adaptation_window=10)
    
    def test_initialization(self, adaptive_system):
        """Test adaptive system initialization."""
        assert adaptive_system.target_fpr == 0.05
        assert adaptive_system.adaptation_window == 10
        assert len(adaptive_system.alert_history) == 0
        assert len(adaptive_system.feedback_history) == 0
    
    def test_check_and_adapt_no_violation(self, adaptive_system):
        """Test checking when no violation occurs."""
        metrics = {'demographic_parity': 0.05}
        alert = adaptive_system.check_and_adapt(metrics)
        
        assert alert is None
    
    def test_check_and_adapt_with_violation(self, adaptive_system):
        """Test checking when violation occurs."""
        metrics = {'demographic_parity': 0.15}
        alert = adaptive_system.check_and_adapt(metrics)
        
        assert alert is not None
        assert len(adaptive_system.alert_history) == 1
    
    def test_provide_feedback(self, adaptive_system):
        """Test providing feedback on alerts."""
        adaptive_system.provide_feedback('alert_001', is_true_positive=True)
        
        assert len(adaptive_system.feedback_history) == 1
        assert adaptive_system.feedback_history[0]['alert_id'] == 'alert_001'
        assert adaptive_system.feedback_history[0]['is_true_positive'] is True
    
    def test_adapt_thresholds_high_fpr(self, adaptive_system):
        """Test threshold adaptation with high false positive rate."""
        # Simulate alerts
        for i in range(15):
            metrics = {'demographic_parity': 0.12}
            alert = adaptive_system.check_and_adapt(metrics)
            
            if alert:
                # Provide feedback (80% false positives)
                is_tp = i % 5 == 0
                adaptive_system.provide_feedback(
                    alert.alert_id,
                    is_true_positive=is_tp
                )
        
        # Threshold should increase due to high FPR
        original_threshold = 0.10
        new_threshold = adaptive_system.thresholds['demographic_parity']
        
        # With high FPR, threshold should increase
        assert new_threshold >= original_threshold
    
    def test_adapt_thresholds_low_fpr(self, adaptive_system):
        """Test threshold adaptation with low false positive rate."""
        # Simulate alerts with low FPR
        for i in range(15):
            metrics = {'demographic_parity': 0.12}
            alert = adaptive_system.check_and_adapt(metrics)
            
            if alert:
                # All true positives
                adaptive_system.provide_feedback(
                    alert.alert_id,
                    is_true_positive=True
                )
        
        # With very low FPR, threshold might decrease
        original_threshold = 0.10
        new_threshold = adaptive_system.thresholds['demographic_parity']
        
        # Threshold might decrease or stay same
        assert new_threshold <= original_threshold * 1.1


class TestAlertAggregator:
    """Tests for AlertAggregator."""
    
    @pytest.fixture
    def aggregator(self):
        """Create alert aggregator."""
        return AlertAggregator(grouping_window_minutes=60, max_alerts_per_window=5)
    
    def test_initialization(self, aggregator):
        """Test aggregator initialization."""
        assert aggregator.max_alerts == 5
        assert len(aggregator.alert_buffer) == 0
    
    def test_add_alert_first(self, aggregator):
        """Test adding first alert."""
        alert = FairnessAlert(
            alert_id="alert_001",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="demographic_parity",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        emitted = aggregator.add_alert(alert)
        
        assert len(emitted) == 0  # First alert doesn't emit
        assert len(aggregator.alert_buffer) == 1
    
    def test_add_alert_within_window(self, aggregator):
        """Test adding alerts within time window."""
        base_time = datetime.now()
        
        for i in range(3):
            alert = FairnessAlert(
                alert_id=f"alert_{i}",
                timestamp=base_time + timedelta(minutes=i*10),
                alert_type=AlertType.THRESHOLD_VIOLATION,
                severity=AlertSeverity.MEDIUM,
                metric_name="demographic_parity",
                current_value=0.12,
                threshold=0.10,
                affected_groups=[],
                message=f"Test {i}",
            )
            
            emitted = aggregator.add_alert(alert)
            
            # Within window, no emission
            if i < 2:
                assert len(emitted) == 0
        
        assert len(aggregator.alert_buffer) == 3
    
    def test_add_alert_window_expires(self, aggregator):
        """Test alert emission when window expires."""
        base_time = datetime.now()
        
        # Add first alert
        alert1 = FairnessAlert(
            alert_id="alert_001",
            timestamp=base_time,
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="demographic_parity",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test 1",
            priority_score=80.0,
        )
        aggregator.add_alert(alert1)
        
        # Add alert after window
        alert2 = FairnessAlert(
            alert_id="alert_002",
            timestamp=base_time + timedelta(minutes=65),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.MEDIUM,
            metric_name="equalized_odds",
            current_value=0.12,
            threshold=0.10,
            affected_groups=[],
            message="Test 2",
            priority_score=60.0,
        )
        
        emitted = aggregator.add_alert(alert2)
        
        # Should emit aggregated alerts
        assert len(emitted) > 0
        assert len(aggregator.alert_buffer) == 0  # Buffer cleared
    
    def test_merge_alerts_same_metric(self, aggregator):
        """Test merging multiple alerts for same metric."""
        base_time = datetime.now()
        
        alerts = [
            FairnessAlert(
                alert_id=f"alert_{i}",
                timestamp=base_time + timedelta(minutes=i*10),
                alert_type=AlertType.THRESHOLD_VIOLATION,
                severity=AlertSeverity.MEDIUM if i == 0 else AlertSeverity.HIGH,
                metric_name="demographic_parity",
                current_value=0.12 + i*0.01,
                threshold=0.10,
                affected_groups=[f"group_{i}"],
                message=f"Test {i}",
                priority_score=70.0 + i*5,
            )
            for i in range(3)
        ]
        
        merged = aggregator._merge_alerts(alerts)
        
        assert merged.alert_type == AlertType.CONSISTENT_BIAS
        assert merged.severity == AlertSeverity.HIGH  # Highest severity
        assert len(merged.affected_groups) == 3  # All groups combined
        assert 'Aggregated' in merged.message
    
    def test_prioritization(self, aggregator):
        """Test alert prioritization."""
        base_time = datetime.now()
        
        # Add alerts with different priorities
        for i, priority in enumerate([50, 90, 70, 60, 85]):
            alert = FairnessAlert(
                alert_id=f"alert_{i}",
                timestamp=base_time + timedelta(minutes=i),
                alert_type=AlertType.THRESHOLD_VIOLATION,
                severity=AlertSeverity.MEDIUM,
                metric_name=f"metric_{i}",
                current_value=0.12,
                threshold=0.10,
                affected_groups=[],
                message=f"Test {i}",
                priority_score=priority,
            )
            aggregator.alert_buffer.append(alert)
        
        # Emit after 65 minutes
        alert_trigger = FairnessAlert(
            alert_id="trigger",
            timestamp=base_time + timedelta(minutes=65),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.LOW,
            metric_name="trigger",
            current_value=0.11,
            threshold=0.10,
            affected_groups=[],
            message="Trigger",
            priority_score=40.0,
        )
        
        emitted = aggregator.add_alert(alert_trigger)
        
        # Should emit top 5 by priority
        assert len(emitted) <= 5
        
        # Check ordering (highest priority first)
        if len(emitted) > 1:
            for i in range(len(emitted) - 1):
                assert emitted[i].priority_score >= emitted[i+1].priority_score


class TestAlertNotifier:
    """Tests for AlertNotifier."""
    
    @pytest.fixture
    def notifier(self):
        """Create alert notifier."""
        return AlertNotifier()
    
    def test_initialization(self, notifier):
        """Test notifier initialization."""
        assert len(notifier.handlers) == 0
        assert len(notifier.routing_rules) == 0
    
    def test_register_handler(self, notifier):
        """Test registering notification handler."""
        def test_handler(alert):
            pass
        
        notifier.register_handler('email', test_handler)
        
        assert 'email' in notifier.handlers
        assert notifier.handlers['email'] == test_handler
    
    def test_add_routing_rule(self, notifier):
        """Test adding routing rule."""
        notifier.add_routing_rule(AlertSeverity.HIGH, ['email', 'slack'])
        
        assert len(notifier.routing_rules) == 1
        assert notifier.routing_rules[0]['severity'] == AlertSeverity.HIGH
        assert 'email' in notifier.routing_rules[0]['channels']
    
    def test_determine_channels_single_rule(self, notifier):
        """Test channel determination with single rule."""
        notifier.add_routing_rule(AlertSeverity.HIGH, ['email'])
        notifier.add_routing_rule(AlertSeverity.CRITICAL, ['pagerduty'])
        
        alert_high = FairnessAlert(
            alert_id="test",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="test",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        channels = notifier._determine_channels(alert_high)
        
        assert 'email' in channels
    
    def test_determine_channels_multiple_rules(self, notifier):
        """Test channel determination with multiple matching rules."""
        notifier.add_routing_rule(AlertSeverity.MEDIUM, ['slack'])
        notifier.add_routing_rule(AlertSeverity.HIGH, ['email'])
        notifier.add_routing_rule(AlertSeverity.CRITICAL, ['pagerduty'])
        
        alert_critical = FairnessAlert(
            alert_id="test",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.CRITICAL,
            metric_name="test",
            current_value=0.25,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        channels = notifier._determine_channels(alert_critical)
        
        # Critical should match all rules
        assert 'slack' in channels
        assert 'email' in channels
        assert 'pagerduty' in channels
    
    def test_notify_success(self, notifier):
        """Test successful notification."""
        handler_called = []
        
        def mock_handler(alert):
            handler_called.append(alert.alert_id)
        
        notifier.register_handler('test_channel', mock_handler)
        notifier.add_routing_rule(AlertSeverity.HIGH, ['test_channel'])
        
        alert = FairnessAlert(
            alert_id="alert_001",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="test",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        notifier.notify(alert)
        
        assert 'alert_001' in handler_called
    
    def test_notify_missing_handler(self, notifier, caplog):
        """Test notification with missing handler."""
        notifier.add_routing_rule(AlertSeverity.HIGH, ['nonexistent'])
        
        alert = FairnessAlert(
            alert_id="alert_001",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="test",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        notifier.notify(alert)
        
        # Should log warning
        assert any('No handler registered' in record.message for record in caplog.records)
    
    def test_notify_handler_exception(self, notifier, caplog):
        """Test handling of handler exceptions."""
        def failing_handler(alert):
            raise Exception("Handler failed")
        
        notifier.register_handler('failing', failing_handler)
        notifier.add_routing_rule(AlertSeverity.HIGH, ['failing'])
        
        alert = FairnessAlert(
            alert_id="alert_001",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="test",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test",
        )
        
        # Should not raise exception
        notifier.notify(alert)


class TestNotificationHandlers:
    """Tests for notification handler functions."""
    
    def test_log_handler(self, caplog):
        """Test log handler."""
        alert = FairnessAlert(
            alert_id="test",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.HIGH,
            metric_name="demographic_parity",
            current_value=0.15,
            threshold=0.10,
            affected_groups=[],
            message="Test alert",
        )
        
        log_handler(alert)
        
        # Check log was created
        assert any('ALERT' in record.message for record in caplog.records)
    
    def test_console_handler(self, capsys):
        """Test console handler."""
        alert = FairnessAlert(
            alert_id="test",
            timestamp=datetime.now(),
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=AlertSeverity.CRITICAL,
            metric_name="demographic_parity",
            current_value=0.25,
            threshold=0.10,
            affected_groups=['group_A', 'group_B'],
            message="Critical violation",
            recommended_actions=["Action 1", "Action 2"],
        )
        
        console_handler(alert)
        
        captured = capsys.readouterr()
        
        assert 'FAIRNESS ALERT' in captured.out
        assert 'CRITICAL' in captured.out
        assert 'demographic_parity' in captured.out
        assert 'Action 1' in captured.out


class TestIntegration:
    """Integration tests for alerting system."""
    
    def test_full_alerting_workflow(self):
        """Test complete alerting workflow."""
        # 1. Setup threshold system
        thresholds = {'demographic_parity': 0.10}
        threshold_system = ThresholdAlertSystem(thresholds)
        
        # 2. Setup notifier
        notifier = AlertNotifier()
        
        notified_alerts = []
        def capture_handler(alert):
            notified_alerts.append(alert)
        
        notifier.register_handler('capture', capture_handler)
        notifier.add_routing_rule(AlertSeverity.HIGH, ['capture'])
        
        # 3. Check metrics and trigger alert
        metrics = {'demographic_parity': 0.20}
        alert = threshold_system.check_thresholds(metrics)
        
        assert alert is not None
        
        # 4. Notify
        notifier.notify(alert)
        
        assert len(notified_alerts) == 1
        assert notified_alerts[0].metric_name == 'demographic_parity'
    
    def test_adaptive_system_with_aggregation(self):
        """Test adaptive system with alert aggregation."""
        thresholds = {'demographic_parity': 0.10}
        adaptive = AdaptiveAlertSystem(thresholds, adaptation_window=5)
        aggregator = AlertAggregator(grouping_window_minutes=60, max_alerts_per_window=3)
        
        base_time = datetime.now()
        
        # Generate series of alerts
        for i in range(8):
            metrics = {'demographic_parity': 0.12 + i*0.01}
            alert = adaptive.check_and_adapt(
                metrics,
                timestamp=base_time + timedelta(minutes=i*5)
            )
            
            if alert:
                emitted = aggregator.add_alert(alert)
                
                # Provide feedback
                adaptive.provide_feedback(
                    alert.alert_id,
                    is_true_positive=(i % 2 == 0)
                )
        
        # Verify adaptation occurred
        assert len(adaptive.feedback_history) > 0