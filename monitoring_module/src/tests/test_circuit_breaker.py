"""
Tests for Fairness Circuit Breaker System

Tests the circuit breaker functionality including state transitions,
interventions, and recovery mechanisms.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from monitoring_module.src.circuit_breaker import (
    CircuitState,
    InterventionType,
    CircuitBreakerConfig,
    CircuitBreakerEvent,
    FairnessCircuitBreaker,
    CircuitBreakerMonitor,
)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        
        assert config.critical_threshold == 0.20
        assert config.failure_count_threshold == 3
        assert config.cooldown_period_seconds == 300
        assert config.recovery_check_interval == 60
        assert config.intervention_type == InterventionType.ROUTE_TO_BASELINE
        assert config.auto_recovery is True
        assert config.recovery_threshold == 0.10
        assert config.recovery_sample_size == 100
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CircuitBreakerConfig(
            critical_threshold=0.15,
            failure_count_threshold=5,
            intervention_type=InterventionType.REDUCE_TRAFFIC,
            auto_recovery=False
        )
        
        assert config.critical_threshold == 0.15
        assert config.failure_count_threshold == 5
        assert config.intervention_type == InterventionType.REDUCE_TRAFFIC
        assert config.auto_recovery is False


class TestFairnessCircuitBreaker:
    """Tests for FairnessCircuitBreaker."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return CircuitBreakerConfig(
            critical_threshold=0.20,
            failure_count_threshold=3,
            cooldown_period_seconds=60,
            intervention_type=InterventionType.ROUTE_TO_BASELINE,
            baseline_model_id='safe_baseline_v1'
        )
    
    @pytest.fixture
    def breaker(self, config):
        """Create circuit breaker instance."""
        return FairnessCircuitBreaker(config)
    
    def test_initialization(self, breaker, config):
        """Test circuit breaker initialization."""
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0
        assert breaker.last_failure_time is None
        assert len(breaker.events) == 0
        assert len(breaker.metrics_buffer) == 0
        assert breaker.config == config
    
    def test_initialization_with_notification_handler(self, config):
        """Test initialization with notification handler."""
        handler = Mock()
        breaker = FairnessCircuitBreaker(config, notification_handler=handler)
        
        assert breaker.notification_handler == handler
    
    def test_record_metrics_no_violation(self, breaker):
        """Test recording metrics without violation."""
        metrics = {'demographic_parity': 0.05, 'equalized_odds': 0.08}
        
        event = breaker.record_metrics(metrics)
        
        assert event is None
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 0
        assert len(breaker.metrics_buffer) == 1
    
    def test_record_metrics_single_violation(self, breaker):
        """Test single violation doesn't open circuit."""
        metrics = {'demographic_parity': 0.25}
        
        event = breaker.record_metrics(metrics)
        
        assert event is None
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 1
        assert breaker.last_failure_time is not None
    
    def test_record_metrics_consecutive_violations_opens_circuit(self, breaker):
        """Test consecutive violations open circuit."""
        metrics = {'demographic_parity': 0.25}
        
        # First two violations
        breaker.record_metrics(metrics)
        breaker.record_metrics(metrics)
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.consecutive_failures == 2
        
        # Third violation should open circuit
        event = breaker.record_metrics(metrics)
        
        assert event is not None
        assert breaker.state == CircuitState.OPEN
        assert event.event_type == 'opened'
        assert event.trigger_metric == 'demographic_parity'
        assert event.trigger_value == 0.25
        assert event.state_before == CircuitState.CLOSED
        assert event.state_after == CircuitState.OPEN
    
    def test_record_metrics_resets_counter_on_success(self, breaker):
        """Test failure counter resets after successful metrics."""
        # Two violations
        breaker.record_metrics({'demographic_parity': 0.25})
        breaker.record_metrics({'demographic_parity': 0.25})
        assert breaker.consecutive_failures == 2
        
        # Good metrics reset counter
        breaker.record_metrics({'demographic_parity': 0.05})
        assert breaker.consecutive_failures == 0
        
        # New violation starts count fresh
        breaker.record_metrics({'demographic_parity': 0.25})
        assert breaker.consecutive_failures == 1
    
    def test_circuit_open_notification(self, config):
        """Test notification handler is called when circuit opens."""
        handler = Mock()
        breaker = FairnessCircuitBreaker(config, notification_handler=handler)
        
        metrics = {'demographic_parity': 0.25}
        
        # Trigger circuit open
        for _ in range(3):
            breaker.record_metrics(metrics)
        
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == 'opened'
    
    def test_metrics_buffer_size_limit(self, breaker):
        """Test metrics buffer maintains size limit."""
        # Add more than 1000 metrics
        for i in range(1500):
            breaker.record_metrics({'demographic_parity': 0.05})
        
        assert len(breaker.metrics_buffer) == 1000
    
    def test_should_use_baseline_closed_state(self, breaker):
        """Test baseline usage in closed state."""
        assert breaker.should_use_baseline() is False
    
    def test_should_use_baseline_open_state(self, breaker):
        """Test baseline usage in open state."""
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        assert breaker.should_use_baseline() is True
    
    def test_should_use_baseline_different_intervention(self, config):
        """Test baseline usage with different intervention type."""
        config.intervention_type = InterventionType.ALERT_ONLY
        breaker = FairnessCircuitBreaker(config)
        
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        assert breaker.should_use_baseline() is False
    
    def test_get_traffic_allocation_closed(self, breaker):
        """Test traffic allocation in closed state."""
        allocation = breaker.get_traffic_allocation()
        
        assert allocation['production'] == 1.0
        assert allocation['baseline'] == 0.0
    
    def test_get_traffic_allocation_open(self, breaker):
        """Test traffic allocation in open state."""
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        allocation = breaker.get_traffic_allocation()
        
        assert allocation['production'] == 0.0
        assert allocation['baseline'] == 1.0
    
    def test_get_traffic_allocation_half_open(self, breaker):
        """Test traffic allocation in half-open state."""
        breaker.state = CircuitState.HALF_OPEN
        
        allocation = breaker.get_traffic_allocation()
        
        assert allocation['production'] == 0.2
        assert allocation['baseline'] == 0.8
    
    def test_get_traffic_allocation_reduce_traffic_intervention(self, config):
        """Test traffic allocation with reduce traffic intervention."""
        config.intervention_type = InterventionType.REDUCE_TRAFFIC
        breaker = FairnessCircuitBreaker(config)
        
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        allocation = breaker.get_traffic_allocation()
        
        assert allocation['production'] == 0.1
        assert allocation['baseline'] == 0.9
    
    def test_manual_override(self, breaker):
        """Test manual state override."""
        event = breaker.manual_override(CircuitState.OPEN, "Emergency maintenance")
        
        assert breaker.state == CircuitState.OPEN
        assert event.event_type == 'manual_override'
        assert event.metadata['reason'] == "Emergency maintenance"
        assert event.metadata['manual'] is True
    
    def test_manual_override_notification(self, config):
        """Test notification on manual override."""
        handler = Mock()
        breaker = FairnessCircuitBreaker(config, notification_handler=handler)
        
        breaker.manual_override(CircuitState.OPEN, "Testing")
        
        handler.assert_called_once()
    
    def test_get_status(self, breaker):
        """Test getting circuit breaker status."""
        status = breaker.get_status()
        
        assert 'state' in status
        assert 'consecutive_failures' in status
        assert 'last_failure_time' in status
        assert 'using_baseline' in status
        assert 'traffic_allocation' in status
        assert 'total_events' in status
        assert 'config' in status
        
        assert status['state'] == 'CLOSED'
        assert status['consecutive_failures'] == 0
    
    def test_get_event_history_empty(self, breaker):
        """Test getting empty event history."""
        history = breaker.get_event_history()
        
        assert history == []
    
    def test_get_event_history_with_events(self, breaker):
        """Test getting event history."""
        # Trigger some events
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        history = breaker.get_event_history()
        
        assert len(history) == 1
        assert history[0]['event_type'] == 'opened'
        assert 'timestamp' in history[0]
        assert 'trigger_metric' in history[0]
    
    def test_get_event_history_with_limit(self, breaker):
        """Test getting limited event history."""
        # Create multiple events
        for i in range(5):
            breaker.manual_override(CircuitState.OPEN, f"Test {i}")
        
        history = breaker.get_event_history(limit=2)
        
        assert len(history) == 2
    
    def test_auto_recovery_transition_to_half_open(self, breaker):
        """Test automatic transition to half-open after cooldown."""
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        assert breaker.state == CircuitState.OPEN
        
        # Simulate time passing
        future_time = datetime.now() + timedelta(seconds=61)
        
        event = breaker.record_metrics(
            {'demographic_parity': 0.05},
            timestamp=future_time
        )
        
        assert event is not None
        assert breaker.state == CircuitState.HALF_OPEN
        assert event.event_type == 'half_open'
    
    def test_no_auto_recovery_when_disabled(self, config):
        """Test no auto recovery when disabled."""
        config.auto_recovery = False
        breaker = FairnessCircuitBreaker(config)
        
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        # Wait for cooldown
        future_time = datetime.now() + timedelta(seconds=61)
        event = breaker.record_metrics(
            {'demographic_parity': 0.05},
            timestamp=future_time
        )
        
        # Should remain open
        assert breaker.state == CircuitState.OPEN
    
    def test_recovery_verification_insufficient_samples(self, breaker):
        """Test recovery requires sufficient samples."""
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()
        
        # Add few good metrics
        for _ in range(10):
            breaker.record_metrics({'demographic_parity': 0.05})
        
        # Should still be half-open (need 100 samples)
        assert breaker.state == CircuitState.HALF_OPEN
    
    def test_recovery_verification_success(self, breaker):
        """Test successful recovery closes circuit."""
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()
        
        # Add sufficient good metrics
        for _ in range(breaker.config.recovery_sample_size + 10):
            event = breaker.record_metrics({'demographic_parity': 0.05})
        
        # Should close circuit
        assert breaker.state == CircuitState.CLOSED
        assert any(e.event_type == 'closed' for e in breaker.events)
    
    def test_recovery_verification_failure(self, breaker):
        """Test failed recovery reopens circuit."""
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()
        
        # Add some good metrics, then a bad one
        for _ in range(50):
            breaker.record_metrics({'demographic_parity': 0.05})
        
        # Add bad metric
        event = breaker.record_metrics({'demographic_parity': 0.25})
        
        # Should reopen circuit
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerMonitor:
    """Tests for CircuitBreakerMonitor."""
    
    @pytest.fixture
    def monitor(self):
        """Create monitor instance."""
        return CircuitBreakerMonitor()
    
    def test_initialization(self, monitor):
        """Test monitor initialization."""
        assert len(monitor.breakers) == 0
    
    def test_register_breaker(self, monitor):
        """Test registering a circuit breaker."""
        config = CircuitBreakerConfig()
        breaker = FairnessCircuitBreaker(config)
        
        monitor.register_breaker('model_a', breaker)
        
        assert 'model_a' in monitor.breakers
        assert monitor.breakers['model_a'] == breaker
    
    def test_get_system_status_empty(self, monitor):
        """Test system status with no breakers."""
        status = monitor.get_system_status()
        
        assert status['total_breakers'] == 0
        assert len(status['breakers']) == 0
        assert status['healthy'] is True
    
    def test_get_system_status_with_breakers(self, monitor):
        """Test system status with multiple breakers."""
        # Add breakers in different states
        config = CircuitBreakerConfig()
        
        breaker1 = FairnessCircuitBreaker(config)
        breaker2 = FairnessCircuitBreaker(config)
        
        # Open one circuit
        for _ in range(3):
            breaker2.record_metrics({'demographic_parity': 0.25})
        
        monitor.register_breaker('model_a', breaker1)
        monitor.register_breaker('model_b', breaker2)
        
        status = monitor.get_system_status()
        
        assert status['total_breakers'] == 2
        assert len(status['breakers']) == 2
        assert status['state_summary']['CLOSED'] == 1
        assert status['state_summary']['OPEN'] == 1
        assert status['healthy'] is False
    
    def test_get_alerts_no_issues(self, monitor):
        """Test getting alerts when all healthy."""
        config = CircuitBreakerConfig()
        breaker = FairnessCircuitBreaker(config)
        monitor.register_breaker('model_a', breaker)
        
        alerts = monitor.get_alerts()
        
        assert len(alerts) == 0
    
    def test_get_alerts_with_open_circuit(self, monitor):
        """Test getting alerts with open circuit."""
        config = CircuitBreakerConfig()
        breaker = FairnessCircuitBreaker(config)
        
        # Open circuit
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        
        monitor.register_breaker('model_a', breaker)
        
        alerts = monitor.get_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]['breaker_name'] == 'model_a'
        assert alerts[0]['state'] == 'OPEN'
        assert alerts[0]['severity'] == 'CRITICAL'
    
    def test_get_alerts_with_half_open_circuit(self, monitor):
        """Test getting alerts with half-open circuit."""
        config = CircuitBreakerConfig()
        breaker = FairnessCircuitBreaker(config)
        
        breaker.state = CircuitState.HALF_OPEN
        breaker.last_state_change = datetime.now()
        
        monitor.register_breaker('model_a', breaker)
        
        alerts = monitor.get_alerts()
        
        assert len(alerts) == 1
        assert alerts[0]['severity'] == 'WARNING'


class TestIntegration:
    """Integration tests for circuit breaker system."""
    
    def test_complete_lifecycle(self):
        """Test complete circuit breaker lifecycle."""
        config = CircuitBreakerConfig(
            critical_threshold=0.20,
            failure_count_threshold=3,
            cooldown_period_seconds=5,
            recovery_sample_size=10
        )
        
        breaker = FairnessCircuitBreaker(config)
        
        # Start: CLOSED
        assert breaker.state == CircuitState.CLOSED
        
        # Trigger violations -> OPEN
        for _ in range(3):
            breaker.record_metrics({'demographic_parity': 0.25})
        assert breaker.state == CircuitState.OPEN
        
        # Wait for cooldown -> HALF_OPEN
        import time
        time.sleep(6)
        breaker.record_metrics(
            {'demographic_parity': 0.05},
            timestamp=datetime.now()
        )
        assert breaker.state == CircuitState.HALF_OPEN
        
        # Good metrics -> CLOSED
        for _ in range(15):
            breaker.record_metrics({'demographic_parity': 0.05})
        assert breaker.state == CircuitState.CLOSED
    
    def test_production_simulation(self):
        """Simulate production monitoring scenario."""
        config = CircuitBreakerConfig(
            critical_threshold=0.15,
            failure_count_threshold=2
        )
        
        monitor = CircuitBreakerMonitor()
        breaker = FairnessCircuitBreaker(config)
        monitor.register_breaker('credit_model', breaker)
        
        # Simulate predictions with degrading fairness
        np.random.seed(42)
        
        predictions_made = 0
        baseline_used = 0
        
        for batch in range(20):
            # Simulate fairness degradation
            fairness = 0.08 + (batch * 0.02)
            
            breaker.record_metrics({'demographic_parity': fairness})
            
            # Route traffic based on circuit state
            if breaker.should_use_baseline():
                baseline_used += 1
            else:
                predictions_made += 1
        
        # Verify system responded to degradation
        assert baseline_used > 0
        assert breaker.state in [CircuitState.OPEN, CircuitState.HALF_OPEN]
    
    def test_multiple_models_monitoring(self):
        """Test monitoring multiple models simultaneously."""
        monitor = CircuitBreakerMonitor()
        
        # Create breakers for different models
        for model_name in ['model_a', 'model_b', 'model_c']:
            config = CircuitBreakerConfig()
            breaker = FairnessCircuitBreaker(config)
            monitor.register_breaker(model_name, breaker)
        
        # Degrade one model
        for _ in range(3):
            monitor.breakers['model_b'].record_metrics(
                {'demographic_parity': 0.25}
            )
        
        # Check system status
        status = monitor.get_system_status()
        
        assert status['total_breakers'] == 3
        assert status['state_summary']['OPEN'] == 1
        assert status['state_summary']['CLOSED'] == 2
        assert not status['healthy']
        
        # Check alerts
        alerts = monitor.get_alerts()
        assert len(alerts) == 1
        assert alerts[0]['breaker_name'] == 'model_b'