"""
Fairness Circuit Breaker System

Automatically triggers interventions when severe fairness violations occur,
including routing traffic to safe baseline models.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from shared.logging import get_logger

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"  # Normal operation
    OPEN = "OPEN"  # Circuit broken, using fallback
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class InterventionType(Enum):
    """Types of automated interventions."""
    ROUTE_TO_BASELINE = "route_to_baseline"
    REDUCE_TRAFFIC = "reduce_traffic"
    ENABLE_MITIGATION = "enable_mitigation"
    ALERT_ONLY = "alert_only"
    STOP_PREDICTIONS = "stop_predictions"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    
    # Thresholds
    critical_threshold: float = 0.20  # Fairness metric threshold
    failure_count_threshold: int = 3  # Consecutive violations before opening
    
    # Timing
    cooldown_period_seconds: int = 300  # 5 minutes before trying half-open
    recovery_check_interval: int = 60  # Check recovery every minute
    
    # Interventions
    intervention_type: InterventionType = InterventionType.ROUTE_TO_BASELINE
    baseline_model_id: Optional[str] = None
    
    # Recovery
    auto_recovery: bool = True
    recovery_threshold: float = 0.10  # Metric must be below this to recover
    recovery_sample_size: int = 100  # Samples needed to verify recovery


@dataclass
class CircuitBreakerEvent:
    """Record of circuit breaker event."""
    
    event_id: str
    timestamp: datetime
    event_type: str  # 'opened', 'closed', 'half_open'
    trigger_metric: str
    trigger_value: float
    threshold: float
    state_before: CircuitState
    state_after: CircuitState
    intervention_taken: Optional[InterventionType] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class FairnessCircuitBreaker:
    """
    Circuit breaker for fairness violations.
    
    Automatically protects system from severe fairness degradation by:
    - Monitoring fairness metrics in real-time
    - Opening circuit when violations exceed threshold
    - Routing traffic to safe baseline model
    - Testing recovery before fully reopening
    
    Example:
        >>> config = CircuitBreakerConfig(
        ...     critical_threshold=0.20,
        ...     intervention_type=InterventionType.ROUTE_TO_BASELINE,
        ...     baseline_model_id='safe_baseline_v1'
        ... )
        >>> 
        >>> breaker = FairnessCircuitBreaker(config)
        >>> 
        >>> # In prediction loop
        >>> if breaker.should_use_baseline():
        ...     predictions = baseline_model.predict(X)
        ... else:
        ...     predictions = production_model.predict(X)
        >>> 
        >>> # Report metrics
        >>> breaker.record_metrics({'demographic_parity': 0.25})
    """
    
    def __init__(
        self,
        config: CircuitBreakerConfig,
        notification_handler: Optional[Callable] = None
    ):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
            notification_handler: Function to call when state changes
        """
        self.config = config
        self.notification_handler = notification_handler
        
        # State
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change: Optional[datetime] = None
        
        # History
        self.events: List[CircuitBreakerEvent] = []
        self.metrics_buffer: List[Dict[str, float]] = []
        
        logger.info(
            f"Initialized FairnessCircuitBreaker "
            f"(threshold={config.critical_threshold}, "
            f"intervention={config.intervention_type.value})"
        )
    
    def record_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Optional[CircuitBreakerEvent]:
        """
        Record fairness metrics and check for violations.
        
        Args:
            metrics: Current fairness metrics
            timestamp: Timestamp of metrics
        
        Returns:
            CircuitBreakerEvent if state changed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store in buffer
        self.metrics_buffer.append({
            'timestamp': timestamp,
            **metrics
        })
        
        # Keep only recent history
        if len(self.metrics_buffer) > 1000:
            self.metrics_buffer = self.metrics_buffer[-1000:]
        
        # Check current state
        if self.state == CircuitState.CLOSED:
            return self._check_for_violations(metrics, timestamp)
        
        elif self.state == CircuitState.HALF_OPEN:
            return self._check_recovery(metrics, timestamp)
        
        elif self.state == CircuitState.OPEN:
            # Check if cooldown expired
            if self._should_attempt_recovery(timestamp):
                return self._transition_to_half_open(timestamp)
        
        return None
    
    def _check_for_violations(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
    ) -> Optional[CircuitBreakerEvent]:
        """Check if metrics violate thresholds."""
        violations = []
        
        for metric_name, value in metrics.items():
            if value > self.config.critical_threshold:
                violations.append((metric_name, value))
        
        if violations:
            self.consecutive_failures += 1
            self.last_failure_time = timestamp
            
            logger.warning(
                f"Fairness violation detected: {violations} "
                f"({self.consecutive_failures}/{self.config.failure_count_threshold})"
            )
            
            # Open circuit if threshold reached
            if self.consecutive_failures >= self.config.failure_count_threshold:
                return self._open_circuit(violations[0], timestamp)
        else:
            # Reset failure counter
            self.consecutive_failures = 0
        
        return None
    
    def _open_circuit(
        self,
        violation: tuple,
        timestamp: datetime
    ) -> CircuitBreakerEvent:
        """Open circuit breaker."""
        metric_name, metric_value = violation
        
        event = CircuitBreakerEvent(
            event_id=f"breaker_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            event_type='opened',
            trigger_metric=metric_name,
            trigger_value=metric_value,
            threshold=self.config.critical_threshold,
            state_before=self.state,
            state_after=CircuitState.OPEN,
            intervention_taken=self.config.intervention_type,
            metadata={
                'consecutive_failures': self.consecutive_failures,
                'baseline_model_id': self.config.baseline_model_id
            }
        )
        
        self.state = CircuitState.OPEN
        self.last_state_change = timestamp
        self.events.append(event)
        
        logger.critical(
            f"🔴 CIRCUIT BREAKER OPENED: {metric_name}={metric_value:.3f} "
            f"(threshold={self.config.critical_threshold:.3f}). "
            f"Intervention: {self.config.intervention_type.value}"
        )
        
        # Notify
        if self.notification_handler:
            self.notification_handler(event)
        
        return event
    
    def _should_attempt_recovery(self, timestamp: datetime) -> bool:
        """Check if should attempt recovery."""
        if not self.config.auto_recovery:
            return False
        
        if self.last_state_change is None:
            return False
        
        elapsed = (timestamp - self.last_state_change).total_seconds()
        
        return elapsed >= self.config.cooldown_period_seconds
    
    def _transition_to_half_open(
        self,
        timestamp: datetime
    ) -> CircuitBreakerEvent:
        """Transition to half-open state to test recovery."""
        event = CircuitBreakerEvent(
            event_id=f"breaker_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            event_type='half_open',
            trigger_metric='cooldown_expired',
            trigger_value=0.0,
            threshold=0.0,
            state_before=self.state,
            state_after=CircuitState.HALF_OPEN,
            metadata={'testing_recovery': True}
        )
        
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = timestamp
        self.events.append(event)
        
        logger.info("Circuit breaker entering HALF_OPEN state - testing recovery")
        
        return event
    
    def _check_recovery(
        self,
        metrics: Dict[str, float],
        timestamp: datetime
    ) -> Optional[CircuitBreakerEvent]:
        """Check if system has recovered."""
        # Get recent metrics from buffer
        recent = [
            m for m in self.metrics_buffer
            if (timestamp - m['timestamp']).total_seconds() < 
               self.config.recovery_check_interval
        ]
        
        if len(recent) < self.config.recovery_sample_size:
            # Not enough samples yet
            return None
        
        # Check if all recent metrics are below recovery threshold
        all_good = True
        for metric_dict in recent:
            for metric_name, value in metric_dict.items():
                if metric_name == 'timestamp':
                    continue
                if value > self.config.recovery_threshold:
                    all_good = False
                    break
            if not all_good:
                break
        
        if all_good:
            # Recovery successful - close circuit
            return self._close_circuit(timestamp)
        else:
            # Recovery failed - reopen
            return self._reopen_circuit(timestamp)
    
    def _close_circuit(self, timestamp: datetime) -> CircuitBreakerEvent:
        """Close circuit breaker (normal operation)."""
        event = CircuitBreakerEvent(
            event_id=f"breaker_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            event_type='closed',
            trigger_metric='recovery_verified',
            trigger_value=0.0,
            threshold=self.config.recovery_threshold,
            state_before=self.state,
            state_after=CircuitState.CLOSED,
            metadata={'recovery_successful': True}
        )
        
        self.state = CircuitState.CLOSED
        self.last_state_change = timestamp
        self.consecutive_failures = 0
        self.events.append(event)
        
        logger.info("✅ Circuit breaker CLOSED - system recovered")
        
        if self.notification_handler:
            self.notification_handler(event)
        
        return event
    
    def _reopen_circuit(self, timestamp: datetime) -> CircuitBreakerEvent:
        """Reopen circuit after failed recovery attempt."""
        event = CircuitBreakerEvent(
            event_id=f"breaker_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            event_type='reopened',
            trigger_metric='recovery_failed',
            trigger_value=0.0,
            threshold=0.0,
            state_before=self.state,
            state_after=CircuitState.OPEN,
            metadata={'recovery_failed': True}
        )
        
        self.state = CircuitState.OPEN
        self.last_state_change = timestamp
        self.events.append(event)
        
        logger.warning("⚠️ Circuit breaker REOPENED - recovery failed")
        
        return event
    
    def should_use_baseline(self) -> bool:
        """
        Check if should route to baseline model.
        
        Returns:
            True if circuit is open/half-open and intervention is baseline routing
        """
        if self.state == CircuitState.CLOSED:
            return False
        
        return self.config.intervention_type == InterventionType.ROUTE_TO_BASELINE
    
    def get_traffic_allocation(self) -> Dict[str, float]:
        """
        Get traffic allocation based on circuit state.
        
        Returns:
            Dictionary with 'production' and 'baseline' percentages
        """
        if self.state == CircuitState.CLOSED:
            return {'production': 1.0, 'baseline': 0.0}
        
        elif self.state == CircuitState.HALF_OPEN:
            # Send 20% to production for testing
            return {'production': 0.2, 'baseline': 0.8}
        
        elif self.state == CircuitState.OPEN:
            if self.config.intervention_type == InterventionType.ROUTE_TO_BASELINE:
                return {'production': 0.0, 'baseline': 1.0}
            elif self.config.intervention_type == InterventionType.REDUCE_TRAFFIC:
                return {'production': 0.1, 'baseline': 0.9}
            else:
                return {'production': 0.0, 'baseline': 1.0}
    
    def manual_override(
        self,
        new_state: CircuitState,
        reason: str
    ) -> CircuitBreakerEvent:
        """
        Manually override circuit state.
        
        Args:
            new_state: Desired state
            reason: Reason for override
        
        Returns:
            Event record
        """
        timestamp = datetime.now()
        
        event = CircuitBreakerEvent(
            event_id=f"breaker_manual_{timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=timestamp,
            event_type='manual_override',
            trigger_metric='manual',
            trigger_value=0.0,
            threshold=0.0,
            state_before=self.state,
            state_after=new_state,
            metadata={'reason': reason, 'manual': True}
        )
        
        self.state = new_state
        self.last_state_change = timestamp
        self.events.append(event)
        
        logger.warning(
            f"Manual circuit breaker override: {new_state.value} - {reason}"
        )
        
        if self.notification_handler:
            self.notification_handler(event)
        
        return event
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            'state': self.state.value,
            'consecutive_failures': self.consecutive_failures,
            'last_failure_time': self.last_failure_time,
            'last_state_change': self.last_state_change,
            'using_baseline': self.should_use_baseline(),
            'traffic_allocation': self.get_traffic_allocation(),
            'total_events': len(self.events),
            'config': {
                'critical_threshold': self.config.critical_threshold,
                'intervention_type': self.config.intervention_type.value,
                'auto_recovery': self.config.auto_recovery
            }
        }
    
    def get_event_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get circuit breaker event history.
        
        Args:
            limit: Maximum number of events to return
        
        Returns:
            List of event dictionaries
        """
        events = self.events[-limit:] if limit else self.events
        
        return [
            {
                'event_id': e.event_id,
                'timestamp': e.timestamp.isoformat(),
                'event_type': e.event_type,
                'trigger_metric': e.trigger_metric,
                'trigger_value': e.trigger_value,
                'state_transition': f"{e.state_before.value} -> {e.state_after.value}",
                'intervention': e.intervention_taken.value if e.intervention_taken else None,
                'metadata': e.metadata
            }
            for e in events
        ]


class CircuitBreakerMonitor:
    """
    Monitor multiple circuit breakers across different models/services.
    
    Provides centralized monitoring and coordination.
    """
    
    def __init__(self):
        """Initialize circuit breaker monitor."""
        self.breakers: Dict[str, FairnessCircuitBreaker] = {}
        logger.info("Initialized CircuitBreakerMonitor")
    
    def register_breaker(
        self,
        name: str,
        breaker: FairnessCircuitBreaker
    ):
        """Register a circuit breaker."""
        self.breakers[name] = breaker
        logger.info(f"Registered circuit breaker: {name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        status = {
            'timestamp': datetime.now().isoformat(),
            'total_breakers': len(self.breakers),
            'breakers': {}
        }
        
        # Count states
        state_counts = {'CLOSED': 0, 'OPEN': 0, 'HALF_OPEN': 0}
        
        for name, breaker in self.breakers.items():
            breaker_status = breaker.get_status()
            status['breakers'][name] = breaker_status
            state_counts[breaker_status['state']] += 1
        
        status['state_summary'] = state_counts
        status['healthy'] = state_counts['OPEN'] == 0
        
        return status
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get alerts from all breakers."""
        alerts = []
        
        for name, breaker in self.breakers.items():
            if breaker.state != CircuitState.CLOSED:
                alerts.append({
                    'breaker_name': name,
                    'state': breaker.state.value,
                    'severity': 'CRITICAL' if breaker.state == CircuitState.OPEN else 'WARNING',
                    'last_change': breaker.last_state_change,
                    'consecutive_failures': breaker.consecutive_failures
                })
        
        return alerts