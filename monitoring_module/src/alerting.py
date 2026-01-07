"""
Alerting System for Fairness Monitoring

Provides intelligent alerting for fairness violations with severity
classification, priority scoring, and notification management.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from shared.logging import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertType(Enum):
    """Types of fairness alerts."""
    THRESHOLD_VIOLATION = "threshold_violation"
    DRIFT_DETECTED = "drift_detected"
    RAPID_DEGRADATION = "rapid_degradation"
    CONSISTENT_BIAS = "consistent_bias"
    INTERSECTIONAL_ISSUE = "intersectional_issue"


@dataclass
class FairnessAlert:
    """
    Represents a fairness alert.
    
    Attributes:
        alert_id: Unique identifier
        timestamp: When alert was triggered
        alert_type: Type of alert
        severity: Severity level
        metric_name: Affected fairness metric
        current_value: Current metric value
        threshold: Expected threshold
        affected_groups: Groups affected
        message: Human-readable description
        evidence: Supporting data
        recommended_actions: Suggested remediation steps
        priority_score: Numerical priority (0-100)
    """
    
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    metric_name: str
    current_value: float
    threshold: float
    affected_groups: List[str]
    message: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommended_actions: List[str] = field(default_factory=list)
    priority_score: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'affected_groups': self.affected_groups,
            'message': self.message,
            'evidence': self.evidence,
            'recommended_actions': self.recommended_actions,
            'priority_score': self.priority_score,
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"[{self.severity.value}] {self.alert_type.value}: "
            f"{self.metric_name}={self.current_value:.3f} "
            f"(threshold={self.threshold:.3f})"
        )


class ThresholdAlertSystem:
    """
    Threshold-based alerting for fairness metrics.
    
    Triggers alerts when metrics exceed configured thresholds.
    """
    
    def __init__(
        self,
        thresholds: Dict[str, float],
        severity_rules: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """
        Initialize alert system.
        
        Args:
            thresholds: Dictionary mapping metric names to threshold values
            severity_rules: Rules for severity classification
                Example: {
                    'demographic_parity': {
                        'critical': 0.20,
                        'high': 0.15,
                        'medium': 0.10,
                        'low': 0.05
                    }
                }
        """
        self.thresholds = thresholds
        self.severity_rules = severity_rules or self._default_severity_rules()
        
        logger.info(
            f"Initialized ThresholdAlertSystem with {len(thresholds)} thresholds"
        )
    
    def check_thresholds(
        self,
        metrics: Dict[str, float],
        group_metrics: Optional[Dict[str, Dict[str, float]]] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[FairnessAlert]:
        """
        Check if any metrics exceed thresholds.
        
        Args:
            metrics: Current metric values
            group_metrics: Per-group metric values
            timestamp: Current timestamp
        
        Returns:
            FairnessAlert if threshold violated, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, current_value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            threshold = self.thresholds[metric_name]
            
            # Check violation
            if current_value > threshold:
                severity = self._determine_severity(metric_name, current_value)
                
                # Determine affected groups
                affected_groups = []
                if group_metrics and metric_name in group_metrics:
                    affected_groups = self._identify_affected_groups(
                        group_metrics[metric_name], threshold
                    )
                
                # Create alert
                alert = self._create_threshold_alert(
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    severity=severity,
                    affected_groups=affected_groups,
                    timestamp=timestamp
                )
                
                logger.warning(f"Threshold alert: {alert}")
                return alert
        
        return None
    
    def _default_severity_rules(self) -> Dict[str, Dict[str, float]]:
        """Default severity classification rules."""
        default_rule = {
            'critical': 0.25,
            'high': 0.20,
            'medium': 0.15,
            'low': 0.10,
        }
        
        return {
            metric: default_rule.copy()
            for metric in self.thresholds.keys()
        }
    
    def _determine_severity(
        self,
        metric_name: str,
        value: float
    ) -> AlertSeverity:
        """Determine alert severity based on value."""
        if metric_name not in self.severity_rules:
            return AlertSeverity.MEDIUM
        
        rules = self.severity_rules[metric_name]
        
        if value >= rules.get('critical', float('inf')):
            return AlertSeverity.CRITICAL
        elif value >= rules.get('high', float('inf')):
            return AlertSeverity.HIGH
        elif value >= rules.get('medium', float('inf')):
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _identify_affected_groups(
        self,
        group_values: Dict[str, float],
        threshold: float
    ) -> List[str]:
        """Identify which groups are affected."""
        return [
            group for group, value in group_values.items()
            if value > threshold
        ]
    
    def _create_threshold_alert(
        self,
        metric_name: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity,
        affected_groups: List[str],
        timestamp: datetime
    ) -> FairnessAlert:
        """Create threshold violation alert."""
        excess = current_value - threshold
        excess_pct = (excess / threshold) * 100
        
        message = (
            f"Fairness violation detected: {metric_name} = {current_value:.3f} "
            f"exceeds threshold of {threshold:.3f} by {excess:.3f} ({excess_pct:.1f}%)"
        )
        
        if affected_groups:
            message += f". Affected groups: {', '.join(affected_groups)}"
        
        # Recommended actions
        actions = [
            f"Review model predictions for {metric_name}",
            "Check for data drift in protected attributes",
            "Analyze recent prediction patterns",
        ]
        
        if severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            actions.append("Consider implementing bias mitigation")
            actions.append("Notify stakeholders immediately")
        
        # Priority score
        priority_score = self._calculate_priority(
            severity, excess_pct, len(affected_groups)
        )
        
        alert = FairnessAlert(
            alert_id=self._generate_alert_id(timestamp, metric_name),
            timestamp=timestamp,
            alert_type=AlertType.THRESHOLD_VIOLATION,
            severity=severity,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            affected_groups=affected_groups,
            message=message,
            evidence={
                'excess': excess,
                'excess_percentage': excess_pct,
                'group_count': len(affected_groups),
            },
            recommended_actions=actions,
            priority_score=priority_score,
        )
        
        return alert
    
    def _calculate_priority(
        self,
        severity: AlertSeverity,
        excess_pct: float,
        n_groups: int
    ) -> float:
        """Calculate priority score (0-100)."""
        # Base score from severity
        severity_scores = {
            AlertSeverity.CRITICAL: 90,
            AlertSeverity.HIGH: 70,
            AlertSeverity.MEDIUM: 50,
            AlertSeverity.LOW: 30,
            AlertSeverity.INFO: 10,
        }
        
        base_score = severity_scores[severity]
        
        # Adjust for excess magnitude
        magnitude_adjustment = min(excess_pct / 10, 10)  # Max +10
        
        # Adjust for number of affected groups
        group_adjustment = min(n_groups * 2, 10)  # Max +10
        
        priority = min(base_score + magnitude_adjustment + group_adjustment, 100)
        
        return priority
    
    def _generate_alert_id(self, timestamp: datetime, metric: str) -> str:
        """Generate unique alert ID."""
        return f"alert_{metric}_{timestamp.strftime('%Y%m%d_%H%M%S')}"


class AdaptiveAlertSystem:
    """
    Adaptive alerting with dynamic threshold adjustment.
    
    Adjusts thresholds based on historical false positive rates.
    """
    
    def __init__(
        self,
        initial_thresholds: Dict[str, float],
        target_fpr: float = 0.05,
        adaptation_window: int = 100
    ):
        """
        Initialize adaptive alert system.
        
        Args:
            initial_thresholds: Initial threshold values
            target_fpr: Target false positive rate
            adaptation_window: Window for computing FPR
        """
        self.thresholds = initial_thresholds.copy()
        self.target_fpr = target_fpr
        self.adaptation_window = adaptation_window
        
        # Track alert history
        self.alert_history: List[Dict[str, Any]] = []
        self.feedback_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized AdaptiveAlertSystem (target_fpr={target_fpr})"
        )
    
    def check_and_adapt(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Optional[FairnessAlert]:
        """
        Check thresholds and adapt based on history.
        
        Args:
            metrics: Current metric values
            timestamp: Current timestamp
        
        Returns:
            FairnessAlert if threshold violated
        """
        # Check thresholds
        base_system = ThresholdAlertSystem(self.thresholds)
        alert = base_system.check_thresholds(metrics, timestamp=timestamp)
        
        # Record alert
        if alert:
            self.alert_history.append({
                'timestamp': timestamp or datetime.now(),
                'metric': alert.metric_name,
                'value': alert.current_value,
                'threshold': alert.threshold,
                'alerted': True,
            })
        
        # Adapt thresholds periodically
        if len(self.alert_history) >= self.adaptation_window:
            self._adapt_thresholds()
        
        return alert
    
    def provide_feedback(
        self,
        alert_id: str,
        is_true_positive: bool,
        timestamp: Optional[datetime] = None
    ):
        """
        Provide feedback on alert accuracy.
        
        Args:
            alert_id: Alert identifier
            is_true_positive: Whether alert was a true positive
            timestamp: Feedback timestamp
        """
        self.feedback_history.append({
            'alert_id': alert_id,
            'is_true_positive': is_true_positive,
            'timestamp': timestamp or datetime.now(),
        })
        
        logger.info(
            f"Feedback recorded for {alert_id}: "
            f"{'TP' if is_true_positive else 'FP'}"
        )
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on false positive rate."""
        # Compute FPR for each metric
        for metric in self.thresholds.keys():
            metric_alerts = [
                a for a in self.alert_history[-self.adaptation_window:]
                if a.get('metric') == metric and a.get('alerted')
            ]
            
            if not metric_alerts:
                continue
            
            # Get corresponding feedback
            alert_ids = [a.get('alert_id') for a in metric_alerts]
            feedback = [
                f for f in self.feedback_history
                if f.get('alert_id') in alert_ids
            ]
            
            if not feedback:
                continue
            
            # Compute FPR
            n_false_positives = sum(
                1 for f in feedback if not f['is_true_positive']
            )
            fpr = n_false_positives / len(feedback)
            
            # Adjust threshold
            if fpr > self.target_fpr:
                # Too many false positives - increase threshold
                adjustment = 1.1
                self.thresholds[metric] *= adjustment
                logger.info(
                    f"Increasing {metric} threshold by 10% "
                    f"(FPR={fpr:.3f} > target={self.target_fpr:.3f})"
                )
            elif fpr < self.target_fpr / 2:
                # Too few alerts - decrease threshold
                adjustment = 0.95
                self.thresholds[metric] *= adjustment
                logger.info(
                    f"Decreasing {metric} threshold by 5% "
                    f"(FPR={fpr:.3f} < target/2={self.target_fpr/2:.3f})"
                )


class AlertAggregator:
    """
    Aggregates and prioritizes multiple alerts.
    
    Prevents alert fatigue by grouping related alerts and
    prioritizing based on severity and recency.
    """
    
    def __init__(
        self,
        grouping_window_minutes: int = 60,
        max_alerts_per_window: int = 5
    ):
        """
        Initialize aggregator.
        
        Args:
            grouping_window_minutes: Time window for grouping
            max_alerts_per_window: Maximum alerts to emit per window
        """
        self.grouping_window = pd.Timedelta(minutes=grouping_window_minutes)
        self.max_alerts = max_alerts_per_window
        
        self.alert_buffer: List[FairnessAlert] = []
        self.last_emission_time: Optional[datetime] = None
        
        logger.info(
            f"Initialized AlertAggregator "
            f"(window={grouping_window_minutes}m, max={max_alerts_per_window})"
        )
    
    def add_alert(self, alert: FairnessAlert) -> List[FairnessAlert]:
        """
        Add alert to buffer and return alerts to emit.
        
        Args:
            alert: New alert
        
        Returns:
            List of alerts to emit (may be empty)
        """
        self.alert_buffer.append(alert)
        
        # Check if should emit
        if self.last_emission_time is None:
            self.last_emission_time = alert.timestamp
            return []
        
        time_since_last = alert.timestamp - self.last_emission_time
        
        if time_since_last >= self.grouping_window:
            # Emit aggregated alerts
            alerts_to_emit = self._aggregate_and_prioritize()
            self.last_emission_time = alert.timestamp
            self.alert_buffer = []
            return alerts_to_emit
        
        return []
    
    def _aggregate_and_prioritize(self) -> List[FairnessAlert]:
        """Aggregate buffer and return top priority alerts."""
        if not self.alert_buffer:
            return []
        
        # Group by metric
        grouped = {}
        for alert in self.alert_buffer:
            key = alert.metric_name
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(alert)
        
        # Create aggregated alerts
        aggregated = []
        for metric, alerts in grouped.items():
            if len(alerts) == 1:
                aggregated.append(alerts[0])
            else:
                # Merge multiple alerts for same metric
                merged = self._merge_alerts(alerts)
                aggregated.append(merged)
        
        # Sort by priority
        aggregated.sort(key=lambda a: a.priority_score, reverse=True)
        
        # Return top N
        return aggregated[:self.max_alerts]
    
    def _merge_alerts(self, alerts: List[FairnessAlert]) -> FairnessAlert:
        """Merge multiple alerts for same metric."""
        # Use highest severity
        max_severity = max(a.severity for a in alerts)
        
        # Average value
        avg_value = np.mean([a.current_value for a in alerts])
        
        # Combine affected groups
        all_groups = set()
        for a in alerts:
            all_groups.update(a.affected_groups)
        
        # Latest timestamp
        latest_timestamp = max(a.timestamp for a in alerts)
        
        # Highest priority
        max_priority = max(a.priority_score for a in alerts)
        
        message = (
            f"Aggregated alert: {len(alerts)} {alerts[0].metric_name} "
            f"violations detected in past hour. "
            f"Average value: {avg_value:.3f}"
        )
        
        return FairnessAlert(
            alert_id=f"aggregated_{alerts[0].metric_name}_{latest_timestamp.strftime('%Y%m%d_%H%M%S')}",
            timestamp=latest_timestamp,
            alert_type=AlertType.CONSISTENT_BIAS,
            severity=max_severity,
            metric_name=alerts[0].metric_name,
            current_value=avg_value,
            threshold=alerts[0].threshold,
            affected_groups=list(all_groups),
            message=message,
            evidence={
                'alert_count': len(alerts),
                'value_range': (
                    min(a.current_value for a in alerts),
                    max(a.current_value for a in alerts)
                ),
            },
            recommended_actions=alerts[0].recommended_actions,
            priority_score=max_priority,
        )


class AlertNotifier:
    """
    Notification system for fairness alerts.
    
    Supports multiple notification channels with routing logic.
    """
    
    def __init__(self):
        """Initialize notifier."""
        self.handlers: Dict[str, Callable] = {}
        self.routing_rules: List[Dict[str, Any]] = []
        
        logger.info("Initialized AlertNotifier")
    
    def register_handler(
        self,
        channel: str,
        handler: Callable[[FairnessAlert], None]
    ):
        """
        Register notification handler.
        
        Args:
            channel: Channel name (e.g., 'email', 'slack', 'pagerduty')
            handler: Function to handle notification
        """
        self.handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel}")
    
    def add_routing_rule(
        self,
        severity_threshold: AlertSeverity,
        channels: List[str]
    ):
        """
        Add routing rule.
        
        Args:
            severity_threshold: Minimum severity to trigger
            channels: List of channels to notify
        """
        self.routing_rules.append({
            'severity': severity_threshold,
            'channels': channels,
        })
        logger.info(
            f"Added routing rule: {severity_threshold.value} -> {channels}"
        )
    
    def notify(self, alert: FairnessAlert):
        """
        Send notifications for alert.
        
        Args:
            alert: Alert to notify about
        """
        # Determine channels
        channels = self._determine_channels(alert)
        
        if not channels:
            logger.info(f"No channels configured for {alert.severity.value}")
            return
        
        # Send to each channel
        for channel in channels:
            if channel not in self.handlers:
                logger.warning(f"No handler registered for channel: {channel}")
                continue
            
            try:
                self.handlers[channel](alert)
                logger.info(f"Notification sent to {channel}: {alert.alert_id}")
            except Exception as e:
                logger.error(f"Failed to send to {channel}: {e}")
    
    def _determine_channels(self, alert: FairnessAlert) -> List[str]:
        """Determine which channels to notify based on routing rules."""
        severity_order = {
            AlertSeverity.CRITICAL: 5,
            AlertSeverity.HIGH: 4,
            AlertSeverity.MEDIUM: 3,
            AlertSeverity.LOW: 2,
            AlertSeverity.INFO: 1,
        }
        
        alert_severity_level = severity_order[alert.severity]
        
        channels = set()
        for rule in self.routing_rules:
            rule_severity_level = severity_order[rule['severity']]
            
            if alert_severity_level >= rule_severity_level:
                channels.update(rule['channels'])
        
        return list(channels)


# Example notification handlers
def log_handler(alert: FairnessAlert):
    """Simple logging handler."""
    logger.warning(f"ALERT: {alert}")


def console_handler(alert: FairnessAlert):
    """Console output handler."""
    print("\n" + "=" * 60)
    print(f"🚨 FAIRNESS ALERT [{alert.severity.value}]")
    print("=" * 60)
    print(f"Type: {alert.alert_type.value}")
    print(f"Metric: {alert.metric_name}")
    print(f"Value: {alert.current_value:.3f} (threshold: {alert.threshold:.3f})")
    print(f"Time: {alert.timestamp}")
    print(f"\nMessage: {alert.message}")
    
    if alert.affected_groups:
        print(f"\nAffected Groups: {', '.join(alert.affected_groups)}")
    
    if alert.recommended_actions:
        print("\nRecommended Actions:")
        for i, action in enumerate(alert.recommended_actions, 1):
            print(f"  {i}. {action}")
    
    print("=" * 60 + "\n")