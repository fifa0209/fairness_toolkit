"""
Drift Detection - Detect changes in fairness metrics over time.

Uses statistical tests to identify when fairness degrades in production.
48-hour scope: Kolmogorov-Smirnov test for distribution comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from datetime import datetime, timedelta

from shared.logging import get_logger
from shared.schemas import MonitoringAlert

logger = get_logger(__name__)


class FairnessDriftDetector:
    """
    Detect drift in fairness metrics using statistical tests.
    
    Compares recent predictions to a reference period to detect
    significant changes in fairness.
    
    Example:
        >>> detector = FairnessDriftDetector(
        ...     reference_window=7,  # days
        ...     detection_window=1    # day
        ... )
        >>> 
        >>> # Set reference period
        >>> detector.set_reference(
        ...     y_true_ref, y_pred_ref, sensitive_ref
        ... )
        >>> 
        >>> # Check for drift
        >>> drift_result = detector.detect_drift(
        ...     y_true_recent, y_pred_recent, sensitive_recent
        ... )
        >>> 
        >>> if drift_result['drift_detected']:
        ...     print(f"Drift in: {drift_result['drifted_metrics']}")
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        test_method: str = 'ks',
        min_samples: int = 100,
    ):
        """
        Initialize drift detector.
        
        Args:
            alpha: Significance level for statistical tests
            test_method: 'ks' (Kolmogorov-Smirnov) or 'chi2' (Chi-square)
            min_samples: Minimum samples per group for reliable test
        """
        self.alpha = alpha
        self.test_method = test_method
        self.min_samples = min_samples
        
        # Reference data
        self.reference_metrics = None
        self.reference_predictions = None
        
        logger.info(
            f"FairnessDriftDetector initialized: "
            f"alpha={alpha}, method={test_method}"
        )
    
    def set_reference(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> None:
        """
        Set reference period data.
        
        Args:
            y_true: Reference true labels
            y_pred: Reference predictions
            sensitive_features: Reference protected attribute
        """
        from measurement_module.src.metrics_engine import compute_metric
        
        # Store reference predictions
        self.reference_predictions = {
            'y_true': y_true,
            'y_pred': y_pred,
            'sensitive': sensitive_features,
        }
        
        # Compute reference metrics
        self.reference_metrics = {}
        
        for metric_name in ['demographic_parity', 'equalized_odds']:
            try:
                value, group_metrics, _ = compute_metric(
                    metric_name, y_true, y_pred, sensitive_features
                )
                self.reference_metrics[metric_name] = {
                    'value': value,
                    'group_metrics': group_metrics,
                }
            except Exception as e:
                logger.error(f"Failed to compute reference {metric_name}: {e}")
        
        logger.info(f"Reference period set: {len(y_true)} samples")
    
    def detect_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict:
        """
        Detect drift compared to reference period.
        
        Args:
            y_true: Current true labels
            y_pred: Current predictions
            sensitive_features: Current protected attribute
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_metrics is None:
            raise ValueError("Must call set_reference() before detect_drift()")
        
        from measurement_module.src.metrics_engine import compute_metric
        
        results = {
            'timestamp': datetime.now(),
            'drift_detected': False,
            'drifted_metrics': [],
            'tests': {},
        }
        
        # Compute current metrics
        for metric_name in self.reference_metrics.keys():
            try:
                current_value, current_group_metrics, _ = compute_metric(
                    metric_name, y_true, y_pred, sensitive_features
                )
                
                reference_value = self.reference_metrics[metric_name]['value']
                
                # Statistical test for drift
                drift_test = self._test_metric_drift(
                    metric_name,
                    current_value,
                    reference_value,
                    y_pred,
                    sensitive_features,
                )
                
                results['tests'][metric_name] = drift_test
                
                if drift_test['significant']:
                    results['drift_detected'] = True
                    results['drifted_metrics'].append(metric_name)
                    
                    logger.warning(
                        f"Drift detected in {metric_name}: "
                        f"{reference_value:.3f} → {current_value:.3f} "
                        f"(p={drift_test['p_value']:.4f})"
                    )
            
            except Exception as e:
                logger.error(f"Drift detection failed for {metric_name}: {e}")
        
        return results
    
    def _test_metric_drift(
        self,
        metric_name: str,
        current_value: float,
        reference_value: float,
        current_predictions: np.ndarray,
        current_sensitive: np.ndarray,
    ) -> Dict:
        """
        Test if metric has drifted significantly.
        
        Uses KS test to compare prediction distributions between groups.
        """
        if self.test_method != 'ks':
            raise NotImplementedError(f"Method {self.test_method} not implemented")
        
        # Get reference predictions
        ref_pred = self.reference_predictions['y_pred']
        ref_sensitive = self.reference_predictions['sensitive']
        
        # Compare distributions per group
        groups = np.unique(current_sensitive)
        p_values = []
        
        for group in groups:
            # Reference group predictions
            ref_mask = ref_sensitive == group
            ref_group_pred = ref_pred[ref_mask]
            
            # Current group predictions
            cur_mask = current_sensitive == group
            cur_group_pred = current_predictions[cur_mask]
            
            if len(ref_group_pred) < self.min_samples or len(cur_group_pred) < self.min_samples:
                logger.warning(f"Insufficient samples for group {group}")
                continue
            
            # KS test
            statistic, p_value = stats.ks_2samp(ref_group_pred, cur_group_pred)
            p_values.append(p_value)
        
        # Overall p-value (minimum across groups)
        min_p_value = min(p_values) if p_values else 1.0
        
        return {
            'current_value': current_value,
            'reference_value': reference_value,
            'change': current_value - reference_value,
            'p_value': min_p_value,
            'significant': min_p_value < self.alpha,
            'test': 'ks_2samp',
        }
    
    def create_alert(
        self,
        drift_result: Dict,
        severity: str = 'HIGH',
    ) -> Optional[MonitoringAlert]:
        """
        Create monitoring alert from drift detection result.
        
        Args:
            drift_result: Result from detect_drift()
            severity: Alert severity level
            
        Returns:
            MonitoringAlert or None if no drift
        """
        if not drift_result['drift_detected']:
            return None
        
        affected_metrics = drift_result['drifted_metrics']
        
        # Build message
        message_parts = []
        for metric in affected_metrics:
            test = drift_result['tests'][metric]
            message_parts.append(
                f"{metric}: {test['reference_value']:.3f} → {test['current_value']:.3f} "
                f"(p={test['p_value']:.4f})"
            )
        
        message = "Fairness drift detected. " + "; ".join(message_parts)
        
        alert = MonitoringAlert(
            alert_type='drift',
            severity=severity,
            metric_name=', '.join(affected_metrics),
            current_value=max(
                drift_result['tests'][m]['current_value']
                for m in affected_metrics
            ),
            reference_value=None,
            affected_groups=affected_metrics,
            message=message,
        )
        
        return alert


class ThresholdAlertSystem:
    """
    Simple threshold-based alerting.
    
    Triggers alerts when metrics exceed predefined thresholds.
    Simpler than drift detection - useful for absolute fairness requirements.
    
    Example:
        >>> alerter = ThresholdAlertSystem(
        ...     thresholds={'demographic_parity': 0.1}
        ... )
        >>> 
        >>> alert = alerter.check_thresholds({
        ...     'demographic_parity': 0.15
        ... })
        >>> 
        >>> if alert:
        ...     send_alert(alert)
    """
    
    def __init__(
        self,
        thresholds: Dict[str, float] = None,
        severity_levels: Dict[str, Dict[str, float]] = None,
    ):
        """
        Initialize alert system.
        
        Args:
            thresholds: Dict of metric -> threshold
            severity_levels: Dict of severity -> threshold multipliers
        """
        self.thresholds = thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds': 0.1,
        }
        
        self.severity_levels = severity_levels or {
            'LOW': 1.0,      # At threshold
            'HIGH': 1.5,     # 50% over threshold
            'CRITICAL': 2.0, # 100% over threshold
        }
    
    def check_thresholds(
        self,
        metrics: Dict[str, float],
        group_sizes: Optional[Dict[str, int]] = None,
    ) -> Optional[MonitoringAlert]:
        """
        Check if any metrics exceed thresholds.
        
        Args:
            metrics: Dict of metric_name -> value
            group_sizes: Dict of group -> sample size (optional)
            
        Returns:
            MonitoringAlert or None if all thresholds pass
        """
        violations = []
        max_severity = None
        max_violation_ratio = 0
        
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            threshold = self.thresholds[metric_name]
            
            if value > threshold:
                violation_ratio = value / threshold
                
                # Determine severity
                severity = 'LOW'
                for sev_name, sev_multiplier in sorted(
                    self.severity_levels.items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    if violation_ratio >= sev_multiplier:
                        severity = sev_name
                        break
                
                violations.append({
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'severity': severity,
                })
                
                if violation_ratio > max_violation_ratio:
                    max_violation_ratio = violation_ratio
                    max_severity = severity
        
        if not violations:
            return None
        
        # Create alert
        message = f"Threshold violations detected: "
        message += ", ".join([
            f"{v['metric']}={v['value']:.3f} (threshold={v['threshold']:.3f})"
            for v in violations
        ])
        
        alert = MonitoringAlert(
            alert_type='threshold_violation',
            severity=max_severity,
            metric_name=', '.join([v['metric'] for v in violations]),
            current_value=max([v['value'] for v in violations]),
            reference_value=None,
            affected_groups=[v['metric'] for v in violations],
            message=message,
        )
        
        return alert