"""
Simple Fairness Analyzer - Minimal wrapper for demo compatibility.

This is a simplified version that works without the shared module dependencies.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

# Use relative imports
from .metrics_engine import (
    compute_metric as _compute_metric,
    interpret_metric,
    compute_group_metrics,
)


@dataclass
class MetricResult:
    """Result of a fairness metric computation."""
    metric_name: str
    value: float
    is_fair: bool
    threshold: float
    group_metrics: Dict[str, float]
    confidence_interval: Optional[tuple] = None
    p_value: Optional[float] = None
    
    def __repr__(self):
        status = "FAIR" if self.is_fair else "UNFAIR"
        ci_str = ""
        if self.confidence_interval:
            ci_str = f", CI=[{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]"
        return (
            f"MetricResult(metric={self.metric_name}, value={self.value:.4f}, "
            f"status={status}{ci_str})"
        )


class FairnessAnalyzer:
    """
    Simple fairness metrics analyzer with bootstrap confidence intervals.
    
    Args:
        bootstrap_samples: Number of bootstrap samples for CI computation
        random_state: Random seed for reproducibility
    """
    
    def __init__(
        self,
        bootstrap_samples: int = 1000,
        random_state: Optional[int] = None,
    ):
        self.bootstrap_samples = bootstrap_samples
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def compute_metric(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        metric: str = 'demographic_parity',
        threshold: float = 0.1,
        compute_ci: bool = True,
        ci_alpha: float = 0.05,
    ) -> MetricResult:
        """
        Compute fairness metric with optional bootstrap confidence interval.
        
        Args:
            y_true: True labels (binary: 0 or 1)
            y_pred: Predicted labels (binary: 0 or 1)
            sensitive_features: Protected attribute (binary: 0 or 1)
            metric: Metric name ('demographic_parity', 'equalized_odds', 'equal_opportunity')
            threshold: Fairness threshold (default: 0.1)
            compute_ci: Whether to compute bootstrap CI (default: True)
            ci_alpha: Significance level for CI (default: 0.05 for 95% CI)
            
        Returns:
            MetricResult with computed metric, fairness status, and optional CI
        """
        # Convert to numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)
        
        # Compute point estimate
        metric_value, group_metrics, group_sizes = _compute_metric(
            metric, y_true, y_pred, sensitive_features, allow_multiple_groups=True
        )
        
        # Determine fairness status
        is_fair = metric_value <= threshold
        
        # Compute bootstrap CI if requested
        confidence_interval = None
        p_value = None
        
        if compute_ci and self.bootstrap_samples > 0:
            confidence_interval, p_value = self._compute_bootstrap_ci(
                y_true, y_pred, sensitive_features,
                metric, ci_alpha
            )
        
        result = MetricResult(
            metric_name=metric,
            value=metric_value,
            is_fair=is_fair,
            threshold=threshold,
            group_metrics=group_metrics,
            confidence_interval=confidence_interval,
            p_value=p_value,
        )
        
        return result
    
    def _compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        metric_name: str,
        alpha: float = 0.05,
    ) -> tuple:
        """Compute bootstrap confidence interval for metric."""
        n = len(y_true)
        bootstrap_values = []
        
        for i in range(self.bootstrap_samples):
            # Sample with replacement
            indices = self.rng.choice(n, size=n, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            sensitive_boot = sensitive_features[indices]
            
            # Compute metric for bootstrap sample
            try:
                value, _, _ = _compute_metric(
                    metric_name, y_true_boot, y_pred_boot, sensitive_boot
                )
                bootstrap_values.append(value)
            except Exception:
                # Skip failed bootstrap samples
                continue
        
        if len(bootstrap_values) < 10:
            return None, None
        
        bootstrap_values = np.array(bootstrap_values)
        
        # Compute percentile-based confidence interval
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_values, lower_percentile)
        ci_upper = np.percentile(bootstrap_values, upper_percentile)
        
        # Compute p-value
        p_value = np.mean(bootstrap_values > 0)
        
        return (ci_lower, ci_upper), p_value
    
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        threshold: float = 0.1,
        compute_ci: bool = True,
    ) -> Dict[str, MetricResult]:
        """Compute all supported fairness metrics."""
        metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
        results = {}
        
        for metric in metrics:
            results[metric] = self.compute_metric(
                y_true, y_pred, sensitive_features,
                metric=metric,
                threshold=threshold,
                compute_ci=compute_ci,
            )
        
        return results
    
    def get_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Get detailed per-group metrics."""
        return compute_group_metrics(y_true, y_pred, sensitive_features)
    
    def interpret(self, result: MetricResult) -> str:
        """Get human-readable interpretation of a metric result."""
        return interpret_metric(
            result.metric_name,
            result.value,
            result.threshold,
            result.group_metrics,
        )