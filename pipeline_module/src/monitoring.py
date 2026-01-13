"""
Monitoring and Evaluation Tools - Track fairness metrics over time.

Provides tools to:
- Monitor fairness metrics across model versions
- Generate reliability diagrams
- Track calibration drift
- Alert on fairness violations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

from shared.logging import get_logger
from calibration import CalibrationEvaluator

logger = get_logger(__name__)


class FairnessMonitor:
    """
    Monitor fairness metrics over time and across model versions.
    
    Example:
        >>> monitor = FairnessMonitor()
        >>> monitor.log_metrics(
        ...     model_version='v1.0',
        ...     metrics={'demographic_parity': 0.05, 'ece': 0.03}
        ... )
        >>> monitor.check_drift(baseline_version='v1.0', current_version='v1.1')
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Initialize fairness monitor.
        
        Args:
            log_file: Path to JSON log file for persisting metrics
        """
        self.log_file = Path(log_file) if log_file else None
        self.metrics_history = []
        
        if self.log_file and self.log_file.exists():
            self.load_history()
    
    def log_metrics(
        self,
        model_version: str,
        metrics: Dict[str, float],
        sensitive_attribute: str = 'unknown',
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Log fairness metrics for a model version.
        
        Args:
            model_version: Model version identifier
            metrics: Dictionary of metric_name -> value
            sensitive_attribute: Protected attribute being monitored
            metadata: Additional metadata (dataset size, etc.)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'sensitive_attribute': sensitive_attribute,
            'metrics': metrics,
            'metadata': metadata or {},
        }
        
        self.metrics_history.append(entry)
        logger.info(f"Logged metrics for {model_version}: {metrics}")
        
        if self.log_file:
            self.save_history()
    
    def get_metrics(
        self,
        model_version: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve logged metrics.
        
        Args:
            model_version: Filter by model version (None = all)
            
        Returns:
            List of metric entries
        """
        if model_version is None:
            return self.metrics_history
        
        return [
            entry for entry in self.metrics_history
            if entry['model_version'] == model_version
        ]
    
    def check_drift(
        self,
        baseline_version: str,
        current_version: str,
        threshold: float = 0.1,
    ) -> Dict[str, any]:
        """
        Check for fairness drift between model versions.
        
        Args:
            baseline_version: Baseline model version
            current_version: Current model version
            threshold: Maximum acceptable drift
            
        Returns:
            Drift report with alerts
        """
        baseline_metrics = self.get_metrics(baseline_version)
        current_metrics = self.get_metrics(current_version)
        
        if not baseline_metrics or not current_metrics:
            raise ValueError("Missing metrics for comparison")
        
        # Use most recent entry for each version
        baseline = baseline_metrics[-1]['metrics']
        current = current_metrics[-1]['metrics']
        
        drift_report = {
            'baseline_version': baseline_version,
            'current_version': current_version,
            'drifts': {},
            'alerts': [],
        }
        
        for metric_name in baseline.keys():
            if metric_name in current:
                baseline_val = baseline[metric_name]
                current_val = current[metric_name]
                drift = abs(current_val - baseline_val)
                
                drift_report['drifts'][metric_name] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'drift': drift,
                    'drift_pct': (drift / baseline_val * 100) if baseline_val != 0 else 0,
                }
                
                if drift > threshold:
                    drift_report['alerts'].append({
                        'metric': metric_name,
                        'severity': 'high' if drift > threshold * 2 else 'medium',
                        'message': f"{metric_name} drifted by {drift:.4f} (threshold: {threshold})",
                    })
        
        if drift_report['alerts']:
            logger.warning(f"Drift detected: {len(drift_report['alerts'])} alerts")
        else:
            logger.info("No significant drift detected")
        
        return drift_report
    
    def save_history(self) -> None:
        """Save metrics history to JSON file."""
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            logger.info(f"Saved metrics history to {self.log_file}")
    
    def load_history(self) -> None:
        """Load metrics history from JSON file."""
        if self.log_file and self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.metrics_history = json.load(f)
            logger.info(f"Loaded {len(self.metrics_history)} metric entries")


class ReliabilityDiagramGenerator:
    """
    Generate reliability diagrams to visualize calibration.
    
    A reliability diagram plots predicted probabilities vs actual frequencies,
    showing how well calibrated a model is.
    """
    
    @staticmethod
    def plot_reliability_diagram(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
        ax: Optional[plt.Axes] = None,
        label: str = 'Model',
    ) -> plt.Figure:
        """
        Create reliability diagram.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            ax: Matplotlib axes (None = create new)
            label: Label for the plot
            
        Returns:
            Matplotlib figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        else:
            fig = ax.get_figure()
        
        # Get calibration curve
        mean_pred, frac_pos = CalibrationEvaluator.get_calibration_curve(
            y_true, y_proba, n_bins
        )
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
        
        # Plot actual calibration
        ax.plot(mean_pred, frac_pos, 'o-', label=label, linewidth=2, markersize=8)
        
        # Styling
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Add ECE annotation
        ece = CalibrationEvaluator.compute_ece(y_true, y_proba, n_bins)
        ax.text(
            0.05, 0.95,
            f'ECE = {ece:.4f}',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        return fig
    
    @staticmethod
    def plot_group_reliability_diagrams(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: np.ndarray,
        n_bins: int = 10,
        figsize: Tuple[int, int] = (15, 5),
    ) -> plt.Figure:
        """
        Create reliability diagrams for each protected group.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            sensitive_features: Protected attribute values
            n_bins: Number of bins
            figsize: Figure size
            
        Returns:
            Matplotlib figure with subplots
        """
        groups = np.unique(sensitive_features)
        n_groups = len(groups)
        
        fig, axes = plt.subplots(1, n_groups, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        
        for idx, group in enumerate(groups):
            mask = sensitive_features == group
            
            ReliabilityDiagramGenerator.plot_reliability_diagram(
                y_true[mask],
                y_proba[mask],
                n_bins=n_bins,
                ax=axes[idx],
                label=f'Group {group}'
            )
            axes[idx].set_title(f'Group: {group}', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_before_after_calibration(
        y_true: np.ndarray,
        y_proba_before: np.ndarray,
        y_proba_after: np.ndarray,
        n_bins: int = 10,
    ) -> plt.Figure:
        """
        Plot reliability diagrams before and after calibration.
        
        Args:
            y_true: True labels
            y_proba_before: Probabilities before calibration
            y_proba_after: Probabilities after calibration
            n_bins: Number of bins
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Before calibration
        ReliabilityDiagramGenerator.plot_reliability_diagram(
            y_true, y_proba_before, n_bins, ax1, 'Before Calibration'
        )
        ax1.set_title('Before Calibration', fontsize=14, fontweight='bold')
        
        # After calibration
        ReliabilityDiagramGenerator.plot_reliability_diagram(
            y_true, y_proba_after, n_bins, ax2, 'After Calibration'
        )
        ax2.set_title('After Calibration', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig


class FairnessAlertSystem:
    """
    Alert system for fairness violations.
    
    Monitors fairness metrics and triggers alerts when thresholds are breached.
    """
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize alert system.
        
        Args:
            thresholds: Dictionary of metric_name -> max_value
        """
        self.thresholds = thresholds or {
            'demographic_parity': 0.1,
            'equalized_odds_tpr': 0.1,
            'equalized_odds_fpr': 0.1,
            'ece': 0.05,
            'group_ece_disparity': 0.03,
        }
        self.alerts = []
    
    def check_metrics(
        self,
        metrics: Dict[str, float],
        model_version: str = 'unknown',
    ) -> List[Dict]:
        """
        Check metrics against thresholds and generate alerts.
        
        Args:
            metrics: Dictionary of metric_name -> value
            model_version: Model version identifier
            
        Returns:
            List of alerts
        """
        new_alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                if value > threshold:
                    severity = 'critical' if value > threshold * 2 else 'warning'
                    
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'model_version': model_version,
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'severity': severity,
                        'message': (
                            f"{metric_name} = {value:.4f} exceeds threshold "
                            f"{threshold:.4f} for {model_version}"
                        ),
                    }
                    
                    new_alerts.append(alert)
                    self.alerts.append(alert)
                    
                    logger.warning(f"ALERT: {alert['message']}")
        
        return new_alerts
    
    def get_alerts(
        self,
        severity: Optional[str] = None,
        metric: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve alerts with optional filtering.
        
        Args:
            severity: Filter by severity ('critical', 'warning')
            metric: Filter by metric name
            
        Returns:
            List of matching alerts
        """
        filtered = self.alerts
        
        if severity:
            filtered = [a for a in filtered if a['severity'] == severity]
        
        if metric:
            filtered = [a for a in filtered if a['metric'] == metric]
        
        return filtered
    
    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts = []
        logger.info("Cleared all alerts")


class FairnessMetricsCalculator:
    """
    Calculate comprehensive fairness metrics.
    """
    
    @staticmethod
    def demographic_parity(
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> float:
        """
        Calculate demographic parity difference.
        
        Measures difference in positive prediction rates across groups.
        
        Returns:
            Maximum difference in positive rates
        """
        groups = np.unique(sensitive_features)
        rates = []
        
        for group in groups:
            mask = sensitive_features == group
            rate = y_pred[mask].mean()
            rates.append(rate)
        
        return max(rates) - min(rates)
    
    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate equalized odds metrics.
        
        Measures TPR and FPR differences across groups.
        
        Returns:
            Dictionary with TPR and FPR disparities
        """
        groups = np.unique(sensitive_features)
        tpr_rates = []
        fpr_rates = []
        
        for group in groups:
            mask = sensitive_features == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # TPR
            tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
            fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_rates.append(tpr)
            
            # FPR
            fp = ((y_true_group == 0) & (y_pred_group == 1)).sum()
            tn = ((y_true_group == 0) & (y_pred_group == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_rates.append(fpr)
        
        return {
            'tpr_disparity': max(tpr_rates) - min(tpr_rates),
            'fpr_disparity': max(fpr_rates) - min(fpr_rates),
        }
    
    @staticmethod
    def calculate_all_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate all fairness metrics.
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Demographic parity
        metrics['demographic_parity'] = FairnessMetricsCalculator.demographic_parity(
            y_pred, sensitive_features
        )
        
        # Equalized odds
        eo_metrics = FairnessMetricsCalculator.equalized_odds(
            y_true, y_pred, sensitive_features
        )
        metrics.update({
            'equalized_odds_tpr': eo_metrics['tpr_disparity'],
            'equalized_odds_fpr': eo_metrics['fpr_disparity'],
        })
        
        # Calibration metrics
        metrics['ece'] = CalibrationEvaluator.compute_ece(y_true, y_proba)
        
        group_ece = CalibrationEvaluator.compute_group_ece(
            y_true, y_proba, sensitive_features
        )
        metrics['group_ece_disparity'] = (
            max(group_ece.values()) - min(group_ece.values())
        )
        
        return metrics