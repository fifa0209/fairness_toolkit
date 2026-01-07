"""
Real-time Fairness Tracker - Monitor fairness metrics in production.

Tracks fairness over time using sliding windows for balance between
detection speed and statistical stability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime
from collections import deque

from shared.logging import get_logger
from shared.validation import validate_predictions

logger = get_logger(__name__)


class RealTimeFairnessTracker:
    """
    Track fairness metrics in real-time with sliding windows.
    
    Maintains a time-series of fairness metrics computed over
    sliding windows of recent predictions.
    
    Example:
        >>> tracker = RealTimeFairnessTracker(window_size=1000)
        >>> 
        >>> # Process batches of predictions
        >>> for batch in prediction_batches:
        ...     tracker.add_batch(
        ...         batch['y_pred'],
        ...         batch['y_true'],
        ...         batch['sensitive']
        ...     )
        ...     
        ...     # Get current metrics
        ...     metrics = tracker.get_current_metrics()
        ...     if metrics['demographic_parity'] > 0.1:
        ...         alert("Fairness violation detected!")
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        metrics: List[str] = None,
        min_samples: int = 100,
    ):
        """
        Initialize real-time tracker.
        
        Args:
            window_size: Number of samples in sliding window
            metrics: List of metrics to track (None = all)
            min_samples: Minimum samples before computing metrics
        """
        self.window_size = window_size
        self.metrics = metrics or ['demographic_parity', 'equalized_odds']
        self.min_samples = min_samples
        
        # Sliding window buffers
        self.y_true_buffer = deque(maxlen=window_size)
        self.y_pred_buffer = deque(maxlen=window_size)
        self.sensitive_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # Time series storage
        self.history = pd.DataFrame(columns=['timestamp'] + self.metrics)
        
        self.n_samples_processed = 0
        
        logger.info(
            f"RealTimeFairnessTracker initialized: "
            f"window={window_size}, metrics={self.metrics}"
        )
    
    def add_batch(
        self,
        y_pred: Union[np.ndarray, List],
        y_true: Union[np.ndarray, List],
        sensitive_features: Union[np.ndarray, List],
        timestamps: Optional[Union[np.ndarray, List]] = None,
    ) -> Dict[str, float]:
        """
        Add a batch of predictions and compute metrics.
        
        Args:
            y_pred: Predicted labels
            y_true: True labels
            sensitive_features: Protected attribute
            timestamps: Timestamps for each prediction (optional)
            
        Returns:
            Dictionary of current metric values
        """
        # Convert to arrays
        y_pred = np.asarray(y_pred)
        y_true = np.asarray(y_true)
        sensitive_features = np.asarray(sensitive_features)
        
        # Validate
        validate_predictions(y_true, y_pred)
        
        if len(y_pred) != len(y_true) or len(y_pred) != len(sensitive_features):
            raise ValueError("All inputs must have same length")
        
        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = [datetime.now()] * len(y_pred)
        
        # Add to buffers
        for i in range(len(y_pred)):
            self.y_pred_buffer.append(y_pred[i])
            self.y_true_buffer.append(y_true[i])
            self.sensitive_buffer.append(sensitive_features[i])
            self.timestamp_buffer.append(timestamps[i])
        
        self.n_samples_processed += len(y_pred)
        
        # Compute metrics if we have enough samples
        if len(self.y_pred_buffer) >= self.min_samples:
            metrics = self._compute_metrics()
            self._record_metrics(metrics)
            return metrics
        else:
            logger.debug(
                f"Insufficient samples: {len(self.y_pred_buffer)}/{self.min_samples}"
            )
            return {}
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute fairness metrics on current window."""
        from measurement_module.src.metrics_engine import compute_metric
        
        # Convert buffers to arrays
        y_true = np.array(self.y_true_buffer)
        y_pred = np.array(self.y_pred_buffer)
        sensitive = np.array(self.sensitive_buffer)
        
        metrics_dict = {}
        
        for metric_name in self.metrics:
            try:
                value, _, _ = compute_metric(
                    metric_name, y_true, y_pred, sensitive
                )
                metrics_dict[metric_name] = value
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                metrics_dict[metric_name] = np.nan
        
        return metrics_dict
    
    def _record_metrics(self, metrics: Dict[str, float]) -> None:
        """Record metrics to time series."""
        record = {
            'timestamp': datetime.now(),
            **metrics
        }
        
        self.history = pd.concat([
            self.history,
            pd.DataFrame([record])
        ], ignore_index=True)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get most recent metric values."""
        if len(self.history) == 0:
            return {}
        
        latest = self.history.iloc[-1]
        return latest.drop('timestamp').to_dict()
    
    def get_time_series(
        self,
        metric: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get time series data for analysis.
        
        Args:
            metric: Specific metric to retrieve (None = all)
            start_time: Filter start time
            end_time: Filter end time
            
        Returns:
            DataFrame with timestamp and metric values
        """
        df = self.history.copy()
        
        # Filter by time
        if start_time:
            df = df[df['timestamp'] >= start_time]
        if end_time:
            df = df[df['timestamp'] <= end_time]
        
        # Filter by metric
        if metric:
            if metric not in df.columns:
                raise ValueError(f"Metric '{metric}' not tracked")
            df = df[['timestamp', metric]]
        
        return df
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all tracked metrics."""
        summary = {}
        
        for metric in self.metrics:
            if metric not in self.history.columns:
                continue
            
            values = self.history[metric].dropna()
            
            if len(values) == 0:
                continue
            
            summary[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'current': values.iloc[-1] if len(values) > 0 else np.nan,
            }
        
        return summary
    
    def reset(self) -> None:
        """Reset tracker (clear all buffers and history)."""
        self.y_true_buffer.clear()
        self.y_pred_buffer.clear()
        self.sensitive_buffer.clear()
        self.timestamp_buffer.clear()
        self.history = pd.DataFrame(columns=['timestamp'] + self.metrics)
        self.n_samples_processed = 0
        
        logger.info("Tracker reset")
    
    def export_history(self, filepath: str) -> None:
        """Export history to CSV."""
        self.history.to_csv(filepath, index=False)
        logger.info(f"Exported history to {filepath}")


class BatchFairnessMonitor:
    """
    Monitor fairness for batch predictions (non-streaming).
    
    Simpler than RealTimeFairnessTracker - designed for batch scoring
    rather than streaming predictions.
    
    Example:
        >>> monitor = BatchFairnessMonitor()
        >>> 
        >>> # Score a batch
        >>> results = monitor.evaluate_batch(
        ...     y_true, y_pred, sensitive_features
        ... )
        >>> 
        >>> print(f"Fair: {results['is_fair']}")
        >>> print(f"Violations: {results['violations']}")
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.1,
        metrics: List[str] = None,
    ):
        """
        Initialize batch monitor.
        
        Args:
            fairness_threshold: Maximum acceptable metric value
            metrics: List of metrics to compute
        """
        self.fairness_threshold = fairness_threshold
        self.metrics = metrics or ['demographic_parity', 'equalized_odds']
    
    def evaluate_batch(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict:
        """
        Evaluate fairness for a batch of predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute
            
        Returns:
            Dictionary with evaluation results
        """
        from measurement_module.src.metrics_engine import compute_metric
        
        results = {
            'timestamp': datetime.now(),
            'n_samples': len(y_true),
            'metrics': {},
            'violations': [],
            'is_fair': True,
        }
        
        # Compute each metric
        for metric_name in self.metrics:
            try:
                value, group_metrics, group_sizes = compute_metric(
                    metric_name, y_true, y_pred, sensitive_features
                )
                
                results['metrics'][metric_name] = {
                    'value': value,
                    'group_metrics': group_metrics,
                    'group_sizes': group_sizes,
                    'passes': value <= self.fairness_threshold,
                }
                
                # Check for violation
                if value > self.fairness_threshold:
                    results['violations'].append({
                        'metric': metric_name,
                        'value': value,
                        'threshold': self.fairness_threshold,
                    })
                    results['is_fair'] = False
            
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                results['metrics'][metric_name] = {'error': str(e)}
        
        return results