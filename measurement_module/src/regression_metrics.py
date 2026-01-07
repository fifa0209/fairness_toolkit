"""
Regression Fairness Metrics - Fairness analysis for regression tasks.

Implements fairness metrics for regression models, including:
- Difference in MAE across groups
- Difference in RMSE across groups
- Difference in R² across groups
- Residual bias analysis
"""

import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mean_absolute_error_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute difference in Mean Absolute Error (MAE) across groups.
    
    Lower MAE indicates better performance. A positive difference means
    the model performs worse for group 1 than group 0.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute (binary)
        
    Returns:
        Tuple of (difference, group_maes, group_sizes)
        
    Example:
        >>> y_true = np.array([1.5, 2.3, 3.1, 2.8, 1.9])
        >>> y_pred = np.array([1.6, 2.1, 3.3, 2.6, 2.0])
        >>> gender = np.array([0, 0, 1, 1, 1])
        >>> diff, group_maes, sizes = mean_absolute_error_difference(
        ...     y_true, y_pred, gender
        ... )
        >>> print(f"MAE difference: {diff:.4f}")
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    group_maes = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        mae = mean_absolute_error(y_true_group, y_pred_group)
        
        group_maes[f"Group_{group}"] = mae
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
    
    # Difference (absolute value)
    maes = list(group_maes.values())
    difference = abs(maes[0] - maes[1])
    
    return difference, group_maes, group_sizes


def root_mean_squared_error_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute difference in Root Mean Squared Error (RMSE) across groups.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute (binary)
        
    Returns:
        Tuple of (difference, group_rmses, group_sizes)
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    group_rmses = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_group, y_pred_group))
        
        group_rmses[f"Group_{group}"] = rmse
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
    
    # Difference (absolute value)
    rmses = list(group_rmses.values())
    difference = abs(rmses[0] - rmses[1])
    
    return difference, group_rmses, group_sizes


def r2_score_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute difference in R² score across groups.
    
    Higher R² indicates better fit. A positive difference means
    the model explains more variance for group 1 than group 0.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute (binary)
        
    Returns:
        Tuple of (difference, group_r2s, group_sizes)
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    group_r2s = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        # Handle case where all y values are the same
        if np.var(y_true_group) == 0:
            r2 = 0.0
        else:
            r2 = r2_score(y_true_group, y_pred_group)
        
        group_r2s[f"Group_{group}"] = r2
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
    
    # Difference (absolute value)
    r2s = list(group_r2s.values())
    difference = abs(r2s[0] - r2s[1])
    
    return difference, group_r2s, group_sizes


def mean_residual_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute difference in mean residuals (bias) across groups.
    
    Mean residual = mean(y_true - y_pred)
    A non-zero mean residual indicates systematic over/under-prediction.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute (binary)
        
    Returns:
        Tuple of (difference, group_residuals, group_sizes)
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    residuals = y_true - y_pred
    
    group_residuals = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        mean_residual = np.mean(residuals[mask])
        
        group_residuals[f"Group_{group}"] = mean_residual
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
    
    # Difference (signed, to capture direction of bias)
    residual_values = list(group_residuals.values())
    difference = abs(residual_values[0] - residual_values[1])
    
    return difference, group_residuals, group_sizes


def compute_regression_metric(
    metric_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute specified regression fairness metric.
    
    Args:
        metric_name: One of 'mae_difference', 'rmse_difference', 'r2_difference', 'residual_bias'
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute
        
    Returns:
        Tuple of (metric_value, group_metrics, group_sizes)
        
    Raises:
        ValueError: If metric_name is not recognized
    """
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_features = np.asarray(sensitive_features)
    
    # Validate inputs
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, y_pred has {len(y_pred)}")
    
    if len(y_true) != len(sensitive_features):
        raise ValueError(f"Length mismatch: y_true has {len(y_true)}, sensitive_features has {len(sensitive_features)}")
    
    # Compute metric
    if metric_name == "mae_difference":
        return mean_absolute_error_difference(y_true, y_pred, sensitive_features)
    
    elif metric_name == "rmse_difference":
        return root_mean_squared_error_difference(y_true, y_pred, sensitive_features)
    
    elif metric_name == "r2_difference":
        return r2_score_difference(y_true, y_pred, sensitive_features)
    
    elif metric_name == "residual_bias":
        return mean_residual_difference(y_true, y_pred, sensitive_features)
    
    else:
        raise ValueError(
            f"Unknown regression metric: {metric_name}. "
            f"Choose from: mae_difference, rmse_difference, r2_difference, residual_bias"
        )


def interpret_regression_metric(
    metric_name: str,
    metric_value: float,
    threshold: float,
    group_metrics: Dict[str, float],
) -> str:
    """
    Generate human-readable interpretation of regression fairness metric.
    
    Args:
        metric_name: Name of the metric
        metric_value: Computed metric value
        threshold: Fairness threshold
        group_metrics: Per-group metric values
        
    Returns:
        Interpretation string
    """
    is_fair = metric_value <= threshold
    status = "FAIR" if is_fair else "UNFAIR"
    
    # Format group metrics
    group_strs = [f"{g}: {v:.3f}" for g, v in group_metrics.items()]
    group_summary = ", ".join(group_strs)
    
    if metric_name == "mae_difference":
        interpretation = (
            f"MAE Difference: {status} (difference={metric_value:.3f}, threshold={threshold:.3f}). "
            f"Group MAEs: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Model has similar error rates across groups."
        else:
            interpretation += "Model has significantly different error rates across groups."
    
    elif metric_name == "rmse_difference":
        interpretation = (
            f"RMSE Difference: {status} (difference={metric_value:.3f}, threshold={threshold:.3f}). "
            f"Group RMSEs: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Model has similar prediction variance across groups."
        else:
            interpretation += "Model has different prediction variance across groups."
    
    elif metric_name == "r2_difference":
        interpretation = (
            f"R² Difference: {status} (difference={metric_value:.3f}, threshold={threshold:.3f}). "
            f"Group R² scores: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Model explains similar variance across groups."
        else:
            interpretation += "Model explains different amounts of variance across groups."
    
    elif metric_name == "residual_bias":
        interpretation = (
            f"Residual Bias: {status} (difference={metric_value:.3f}, threshold={threshold:.3f}). "
            f"Group mean residuals: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Model has no systematic bias across groups."
        else:
            interpretation += "Model systematically over/under-predicts for different groups."
    
    else:
        interpretation = f"{metric_name}: {status} (value={metric_value:.3f})"
    
    return interpretation


def compute_all_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    threshold: float = 0.1,
) -> Dict[str, Dict[str, any]]:
    """
    Compute all regression fairness metrics at once.
    
    Args:
        y_true: True continuous values
        y_pred: Predicted continuous values
        sensitive_features: Protected attribute
        threshold: Fairness threshold
        
    Returns:
        Dictionary mapping metric name -> result dict
        
    Example:
        >>> results = compute_all_regression_metrics(y_true, y_pred, gender)
        >>> for metric, result in results.items():
        ...     print(f"{metric}: {result['value']:.4f} ({'FAIR' if result['is_fair'] else 'UNFAIR'})")
    """
    metrics = ["mae_difference", "rmse_difference", "r2_difference", "residual_bias"]
    
    results = {}
    
    for metric in metrics:
        try:
            value, group_metrics, group_sizes = compute_regression_metric(
                metric, y_true, y_pred, sensitive_features
            )
            
            is_fair = value <= threshold
            interpretation = interpret_regression_metric(
                metric, value, threshold, group_metrics
            )
            
            results[metric] = {
                'value': value,
                'is_fair': is_fair,
                'threshold': threshold,
                'group_metrics': group_metrics,
                'group_sizes': group_sizes,
                'interpretation': interpretation
            }
        except Exception as e:
            print(f"Warning: Failed to compute {metric}: {e}")
            continue
    
    return results