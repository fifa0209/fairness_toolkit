"""
Metrics Engine - Core fairness metrics computation.

Implements demographic parity, equalized odds, and equal opportunity
for binary classification. Focuses on correctness and clarity.

48-hour scope: Binary classification + binary protected attributes only.
"""

import numpy as np
from typing import Dict, Tuple, Union, TYPE_CHECKING
from sklearn.metrics import confusion_matrix

# Fallback implementations for shared modules
try:
    from shared.validation import validate_predictions, safe_divide, ValidationError
    from shared.logging import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    
    class ValidationError(Exception):
        pass
    
    def validate_predictions(y_true, y_pred):
        """Validate binary predictions."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        if len(y_true) != len(y_pred):
            raise ValidationError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")
        
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValidationError("y_true must be binary (0 or 1)")
        
        if not np.all(np.isin(y_pred, [0, 1])):
            raise ValidationError("y_pred must be binary (0 or 1)")
    
    def safe_divide(a, b, default=0.0):
        """Safe division that returns default for division by zero."""
        return a / b if b != 0 else default

if TYPE_CHECKING:
    import pandas as pd


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-group classification metrics.
    
    Args:
        y_true: True labels (binary: 0 or 1)
        y_pred: Predicted labels (binary: 0 or 1)
        sensitive_features: Protected attribute (binary: 0 or 1)
        
    Returns:
        Dictionary mapping group -> metrics
        
    Example:
        {
            'Group_0': {'positive_rate': 0.45, 'tpr': 0.52, 'fpr': 0.38, ...},
            'Group_1': {'positive_rate': 0.60, 'tpr': 0.68, 'fpr': 0.42, ...}
        }
    """
    groups = np.unique(sensitive_features)
    group_metrics = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        # Basic counts
        n = len(y_true_group)
        n_positive_pred = np.sum(y_pred_group == 1)
        n_positive_true = np.sum(y_true_group == 1)
        n_negative_true = np.sum(y_true_group == 0)
        
        # Positive prediction rate (for demographic parity)
        positive_rate = safe_divide(n_positive_pred, n)
        
        # Confusion matrix components
        if n_positive_true > 0 and n_negative_true > 0:
            cm = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            
            # True Positive Rate (Recall, Sensitivity)
            tpr = safe_divide(tp, tp + fn)
            
            # False Positive Rate
            fpr = safe_divide(fp, fp + tn)
            
            # True Negative Rate (Specificity)
            tnr = safe_divide(tn, tn + fp)
            
            # False Negative Rate
            fnr = safe_divide(fn, fn + tp)
            
            # Precision
            precision = safe_divide(tp, tp + fp)
            
        else:
            # Edge case: all same class
            tpr = fpr = tnr = fnr = precision = 0.0
            logger.warning(f"Group {group} has only one class in y_true")
        
        group_metrics[f"Group_{group}"] = {
            'n_samples': n,
            'positive_rate': positive_rate,
            'tpr': tpr,
            'fpr': fpr,
            'tnr': tnr,
            'fnr': fnr,
            'precision': precision,
        }
    
    return group_metrics


def demographic_parity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    allow_multiple_groups: bool = False
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute demographic parity difference.
    
    Demographic Parity: All groups should receive positive predictions at equal rates.
    Difference: |P(ŷ=1|s=0) - P(ŷ=1|s=1)|
    
    Args:
        y_true: True labels (not used, but kept for API consistency)
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        allow_multiple_groups: If True, computes max difference across all groups
        
    Returns:
        Tuple of (difference, group_rates, group_sizes)
        - difference: Maximum pairwise difference in positive prediction rates
        - group_rates: Dictionary mapping "Group_X" to positive prediction rate
        - group_sizes: Dictionary mapping "Group_X" to number of samples
    """
    validate_predictions(y_true, y_pred)
    
    groups = np.unique(sensitive_features)
    
    # Validation for 2 groups (unless multiple groups allowed)
    if not allow_multiple_groups and len(groups) != 2:
        raise ValidationError(f"Expected 2 groups, got {len(groups)}")
    
    # Compute selection rates per group
    group_rates = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        group_rates[f"Group_{group}"] = float(np.mean(y_pred[mask]))
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
    
    # Compute difference
    if allow_multiple_groups:
        # For multiple groups, compute max pairwise difference
        rates = list(group_rates.values())
        difference = max(rates) - min(rates)
    else:
        # Original 2-group logic
        rate_0 = group_rates[f"Group_{groups[0]}"]
        rate_1 = group_rates[f"Group_{groups[1]}"]
        difference = abs(rate_0 - rate_1)
    
    logger.info(f"Demographic parity difference: {difference:.4f}")
    
    return difference, group_rates, group_sizes


def equalized_odds_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute equalized odds difference.
    
    Equalized Odds: TPR and FPR should be equal across groups
    Difference: max(|TPR_0 - TPR_1|, |FPR_0 - FPR_1|)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        
    Returns:
        Tuple of (difference, group_metrics, group_sizes)
    """
    validate_predictions(y_true, y_pred)
    
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValidationError(f"Expected 2 groups, got {len(groups)}")
    
    group_tprs = {}
    group_fprs = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
        
        # Compute TPR and FPR
        if len(np.unique(y_true_group)) == 2:
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()
            
            tpr = safe_divide(tp, tp + fn)
            fpr = safe_divide(fp, fp + tn)
        else:
            tpr = fpr = 0.0
            logger.warning(f"Group {group} has only one class")
        
        group_tprs[f"Group_{group}"] = float(tpr)
        group_fprs[f"Group_{group}"] = float(fpr)
    
    # Max difference in TPR and FPR
    tpr_values = list(group_tprs.values())
    fpr_values = list(group_fprs.values())
    
    tpr_diff = abs(tpr_values[0] - tpr_values[1])
    fpr_diff = abs(fpr_values[0] - fpr_values[1])
    
    difference = max(tpr_diff, fpr_diff)
    
    # Return combined metrics
    group_metrics = {}
    for group in groups:
        group_name = f"Group_{group}"
        group_metrics[group_name] = group_tprs[group_name]  # Use TPR as primary
    
    logger.info(f"Equalized odds difference: {difference:.4f} (TPR: {tpr_diff:.4f}, FPR: {fpr_diff:.4f})")
    
    return difference, group_metrics, group_sizes


def equal_opportunity_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute equal opportunity difference.
    
    Equal Opportunity: TPR should be equal across groups (subset of equalized odds)
    Difference: |TPR_0 - TPR_1|
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        
    Returns:
        Tuple of (difference, group_tprs, group_sizes)
    """
    validate_predictions(y_true, y_pred)
    
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValidationError(f"Expected 2 groups, got {len(groups)}")
    
    group_tprs = {}
    group_sizes = {}
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        group_sizes[f"Group_{group}"] = int(np.sum(mask))
        
        # Compute TPR
        if len(np.unique(y_true_group)) == 2:
            tn, fp, fn, tp = confusion_matrix(
                y_true_group, y_pred_group, labels=[0, 1]
            ).ravel()
            tpr = safe_divide(tp, tp + fn)
        else:
            tpr = 0.0
            logger.warning(f"Group {group} has only one class")
        
        group_tprs[f"Group_{group}"] = float(tpr)
    
    # Difference in TPR
    tpr_values = list(group_tprs.values())
    difference = abs(tpr_values[0] - tpr_values[1])
    
    logger.info(f"Equal opportunity difference: {difference:.4f}")
    
    return difference, group_tprs, group_sizes


def compute_metric(
    metric_name: str,
    y_true: Union[np.ndarray, 'pd.Series'],
    y_pred: Union[np.ndarray, 'pd.Series'],
    sensitive_features: Union[np.ndarray, 'pd.Series'],
    allow_multiple_groups: bool = False,
) -> Tuple[float, Dict[str, float], Dict[str, int]]:
    """
    Compute specified fairness metric.
    
    Args:
        metric_name: One of 'demographic_parity', 'equalized_odds', 'equal_opportunity'
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        allow_multiple_groups: Allow more than 2 groups for demographic_parity
        
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
    validate_predictions(y_true, y_pred)
    
    # Compute metric
    if metric_name == "demographic_parity":
        return demographic_parity_difference(
            y_true, y_pred, sensitive_features, allow_multiple_groups
        )
    
    elif metric_name == "equalized_odds":
        return equalized_odds_difference(y_true, y_pred, sensitive_features)
    
    elif metric_name == "equal_opportunity":
        return equal_opportunity_difference(y_true, y_pred, sensitive_features)
    
    else:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Choose from: demographic_parity, equalized_odds, equal_opportunity"
        )


def interpret_metric(
    metric_name: str,
    metric_value: float,
    threshold: float,
    group_metrics: Dict[str, float],
) -> str:
    """
    Generate human-readable interpretation of fairness metric.
    
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
    
    if metric_name == "demographic_parity":
        interpretation = (
            f"Demographic Parity: {status} (difference={metric_value:.3f}, "
            f"threshold={threshold:.3f}). "
            f"Positive prediction rates: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Groups receive positive predictions at similar rates."
        else:
            interpretation += "Significant disparity in positive prediction rates across groups."
    
    elif metric_name == "equalized_odds":
        interpretation = (
            f"Equalized Odds: {status} (difference={metric_value:.3f}, "
            f"threshold={threshold:.3f}). "
            f"True positive rates: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Similar error rates across groups."
        else:
            interpretation += "Different error rates across groups (check both TPR and FPR)."
    
    elif metric_name == "equal_opportunity":
        interpretation = (
            f"Equal Opportunity: {status} (difference={metric_value:.3f}, "
            f"threshold={threshold:.3f}). "
            f"True positive rates: {group_summary}. "
        )
        
        if is_fair:
            interpretation += "Groups have equal opportunity for positive outcomes."
        else:
            interpretation += "Groups have unequal true positive rates."
    
    else:
        interpretation = f"{metric_name}: {status} (value={metric_value:.3f})"
    
    return interpretation