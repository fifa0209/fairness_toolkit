"""
Effect Sizes - Standardized measures of disparity magnitude.

Implements Cohen's d, risk ratios, and other standardized effect size measures
to quantify the practical significance of fairness disparities.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import warnings


def compute_cohens_d(
    group_0: np.ndarray,
    group_1: np.ndarray,
    pooled: bool = True
) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Cohen's d measures the standardized difference between two means.
    
    Interpretation:
        |d| < 0.2: negligible
        |d| < 0.5: small
        |d| < 0.8: medium
        |d| >= 0.8: large
    
    Args:
        group_0: Values for first group
        group_1: Values for second group
        pooled: If True, use pooled standard deviation. If False, use group_1 std.
        
    Returns:
        Cohen's d effect size
        
    Example:
        >>> group_0 = np.array([0, 0, 1, 1, 0])
        >>> group_1 = np.array([1, 1, 1, 0, 1])
        >>> d = compute_cohens_d(group_0, group_1)
        >>> print(f"Cohen's d: {d:.3f}")
    """
    mean_0 = np.mean(group_0)
    mean_1 = np.mean(group_1)
    
    if pooled:
        # Pooled standard deviation
        n_0 = len(group_0)
        n_1 = len(group_1)
        var_0 = np.var(group_0, ddof=1)
        var_1 = np.var(group_1, ddof=1)
        
        pooled_std = np.sqrt(((n_0 - 1) * var_0 + (n_1 - 1) * var_1) / (n_0 + n_1 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean_1 - mean_0) / pooled_std
    else:
        # Use standard deviation of group_1 (control group)
        std_1 = np.std(group_1, ddof=1)
        
        if std_1 == 0:
            return 0.0
        
        cohens_d = (mean_1 - mean_0) / std_1
    
    return cohens_d


def compute_risk_ratio(
    positive_rate_0: float,
    positive_rate_1: float,
    epsilon: float = 1e-8
) -> float:
    """
    Compute risk ratio (relative risk) between two groups.
    
    Risk ratio = Rate_1 / Rate_0
    
    Interpretation:
        RR = 1: equal rates
        RR > 1: group_1 has higher rate
        RR < 1: group_1 has lower rate
    
    Args:
        positive_rate_0: Positive rate for group 0
        positive_rate_1: Positive rate for group 1
        epsilon: Small value to avoid division by zero
        
    Returns:
        Risk ratio
        
    Example:
        >>> rr = compute_risk_ratio(0.3, 0.6)
        >>> print(f"Risk ratio: {rr:.2f}")  # 2.0 (group 1 has 2x the rate)
    """
    # Add epsilon to avoid division by zero
    rate_0 = max(positive_rate_0, epsilon)
    
    risk_ratio = positive_rate_1 / rate_0
    
    return risk_ratio


def compute_odds_ratio(
    positive_rate_0: float,
    positive_rate_1: float,
    epsilon: float = 1e-8
) -> float:
    """
    Compute odds ratio between two groups.
    
    Odds ratio = (p1/(1-p1)) / (p0/(1-p0))
    
    Args:
        positive_rate_0: Positive rate for group 0
        positive_rate_1: Positive rate for group 1
        epsilon: Small value to avoid division issues
        
    Returns:
        Odds ratio
    """
    # Clip rates to avoid division issues
    rate_0 = np.clip(positive_rate_0, epsilon, 1 - epsilon)
    rate_1 = np.clip(positive_rate_1, epsilon, 1 - epsilon)
    
    odds_0 = rate_0 / (1 - rate_0)
    odds_1 = rate_1 / (1 - rate_1)
    
    odds_ratio = odds_1 / odds_0
    
    return odds_ratio


def compute_disparate_impact_ratio(
    positive_rate_0: float,
    positive_rate_1: float,
    epsilon: float = 1e-8
) -> float:
    """
    Compute disparate impact ratio (80% rule).
    
    Disparate Impact Ratio = min(p0/p1, p1/p0)
    
    The 80% rule states that the ratio should be >= 0.8.
    
    Args:
        positive_rate_0: Positive rate for group 0
        positive_rate_1: Positive rate for group 1
        epsilon: Small value to avoid division by zero
        
    Returns:
        Disparate impact ratio (between 0 and 1)
        
    Example:
        >>> di = compute_disparate_impact_ratio(0.4, 0.6)
        >>> print(f"Disparate impact: {di:.2f}")
        >>> print(f"Passes 80% rule: {di >= 0.8}")
    """
    rate_0 = max(positive_rate_0, epsilon)
    rate_1 = max(positive_rate_1, epsilon)
    
    # Return the minimum of the two ratios
    ratio = min(rate_0 / rate_1, rate_1 / rate_0)
    
    return ratio


def interpret_cohens_d(cohens_d: float) -> str:
    """
    Provide interpretation of Cohen's d effect size.
    
    Args:
        cohens_d: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    direction = "higher" if cohens_d > 0 else "lower"
    
    return f"{magnitude} effect size (|d|={abs_d:.3f}); group 1 has {direction} mean"


def interpret_risk_ratio(risk_ratio: float, threshold: float = 0.8) -> str:
    """
    Provide interpretation of risk ratio.
    
    Args:
        risk_ratio: Risk ratio value
        threshold: Minimum acceptable ratio (default: 0.8 for 80% rule)
        
    Returns:
        Interpretation string
    """
    if risk_ratio < threshold:
        return f"Group 1 has {(1-risk_ratio)*100:.1f}% lower rate (RR={risk_ratio:.3f}, fails 80% rule)"
    elif risk_ratio > (1 / threshold):
        return f"Group 1 has {(risk_ratio-1)*100:.1f}% higher rate (RR={risk_ratio:.3f})"
    else:
        return f"Rates are comparable (RR={risk_ratio:.3f}, passes 80% rule)"


def compute_all_effect_sizes(
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric_type: str = 'binary'
) -> Dict[str, float]:
    """
    Compute all available effect sizes for given predictions and groups.
    
    Args:
        y_pred: Predictions (binary for classification, continuous for regression)
        sensitive_features: Protected attribute (binary)
        metric_type: 'binary' or 'continuous'
        
    Returns:
        Dictionary of effect sizes
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    mask_0 = sensitive_features == groups[0]
    mask_1 = sensitive_features == groups[1]
    
    group_0_preds = y_pred[mask_0]
    group_1_preds = y_pred[mask_1]
    
    effect_sizes = {}
    
    # Cohen's d (works for both binary and continuous)
    effect_sizes['cohens_d'] = compute_cohens_d(group_0_preds, group_1_preds)
    
    if metric_type == 'binary':
        # Rates for binary predictions
        rate_0 = np.mean(group_0_preds)
        rate_1 = np.mean(group_1_preds)
        
        effect_sizes['positive_rate_0'] = rate_0
        effect_sizes['positive_rate_1'] = rate_1
        effect_sizes['risk_ratio'] = compute_risk_ratio(rate_0, rate_1)
        effect_sizes['odds_ratio'] = compute_odds_ratio(rate_0, rate_1)
        effect_sizes['disparate_impact_ratio'] = compute_disparate_impact_ratio(rate_0, rate_1)
    
    return effect_sizes


def compute_standardized_mean_difference(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    """
    Compute standardized mean difference in predictions between groups.
    
    This is particularly useful for regression tasks.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        sensitive_features: Protected attribute
        
    Returns:
        Tuple of (standardized_difference, group_statistics)
    """
    groups = np.unique(sensitive_features)
    if len(groups) != 2:
        raise ValueError(f"Expected 2 groups, got {len(groups)}")
    
    mask_0 = sensitive_features == groups[0]
    mask_1 = sensitive_features == groups[1]
    
    # Compute residuals
    residuals = y_true - y_pred
    
    residuals_0 = residuals[mask_0]
    residuals_1 = residuals[mask_1]
    
    # Cohen's d on residuals
    cohens_d = compute_cohens_d(residuals_0, residuals_1)
    
    group_stats = {
        'mean_residual_0': np.mean(residuals_0),
        'mean_residual_1': np.mean(residuals_1),
        'std_residual_0': np.std(residuals_0),
        'std_residual_1': np.std(residuals_1),
        'cohens_d': cohens_d
    }
    
    return cohens_d, group_stats