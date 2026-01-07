"""
Statistical Validation - Bootstrap confidence intervals and effect sizes.

Provides uncertainty quantification for fairness metrics using bootstrap resampling.
This is what distinguishes academic-quality analysis from ad-hoc metrics.

48-hour scope: Bootstrap CI + Cohen's d effect size only.
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy import stats

from shared.constants import DEFAULT_CONFIDENCE_LEVEL, DEFAULT_BOOTSTRAP_SAMPLES
from shared.logging import get_logger

logger = get_logger(__name__)


def bootstrap_confidence_interval(
    metric_func: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_state: Optional[int] = None,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute bootstrap confidence interval for a fairness metric.
    
    Bootstrap resampling provides uncertainty quantification without
    assuming parametric distributions. Critical for academic rigor.
    
    Args:
        metric_func: Function that computes metric (returns value, group_metrics, group_sizes)
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (point_estimate, (ci_lower, ci_upper))
        
    Example:
        >>> from metrics_engine import demographic_parity_difference
        >>> point_est, ci = bootstrap_confidence_interval(
        ...     demographic_parity_difference,
        ...     y_true, y_pred, sensitive_features,
        ...     n_bootstrap=1000
        ... )
        >>> print(f"Bias: {point_est:.3f}, 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y_true)
    bootstrap_estimates = []
    
    logger.info(f"Computing {n_bootstrap} bootstrap samples...")
    
    # Compute point estimate on full data
    point_estimate, _, _ = metric_func(y_true, y_pred, sensitive_features)
    
    # Bootstrap resampling
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        sensitive_boot = sensitive_features[indices]
        
        try:
            # Compute metric on bootstrap sample
            metric_value, _, _ = metric_func(y_true_boot, y_pred_boot, sensitive_boot)
            bootstrap_estimates.append(metric_value)
        except Exception as e:
            # Skip this bootstrap sample if computation fails
            logger.debug(f"Bootstrap sample {i} failed: {e}")
            continue
    
    if len(bootstrap_estimates) < n_bootstrap * 0.9:
        logger.warning(
            f"Only {len(bootstrap_estimates)}/{n_bootstrap} bootstrap samples succeeded"
        )
    
    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
    ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
    
    logger.info(
        f"Bootstrap CI ({confidence_level*100:.0f}%): "
        f"[{ci_lower:.4f}, {ci_upper:.4f}] (point estimate: {point_estimate:.4f})"
    )
    
    return point_estimate, (ci_lower, ci_upper)


def compute_effect_size_cohens_d(
    group_1_values: np.ndarray,
    group_2_values: np.ndarray,
) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Cohen's d measures the standardized difference between two means:
    d = (mean1 - mean2) / pooled_std
    
    Interpretation:
    - |d| < 0.2: negligible
    - |d| < 0.5: small
    - |d| < 0.8: medium
    - |d| >= 0.8: large
    
    Args:
        group_1_values: Values for first group
        group_2_values: Values for second group
        
    Returns:
        Cohen's d effect size
    """
    n1 = len(group_1_values)
    n2 = len(group_2_values)
    
    if n1 < 2 or n2 < 2:
        logger.warning("Need at least 2 samples per group for Cohen's d")
        return 0.0
    
    mean1 = np.mean(group_1_values)
    mean2 = np.mean(group_2_values)
    
    var1 = np.var(group_1_values, ddof=1)
    var2 = np.var(group_2_values, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        logger.warning("Pooled std is zero, cannot compute Cohen's d")
        return 0.0
    
    cohens_d = (mean1 - mean2) / pooled_std
    
    return cohens_d


def interpret_effect_size(effect_size: float) -> str:
    """
    Interpret Cohen's d effect size magnitude.
    
    Args:
        effect_size: Cohen's d value
        
    Returns:
        Interpretation string
    """
    abs_d = abs(effect_size)
    
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    return f"{magnitude} (|d|={abs_d:.3f})"


def compute_risk_ratio(
    group_1_positive_rate: float,
    group_2_positive_rate: float,
) -> float:
    """
    Compute risk ratio between two groups.
    
    Risk ratio = P(positive | group 1) / P(positive | group 2)
    
    Interpretation:
    - RR = 1.0: Equal rates
    - RR > 1.0: Group 1 has higher positive rate
    - RR < 1.0: Group 2 has higher positive rate
    
    Common threshold: RR should be between 0.8 and 1.25
    
    Args:
        group_1_positive_rate: Positive rate for group 1
        group_2_positive_rate: Positive rate for group 2
        
    Returns:
        Risk ratio
    """
    if group_2_positive_rate == 0:
        logger.warning("Group 2 positive rate is zero, cannot compute risk ratio")
        return np.inf
    
    risk_ratio = group_1_positive_rate / group_2_positive_rate
    
    return risk_ratio


# def interpret_risk_ratio(risk_ratio: float, threshold_range: Tuple[float, float] = (0.8, 1.25)) -> str:
#     """
#     Interpret risk ratio.
    
#     Args:
#         risk_ratio: Computed risk ratio
#         threshold_range: (lower, upper) bounds for fair range
        
#     Returns:
#         Interpretation string
#     """
#     lower, upper = threshold_range
    
#     if np.isinf(risk_ratio):
#         return "undefined (division by zero)"
    
#     if lower <= risk_ratio <= upper:
#         return f"fair (RR={risk_ratio:.3f}, within [{lower}, {upper}])"
#     elif risk_ratio < lower:
#         return f"group 2 favored (RR={risk_ratio:.3f} < {lower})"
#     else:
#         return f"group 1 favored (RR={risk_ratio:.3f} > {upper})"

def interpret_risk_ratio(
    risk_ratio: float,
    threshold_range: Tuple[float, float] = (0.8, 1.25)
) -> str:
    """
    Interpret risk ratio.
    
    Args:
        risk_ratio: Computed risk ratio
        threshold_range: (lower, upper) bounds for fair range
        
    Returns:
        Interpretation string
    """
    lower, upper = threshold_range
    
    if np.isinf(risk_ratio):
        return "undefined (division by zero)"
    
    if lower <= risk_ratio <= upper:
        return f"fair (RR={risk_ratio:.3f}, within [{lower}, {upper}])"
    elif risk_ratio < lower:
        return f"group 2 favored (RR={risk_ratio:.3f} < {lower})"
    else:
        return f"group 1 favored (RR={risk_ratio:.3f} > {upper})"
# def statistical_significance_test(
#     group_1_positive: int,
#     group_1_total: int,
#     group_2_positive: int,
#     group_2_total: int,
#     alpha: float = 0.05,
# ) -> Tuple[bool, float]:
#     """
#     Test if difference in proportions is statistically significant.
    
#     Uses two-proportion z-test (chi-square approximation).
    
#     Args:
#         group_1_positive: Number of positives in group 1
#         group_1_total: Total samples in group 1
#         group_2_positive: Number of positives in group 2
#         group_2_total: Total samples in group 2
#         alpha: Significance level
        
#     Returns:
#         Tuple of (is_significant, p_value)
#     """
#     # Create contingency table
#     contingency = np.array([
#         [group_1_positive, group_1_total - group_1_positive],
#         [group_2_positive, group_2_total - group_2_positive]
#     ])
    
#     # Chi-square test
#     chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
#     is_significant = p_value < alpha
    
#     logger.info(
#         f"Statistical significance test: p={p_value:.4f}, "
#         f"significant={is_significant} (α={alpha})"
#     )
    
#     return is_significant, p_value

def statistical_significance_test(
    group_1_positive: int,
    group_1_total: int,
    group_2_positive: int,
    group_2_total: int,
    alpha: float = 0.05,
) -> Tuple[bool, float]:
    """
    Test if difference in proportions is statistically significant.
    
    Uses two-proportion z-test (chi-square approximation).
    
    Args:
        group_1_positive: Number of positives in group 1
        group_1_total: Total samples in group 1
        group_2_positive: Number of positives in group 2
        group_2_total: Total samples in group 2
        alpha: Significance level
        
    Returns:
        Tuple of (is_significant, p_value)
    """
    # Create contingency table
    contingency = np.array([
        [group_1_positive, group_1_total - group_1_positive],
        [group_2_positive, group_2_total - group_2_positive]
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Convert numpy bool to Python bool
    is_significant = bool(p_value < alpha)
    
    logger.info(
        f"Statistical significance test: p={p_value:.4f}, "
        f"significant={is_significant} (α={alpha})"
    )
    
    return is_significant, float(p_value)
def compute_standard_error_proportion(
    n_positive: int,
    n_total: int,
) -> float:
    """
    Compute standard error for a proportion.
    
    SE = sqrt(p * (1-p) / n)
    
    Args:
        n_positive: Number of positive samples
        n_total: Total number of samples
        
    Returns:
        Standard error
    """
    if n_total == 0:
        return 0.0
    
    p = n_positive / n_total
    se = np.sqrt(p * (1 - p) / n_total)
    
    return se


def parametric_confidence_interval(
    n_positive: int,
    n_total: int,
    confidence_level: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute parametric confidence interval for a proportion.
    
    Uses normal approximation (valid for large n and not extreme proportions).
    
    Args:
        n_positive: Number of positive samples
        n_total: Total number of samples
        confidence_level: Confidence level
        
    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    if n_total == 0:
        return (0.0, 0.0)
    
    p = n_positive / n_total
    se = compute_standard_error_proportion(n_positive, n_total)
    
    # Z-score for confidence level
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    ci_lower = max(0, p - z * se)
    ci_upper = min(1, p + z * se)
    
    return (ci_lower, ci_upper)


def minimum_detectable_effect(
    n_samples: int,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """
    Compute minimum detectable effect size given sample size.
    
    This tells you the smallest fairness violation you can reliably detect.
    
    Args:
        n_samples: Total sample size
        alpha: Significance level
        power: Statistical power (1 - β)
        
    Returns:
        Minimum detectable effect size (Cohen's d)
    """
    # Critical values
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    
    # MDE formula (for two-sample t-test)
    mde = (z_alpha + z_beta) * np.sqrt(4 / n_samples)
    
    return mde

