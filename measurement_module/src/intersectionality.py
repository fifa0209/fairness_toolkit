"""
Intersectionality Analysis - Fairness across multiple protected attributes.

Handles analysis of fairness when considering multiple demographic dimensions
simultaneously (e.g., gender × race, age × disability status).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from itertools import combinations


def create_intersectional_groups(
    *sensitive_features: np.ndarray,
    labels: Optional[List[str]] = None,
    separator: str = "_"
) -> Tuple[np.ndarray, List[str]]:
    """
    Create intersectional group identifiers from multiple protected attributes.
    
    Args:
        *sensitive_features: Variable number of protected attribute arrays
        labels: Optional labels for each attribute (e.g., ['gender', 'race'])
        separator: String to separate attribute values in group names
        
    Returns:
        Tuple of (intersectional_groups, group_labels)
        
    Example:
        >>> gender = np.array([0, 1, 0, 1, 0])
        >>> race = np.array([0, 0, 1, 1, 0])
        >>> groups, labels = create_intersectional_groups(gender, race)
        >>> print(groups)
        ['0_0', '1_0', '0_1', '1_1', '0_0']
    """
    # Validate all arrays have same length
    lengths = [len(sf) for sf in sensitive_features]
    if len(set(lengths)) > 1:
        raise ValueError(f"All arrays must have same length. Got: {lengths}")
    
    n_samples = lengths[0]
    n_attributes = len(sensitive_features)
    
    # Create group identifiers
    intersectional_groups = []
    
    for i in range(n_samples):
        group_id = separator.join(str(sf[i]) for sf in sensitive_features)
        intersectional_groups.append(group_id)
    
    intersectional_groups = np.array(intersectional_groups)
    
    # Create descriptive labels if provided
    if labels:
        unique_groups = np.unique(intersectional_groups)
        group_labels = []
        
        for group in unique_groups:
            values = group.split(separator)
            label_parts = [f"{labels[i]}={values[i]}" for i in range(len(values))]
            group_labels.append(", ".join(label_parts))
    else:
        group_labels = list(np.unique(intersectional_groups))
    
    return intersectional_groups, group_labels


def compute_intersectional_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features_dict: Dict[str, np.ndarray],
    metric_name: str = 'demographic_parity',
    min_group_size: int = 30,
    threshold: float = 0.1,
    analyzer: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Compute fairness metrics for all intersectional groups.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features_dict: Dictionary of attribute_name -> array
        metric_name: Fairness metric to compute
        min_group_size: Minimum samples per group to report
        threshold: Fairness threshold
        analyzer: Optional FairnessAnalyzer instance
        
    Returns:
        Dictionary with intersectional analysis results
        
    Example:
        >>> results = compute_intersectional_metrics(
        ...     y_true=y_test,
        ...     y_pred=y_pred,
        ...     sensitive_features_dict={
        ...         'gender': gender_array,
        ...         'race': race_array
        ...     },
        ...     min_group_size=30
        ... )
    """
    if analyzer is None:
        from .fairness_analyzer_simple import FairnessAnalyzer
        analyzer = FairnessAnalyzer()
    
    # Create intersectional groups
    attribute_names = list(sensitive_features_dict.keys())
    attribute_arrays = list(sensitive_features_dict.values())
    
    intersectional_groups, _ = create_intersectional_groups(
        *attribute_arrays,
        labels=attribute_names
    )
    
    # Compute metric for intersectional groups
    result = analyzer.compute_metric(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=intersectional_groups,
        metric=metric_name,
        threshold=threshold,
        compute_ci=False  # Too expensive for many groups
    )
    
    # Filter out small groups
    reliable_groups = {}
    small_groups = {}
    
    for group_name, size in result.group_sizes.items():
        if size >= min_group_size:
            reliable_groups[group_name] = {
                'size': size,
                'metric_value': result.group_metrics.get(group_name, None)
            }
        else:
            small_groups[group_name] = size
    
    # Identify max disparity
    if reliable_groups:
        values = [g['metric_value'] for g in reliable_groups.values() if g['metric_value'] is not None]
        if values:
            max_disparity = max(values) - min(values)
        else:
            max_disparity = None
    else:
        max_disparity = None
    
    return {
        'metric_name': metric_name,
        'overall_result': result,
        'reliable_groups': reliable_groups,
        'small_groups': small_groups,
        'max_disparity': max_disparity,
        'n_reliable_groups': len(reliable_groups),
        'n_small_groups': len(small_groups),
        'warning': len(small_groups) > 0
    }


def analyze_pairwise_disparities(
    y_pred: np.ndarray,
    sensitive_features_dict: Dict[str, np.ndarray],
    min_group_size: int = 30
) -> pd.DataFrame:
    """
    Analyze pairwise disparities between all intersectional groups.
    
    Computes the difference in positive prediction rates between every
    pair of intersectional groups.
    
    Args:
        y_pred: Predicted labels
        sensitive_features_dict: Dictionary of attribute_name -> array
        min_group_size: Minimum samples per group
        
    Returns:
        DataFrame with pairwise disparities
        
    Example:
        >>> disparities = analyze_pairwise_disparities(
        ...     y_pred=predictions,
        ...     sensitive_features_dict={'gender': gender, 'age': age}
        ... )
        >>> print(disparities.head())
    """
    # Create intersectional groups
    attribute_names = list(sensitive_features_dict.keys())
    attribute_arrays = list(sensitive_features_dict.values())
    
    intersectional_groups, group_labels = create_intersectional_groups(
        *attribute_arrays,
        labels=attribute_names
    )
    
    # Compute positive rates per group
    unique_groups = np.unique(intersectional_groups)
    
    group_stats = {}
    for group in unique_groups:
        mask = intersectional_groups == group
        size = np.sum(mask)
        
        if size >= min_group_size:
            positive_rate = np.mean(y_pred[mask])
            group_stats[group] = {
                'size': size,
                'positive_rate': positive_rate
            }
    
    # Compute all pairwise differences
    pairwise_results = []
    
    group_names = list(group_stats.keys())
    for i, group_1 in enumerate(group_names):
        for group_2 in group_names[i+1:]:
            rate_1 = group_stats[group_1]['positive_rate']
            rate_2 = group_stats[group_2]['positive_rate']
            
            disparity = abs(rate_1 - rate_2)
            
            pairwise_results.append({
                'group_1': group_1,
                'group_2': group_2,
                'rate_1': rate_1,
                'rate_2': rate_2,
                'disparity': disparity,
                'size_1': group_stats[group_1]['size'],
                'size_2': group_stats[group_2]['size']
            })
    
    df = pd.DataFrame(pairwise_results)
    
    if not df.empty:
        df = df.sort_values('disparity', ascending=False)
    
    return df


def identify_most_disadvantaged_groups(
    y_pred: np.ndarray,
    sensitive_features_dict: Dict[str, np.ndarray],
    min_group_size: int = 30,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Identify intersectional groups with lowest positive prediction rates.
    
    Args:
        y_pred: Predicted labels
        sensitive_features_dict: Dictionary of attribute_name -> array
        min_group_size: Minimum samples per group
        top_n: Number of groups to return
        
    Returns:
        DataFrame with most disadvantaged groups
    """
    # Create intersectional groups
    attribute_names = list(sensitive_features_dict.keys())
    attribute_arrays = list(sensitive_features_dict.values())
    
    intersectional_groups, _ = create_intersectional_groups(
        *attribute_arrays,
        labels=attribute_names
    )
    
    # Compute stats per group
    unique_groups = np.unique(intersectional_groups)
    
    group_results = []
    for group in unique_groups:
        mask = intersectional_groups == group
        size = np.sum(mask)
        
        if size >= min_group_size:
            positive_rate = np.mean(y_pred[mask])
            
            group_results.append({
                'group': group,
                'size': size,
                'positive_rate': positive_rate
            })
    
    df = pd.DataFrame(group_results)
    
    if not df.empty:
        df = df.sort_values('positive_rate', ascending=True)
        df = df.head(top_n)
    
    return df


def test_intersectional_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features_dict: Dict[str, np.ndarray],
    metric_name: str = 'demographic_parity',
    threshold: float = 0.1,
    min_group_size: int = 30,
    multiple_comparison_correction: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive intersectional fairness test with optional multiple comparison correction.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features_dict: Dictionary of attribute_name -> array
        metric_name: Fairness metric
        threshold: Fairness threshold
        min_group_size: Minimum samples per group
        multiple_comparison_correction: 'bonferroni' or 'benjamini-hochberg' (None = no correction)
        
    Returns:
        Dictionary with comprehensive test results
    """
    from .fairness_analyzer_simple import FairnessAnalyzer
    
    analyzer = FairnessAnalyzer()
    
    # Get intersectional metrics
    intersectional_results = compute_intersectional_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features_dict=sensitive_features_dict,
        metric_name=metric_name,
        min_group_size=min_group_size,
        threshold=threshold,
        analyzer=analyzer
    )
    
    # Get pairwise disparities
    pairwise_disparities = analyze_pairwise_disparities(
        y_pred=y_pred,
        sensitive_features_dict=sensitive_features_dict,
        min_group_size=min_group_size
    )
    
    # Identify disadvantaged groups
    disadvantaged = identify_most_disadvantaged_groups(
        y_pred=y_pred,
        sensitive_features_dict=sensitive_features_dict,
        min_group_size=min_group_size,
        top_n=5
    )
    
    # Apply multiple comparison correction if requested
    if multiple_comparison_correction and not pairwise_disparities.empty:
        n_comparisons = len(pairwise_disparities)
        
        if multiple_comparison_correction.lower() == 'bonferroni':
            adjusted_threshold = threshold / n_comparisons
            pairwise_disparities['adjusted_threshold'] = adjusted_threshold
            pairwise_disparities['passes_adjusted'] = pairwise_disparities['disparity'] <= adjusted_threshold
            
        elif multiple_comparison_correction.lower() == 'benjamini-hochberg':
            # Benjamini-Hochberg FDR control
            pairwise_disparities = pairwise_disparities.sort_values('disparity', ascending=False)
            pairwise_disparities['rank'] = range(1, len(pairwise_disparities) + 1)
            pairwise_disparities['bh_threshold'] = (pairwise_disparities['rank'] / n_comparisons) * threshold
            pairwise_disparities['passes_bh'] = pairwise_disparities['disparity'] <= pairwise_disparities['bh_threshold']
    
    return {
        'intersectional_metrics': intersectional_results,
        'pairwise_disparities': pairwise_disparities,
        'disadvantaged_groups': disadvantaged,
        'multiple_comparison_correction': multiple_comparison_correction,
        'summary': {
            'n_reliable_groups': intersectional_results['n_reliable_groups'],
            'n_small_groups': intersectional_results['n_small_groups'],
            'max_disparity': intersectional_results['max_disparity'],
            'n_pairwise_comparisons': len(pairwise_disparities) if not pairwise_disparities.empty else 0
        }
    }


def generate_intersectional_report(
    test_results: Dict[str, Any],
    save_path: Optional[str] = None
) -> str:
    """
    Generate human-readable intersectional fairness report.
    
    Args:
        test_results: Output from test_intersectional_fairness()
        save_path: Optional path to save report
        
    Returns:
        Report string
    """
    lines = [
        "=" * 80,
        "INTERSECTIONAL FAIRNESS ANALYSIS REPORT",
        "=" * 80,
        "",
        "SUMMARY",
        "-" * 80,
        f"Reliable groups (n >= min_size): {test_results['summary']['n_reliable_groups']}",
        f"Small groups (excluded): {test_results['summary']['n_small_groups']}",
        f"Pairwise comparisons: {test_results['summary']['n_pairwise_comparisons']}",
        f"Maximum disparity: {test_results['summary']['max_disparity']:.4f}" if test_results['summary']['max_disparity'] else "Maximum disparity: N/A",
        ""
    ]
    
    # Most disadvantaged groups
    if not test_results['disadvantaged_groups'].empty:
        lines.extend([
            "MOST DISADVANTAGED GROUPS",
            "-" * 80
        ])
        
        for _, row in test_results['disadvantaged_groups'].iterrows():
            lines.append(
                f"  {row['group']}: "
                f"positive_rate={row['positive_rate']:.3f}, "
                f"n={row['size']}"
            )
        lines.append("")
    
    # Largest disparities
    if not test_results['pairwise_disparities'].empty:
        lines.extend([
            "LARGEST PAIRWISE DISPARITIES (Top 5)",
            "-" * 80
        ])
        
        top_5 = test_results['pairwise_disparities'].head(5)
        for _, row in top_5.iterrows():
            lines.append(
                f"  {row['group_1']} vs {row['group_2']}: "
                f"disparity={row['disparity']:.4f}"
            )
        lines.append("")
    
    # Multiple comparison correction
    if test_results['multiple_comparison_correction']:
        lines.extend([
            f"MULTIPLE COMPARISON CORRECTION: {test_results['multiple_comparison_correction']}",
            "-" * 80,
            "Adjusted thresholds applied to control family-wise error rate.",
            ""
        ])
    
    lines.extend([
        "=" * 80,
        "END OF REPORT",
        "=" * 80
    ])
    
    report = "\n".join(lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to: {save_path}")
    
    return report