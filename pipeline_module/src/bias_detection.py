"""
Bias Detection Engine - Systematic bias identification in data pipelines.

Detects three types of bias:
1. Representation bias - Demographic distribution mismatches
2. Statistical disparity - Feature distributions differ across groups
3. Proxy detection - Features correlated with protected attributes

48-hour scope: Simple, reliable checks with clear thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from datetime import datetime

from shared.schemas import BiasDetectionResult
from shared.validation import validate_dataframe, validate_protected_attribute
from shared.logging import get_logger, log_bias_detection
from shared.constants import BIAS_DETECTION_THRESHOLDS

logger = get_logger(__name__)


class BiasDetector:
    """
    Detects various types of bias in datasets.
    
    Example:
        >>> detector = BiasDetector()
        >>> result = detector.detect_representation_bias(
        ...     df, 
        ...     protected_attribute='gender',
        ...     reference_distribution={'Female': 0.51, 'Male': 0.49}
        ... )
        >>> if result.detected:
        ...     print(f"Bias detected: {result.severity}")
    """
    
    def __init__(
        self,
        representation_threshold: float = 0.2,
        proxy_threshold: float = 0.5,
        statistical_alpha: float = 0.05,
    ):
        """
        Initialize BiasDetector.
        
        Args:
            representation_threshold: Max acceptable difference in representation (0.2 = 20%)
            proxy_threshold: Min correlation to flag as proxy (0.5 = moderate)
            statistical_alpha: Significance level for statistical tests
        """
        self.representation_threshold = representation_threshold
        self.proxy_threshold = proxy_threshold
        self.statistical_alpha = statistical_alpha
        
        logger.info(
            f"BiasDetector initialized: "
            f"repr_threshold={representation_threshold}, "
            f"proxy_threshold={proxy_threshold}"
        )
    
    def detect_representation_bias(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        reference_distribution: Optional[Dict[str, float]] = None,
    ) -> BiasDetectionResult:
        """
        Detect representation bias - when demographic distribution differs from reference.
        
        Example: If population is 51% female but dataset is 30% female.
        
        Args:
            df: DataFrame containing data
            protected_attribute: Name of protected attribute column
            reference_distribution: Expected distribution (e.g., {'Female': 0.51, 'Male': 0.49})
                                   If None, assumes uniform distribution
            
        Returns:
            BiasDetectionResult with detection status and severity
        """
        logger.info(f"Detecting representation bias for '{protected_attribute}'...")
        
        # Validate input
        validate_dataframe(df, required_columns=[protected_attribute])
        validate_protected_attribute(df, protected_attribute, min_group_size=10)
        
        # Get actual distribution
        actual_counts = df[protected_attribute].value_counts()
        actual_distribution = (actual_counts / len(df)).to_dict()
        
        # If no reference provided, assume uniform
        if reference_distribution is None:
            n_groups = len(actual_counts)
            reference_distribution = {k: 1.0/n_groups for k in actual_counts.index}
            logger.info("Using uniform reference distribution")
        
        # Compute differences
        differences = {}
        max_diff = 0.0
        affected_groups = []
        
        for group in actual_distribution.keys():
            actual = actual_distribution[group]
            reference = reference_distribution.get(group, 1.0 / len(actual_distribution))
            diff = abs(actual - reference)
            differences[group] = diff
            
            if diff > max_diff:
                max_diff = diff
            
            if diff > self.representation_threshold * 0.5:  # Flag if above half threshold
                affected_groups.append(str(group))
        
        # Determine severity
        detected = max_diff > self.representation_threshold
        
        if max_diff > BIAS_DETECTION_THRESHOLDS['representation_bias']['severe']:
            severity = 'high'
        elif max_diff > BIAS_DETECTION_THRESHOLDS['representation_bias']['moderate']:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate recommendations
        recommendations = []
        if detected:
            recommendations.append(
                f"Consider resampling or reweighting to match reference distribution"
            )
            recommendations.append(
                f"Maximum difference: {max_diff:.1%} (threshold: {self.representation_threshold:.1%})"
            )
            for group in affected_groups:
                actual_pct = actual_distribution[group] * 100
                ref_pct = reference_distribution.get(group, 0) * 100
                recommendations.append(
                    f"Group '{group}': {actual_pct:.1f}% (expected: {ref_pct:.1f}%)"
                )
        
        # Create result
        result = BiasDetectionResult(
            bias_type='representation',
            detected=detected,
            severity=severity,
            affected_groups=affected_groups,
            evidence={
                'max_difference': max_diff,
                'threshold': self.representation_threshold,
                'actual_distribution': actual_distribution,
                'reference_distribution': reference_distribution,
                'differences': differences,
            },
            recommendations=recommendations,
        )
        
        # Log result
        log_bias_detection(logger, 'representation', detected, severity, affected_groups)
        
        return result
    
    def detect_proxy_variables(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        feature_columns: Optional[List[str]] = None,
    ) -> BiasDetectionResult:
        """
        Detect proxy variables - features highly correlated with protected attribute.
        
        Example: ZIP code highly correlated with race.
        
        Args:
            df: DataFrame containing data
            protected_attribute: Name of protected attribute column
            feature_columns: List of feature columns to check (None = all numeric)
            
        Returns:
            BiasDetectionResult with detected proxy variables
        """
        logger.info(f"Detecting proxy variables for '{protected_attribute}'...")
        
        # Validate input
        validate_dataframe(df, required_columns=[protected_attribute])
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if protected_attribute in feature_columns:
                feature_columns.remove(protected_attribute)
        
        if len(feature_columns) == 0:
            logger.warning("No numeric features to check for proxy variables")
            return BiasDetectionResult(
                bias_type='proxy',
                detected=False,
                severity='low',
                affected_groups=[],
                evidence={},
                recommendations=["No numeric features available for proxy detection"],
            )
        
        # Encode protected attribute numerically for correlation
        protected_encoded = pd.Categorical(df[protected_attribute]).codes
        
        # Compute correlations
        correlations = {}
        proxy_features = []
        
        for col in feature_columns:
            if col == protected_attribute:
                continue
            
            try:
                # Handle missing values
                mask = df[col].notna() & (protected_encoded >= 0)
                if mask.sum() < 10:
                    continue
                
                # Compute correlation
                corr, p_value = stats.spearmanr(
                    df.loc[mask, col],
                    protected_encoded[mask]
                )
                
                correlations[col] = {
                    'correlation': abs(corr),
                    'p_value': p_value,
                    'significant': p_value < self.statistical_alpha,
                }
                
                # Flag as proxy if correlation is high and significant
                if abs(corr) > self.proxy_threshold and p_value < self.statistical_alpha:
                    proxy_features.append(col)
                    logger.info(f"Proxy detected: {col} (r={corr:.3f}, p={p_value:.4f})")
                
            except Exception as e:
                logger.warning(f"Failed to compute correlation for {col}: {e}")
                continue
        
        # Determine severity
        detected = len(proxy_features) > 0
        
        if detected:
            max_corr = max(correlations[f]['correlation'] for f in proxy_features)
            if max_corr > BIAS_DETECTION_THRESHOLDS['proxy_correlation']['high']:
                severity = 'high'
            elif max_corr > BIAS_DETECTION_THRESHOLDS['proxy_correlation']['medium']:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'low'
        
        # Generate recommendations
        recommendations = []
        if detected:
            recommendations.append(
                f"Found {len(proxy_features)} potential proxy variable(s)"
            )
            recommendations.append(
                "Consider removing or transforming these features to reduce indirect discrimination"
            )
            for feat in proxy_features[:5]:  # Top 5
                corr = correlations[feat]['correlation']
                recommendations.append(
                    f"Feature '{feat}': correlation = {corr:.3f}"
                )
        else:
            recommendations.append("No significant proxy variables detected")
        
        # Create result
        result = BiasDetectionResult(
            bias_type='proxy',
            detected=detected,
            severity=severity,
            affected_groups=proxy_features,
            evidence={
                'correlations': correlations,
                'threshold': self.proxy_threshold,
                'n_proxies': len(proxy_features),
                'proxy_features': proxy_features,
            },
            recommendations=recommendations,
        )
        
        # Log result
        log_bias_detection(logger, 'proxy', detected, severity, proxy_features)
        
        return result
    
    def detect_statistical_disparity(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        feature_columns: Optional[List[str]] = None,
    ) -> BiasDetectionResult:
        """
        Detect statistical disparity - features distributed differently across groups.
        
        Example: Average income significantly different between groups.
        
        Args:
            df: DataFrame containing data
            protected_attribute: Name of protected attribute column
            feature_columns: List of feature columns to check (None = all numeric)
            
        Returns:
            BiasDetectionResult with features showing disparity
        """
        logger.info(f"Detecting statistical disparity for '{protected_attribute}'...")
        
        # Validate input
        validate_dataframe(df, required_columns=[protected_attribute])
        validate_protected_attribute(df, protected_attribute, min_group_size=10)
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if protected_attribute in feature_columns:
                feature_columns.remove(protected_attribute)
        
        if len(feature_columns) == 0:
            logger.warning("No numeric features to check for statistical disparity")
            return BiasDetectionResult(
                bias_type='measurement',
                detected=False,
                severity='low',
                affected_groups=[],
                evidence={},
                recommendations=["No numeric features available for disparity detection"],
            )
        
        # Get groups
        groups = df[protected_attribute].unique()
        if len(groups) != 2:
            logger.warning(f"Expected 2 groups, got {len(groups)}. Using first two.")
            groups = groups[:2]
        
        # Test each feature for disparity
        disparate_features = []
        test_results = {}
        
        for col in feature_columns:
            try:
                # Get data for each group
                group_data = []
                for group in groups:
                    data = df[df[protected_attribute] == group][col].dropna()
                    if len(data) >= 10:  # Need minimum samples
                        group_data.append(data)
                
                if len(group_data) != 2:
                    continue
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(
                    group_data[0], group_data[1], alternative='two-sided'
                )
                
                # Effect size (Cohen's d)
                mean1, mean2 = group_data[0].mean(), group_data[1].mean()
                std1, std2 = group_data[0].std(), group_data[1].std()
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                test_results[col] = {
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'significant': p_value < self.statistical_alpha,
                    'means': {str(groups[0]): mean1, str(groups[1]): mean2},
                }
                
                # Flag if statistically significant and meaningful effect
                if p_value < self.statistical_alpha and effect_size > 0.2:
                    disparate_features.append(col)
                    logger.info(
                        f"Disparity detected: {col} "
                        f"(p={p_value:.4f}, d={effect_size:.3f})"
                    )
                
            except Exception as e:
                logger.warning(f"Failed to test disparity for {col}: {e}")
                continue
        
        # Determine severity
        detected = len(disparate_features) > 0
        
        if detected:
            max_effect = max(test_results[f]['effect_size'] for f in disparate_features)
            if max_effect > 0.8:
                severity = 'high'
            elif max_effect > 0.5:
                severity = 'medium'
            else:
                severity = 'low'
        else:
            severity = 'low'
        
        # Generate recommendations
        recommendations = []
        if detected:
            recommendations.append(
                f"Found {len(disparate_features)} feature(s) with significant disparity"
            )
            recommendations.append(
                "Consider feature normalization or group-specific preprocessing"
            )
            for feat in disparate_features[:5]:  # Top 5
                means = test_results[feat]['means']
                recommendations.append(
                    f"Feature '{feat}': means = {means}"
                )
        else:
            recommendations.append("No significant statistical disparities detected")
        
        # Create result
        result = BiasDetectionResult(
            bias_type='measurement',
            detected=detected,
            severity=severity,
            affected_groups=disparate_features,
            evidence={
                'test_results': test_results,
                'alpha': self.statistical_alpha,
                'n_disparate': len(disparate_features),
                'disparate_features': disparate_features,
            },
            recommendations=recommendations,
        )
        
        # Log result
        log_bias_detection(logger, 'measurement', detected, severity, disparate_features)
        
        return result
    
    def detect_all_bias_types(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        reference_distribution: Optional[Dict[str, float]] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> Dict[str, BiasDetectionResult]:
        """
        Run all bias detection checks at once.
        
        Args:
            df: DataFrame containing data
            protected_attribute: Name of protected attribute column
            reference_distribution: Expected demographic distribution
            feature_columns: Features to check (None = all numeric)
            
        Returns:
            Dictionary mapping bias type -> BiasDetectionResult
        """
        logger.info("Running comprehensive bias detection...")
        
        results = {}
        
        # Representation bias
        try:
            results['representation'] = self.detect_representation_bias(
                df, protected_attribute, reference_distribution
            )
        except Exception as e:
            logger.error(f"Representation bias detection failed: {e}")
        
        # Proxy detection
        try:
            results['proxy'] = self.detect_proxy_variables(
                df, protected_attribute, feature_columns
            )
        except Exception as e:
            logger.error(f"Proxy detection failed: {e}")
        
        # Statistical disparity
        try:
            results['statistical_disparity'] = self.detect_statistical_disparity(
                df, protected_attribute, feature_columns
            )
        except Exception as e:
            logger.error(f"Statistical disparity detection failed: {e}")
        
        # Summary
        total_detected = sum(1 for r in results.values() if r.detected)
        logger.info(f"Bias detection complete: {total_detected}/{len(results)} types detected")
        
        return results