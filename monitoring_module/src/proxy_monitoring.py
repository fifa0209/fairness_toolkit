"""
Proxy-Based Monitoring for Fairness Metrics

Enables fairness monitoring in jurisdictions where collecting sensitive
attributes is prohibited by using legally permissible proxies.

Author: FairML Consulting
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProxyMapping:
    """Configuration for proxy-based monitoring."""
    
    proxy_column: str  # e.g., 'zip_code', 'geography'
    proxy_to_group: Dict[str, int]  # Mapping proxy values to sensitive groups
    confidence_level: float = 0.8  # Confidence in the proxy mapping
    metadata: Dict = None


class ProxyBasedMonitor:
    """
    Monitor fairness using proxy attributes instead of direct sensitive features.
    
    Useful for:
    - GDPR-compliant monitoring in EU
    - Jurisdictions prohibiting collection of sensitive attributes
    - Privacy-preserving fairness tracking
    
    Example:
        >>> # Use ZIP codes as proxy for race/ethnicity
        >>> proxy_config = ProxyMapping(
        ...     proxy_column='zip_code',
        ...     proxy_to_group={
        ...         '10001': 0,  # Majority group
        ...         '10002': 1,  # Minority group
        ...         # ... more mappings
        ...     }
        ... )
        >>> 
        >>> monitor = ProxyBasedMonitor(proxy_config)
        >>> metrics = monitor.compute_proxy_metrics(df)
    """
    
    def __init__(
        self,
        proxy_mapping: ProxyMapping,
        uncertainty_adjustment: bool = True
    ):
        """
        Initialize proxy-based monitor.
        
        Args:
            proxy_mapping: Configuration for proxy mapping
            uncertainty_adjustment: Whether to adjust metrics for proxy uncertainty
        """
        self.proxy_mapping = proxy_mapping
        self.uncertainty_adjustment = uncertainty_adjustment
        
        logger.info(
            f"Initialized ProxyBasedMonitor with proxy: {proxy_mapping.proxy_column}"
        )
    
    def infer_sensitive_attribute(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Infer sensitive attributes from proxy data.
        
        Args:
            df: DataFrame with proxy column
        
        Returns:
            Tuple of (inferred_sensitive, confidence_scores)
        """
        proxy_col = self.proxy_mapping.proxy_column
        
        if proxy_col not in df.columns:
            raise ValueError(f"Proxy column '{proxy_col}' not found in data")
        
        # Map proxy values to sensitive groups
        inferred = df[proxy_col].map(self.proxy_mapping.proxy_to_group)
        
        # Handle unmapped values
        n_unmapped = inferred.isna().sum()
        if n_unmapped > 0:
            logger.warning(
                f"{n_unmapped} samples have unmapped proxy values. "
                f"Assigning to default group 0."
            )
            inferred = inferred.fillna(0).astype(int)
        
        # Confidence scores (uniform for now, could be enhanced)
        confidence = np.full(len(df), self.proxy_mapping.confidence_level)
        
        return inferred.values, confidence
    
    def compute_proxy_metrics(
        self,
        df: pd.DataFrame,
        y_pred_col: str = 'y_pred',
        y_true_col: str = 'y_true'
    ) -> Dict[str, any]:
        """
        Compute fairness metrics using proxy attributes.
        
        Args:
            df: DataFrame with predictions and proxy column
            y_pred_col: Prediction column name
            y_true_col: True label column name
        
        Returns:
            Dictionary with proxy-based metrics and uncertainty estimates
        """
        from measurement_module.src.metrics_engine import compute_metric
        
        # Infer sensitive attributes
        inferred_sensitive, confidence = self.infer_sensitive_attribute(df)
        
        y_pred = df[y_pred_col].values
        y_true = df[y_true_col].values
        
        # Compute base metrics
        results = {}
        
        for metric_name in ['demographic_parity', 'equalized_odds']:
            try:
                value, group_metrics, group_sizes = compute_metric(
                    metric_name, y_true, y_pred, inferred_sensitive
                )
                
                # Adjust for proxy uncertainty if enabled
                if self.uncertainty_adjustment:
                    uncertainty = self._estimate_uncertainty(
                        value, confidence, group_sizes
                    )
                else:
                    uncertainty = 0.0
                
                results[metric_name] = {
                    'value': value,
                    'uncertainty': uncertainty,
                    'lower_bound': max(0, value - uncertainty),
                    'upper_bound': value + uncertainty,
                    'group_metrics': group_metrics,
                    'proxy_based': True,
                    'confidence': confidence.mean(),
                }
                
            except Exception as e:
                logger.error(f"Failed to compute proxy metric {metric_name}: {e}")
        
        return results
    
    def _estimate_uncertainty(
        self,
        metric_value: float,
        confidence_scores: np.ndarray,
        group_sizes: Dict[int, int]
    ) -> float:
        """
        Estimate uncertainty in metric due to proxy imperfection.
        
        Uses a simple model: uncertainty scales with (1 - confidence) and
        inversely with sample size.
        """
        avg_confidence = confidence_scores.mean()
        total_samples = sum(group_sizes.values())
        
        # Uncertainty from proxy imperfection
        proxy_uncertainty = metric_value * (1 - avg_confidence)
        
        # Uncertainty from sample size (using sqrt(n) rule)
        sampling_uncertainty = 1.96 / np.sqrt(total_samples)  # 95% CI
        
        # Combined uncertainty
        total_uncertainty = np.sqrt(proxy_uncertainty**2 + sampling_uncertainty**2)
        
        return total_uncertainty
    
    def validate_proxy_quality(
        self,
        df: pd.DataFrame,
        true_sensitive_col: str
    ) -> Dict[str, float]:
        """
        Validate proxy quality against ground truth (when available).
        
        Args:
            df: DataFrame with both proxy and true sensitive attributes
            true_sensitive_col: Column with true sensitive attribute
        
        Returns:
            Dictionary with validation metrics
        """
        inferred, _ = self.infer_sensitive_attribute(df)
        true_sensitive = df[true_sensitive_col].values
        
        # Compute accuracy metrics
        accuracy = (inferred == true_sensitive).mean()
        
        # Per-group accuracy
        group_accuracy = {}
        for group in np.unique(true_sensitive):
            mask = true_sensitive == group
            group_acc = (inferred[mask] == true_sensitive[mask]).mean()
            group_accuracy[int(group)] = group_acc
        
        # Confusion matrix elements
        from sklearn.metrics import confusion_matrix, cohen_kappa_score
        
        cm = confusion_matrix(true_sensitive, inferred)
        kappa = cohen_kappa_score(true_sensitive, inferred)
        
        return {
            'overall_accuracy': accuracy,
            'group_accuracy': group_accuracy,
            'cohen_kappa': kappa,
            'confusion_matrix': cm.tolist(),
            'recommendation': self._interpret_validation(accuracy, kappa)
        }
    
    def _interpret_validation(self, accuracy: float, kappa: float) -> str:
        """Generate recommendation based on validation results."""
        if accuracy >= 0.90 and kappa >= 0.80:
            return "EXCELLENT: Proxy is highly reliable for fairness monitoring"
        elif accuracy >= 0.80 and kappa >= 0.60:
            return "GOOD: Proxy is acceptable with uncertainty adjustments"
        elif accuracy >= 0.70:
            return "FAIR: Use with caution and wide confidence intervals"
        else:
            return "POOR: Proxy not recommended - seek alternative approach"


class GeographicProxyBuilder:
    """
    Build proxy mappings from geographic data (ZIP codes, census tracts).
    
    Example:
        >>> builder = GeographicProxyBuilder()
        >>> mapping = builder.build_from_census_data(
        ...     census_df,
        ...     geography_col='zip_code',
        ...     demographic_cols=['pct_minority', 'median_income']
        ... )
    """
    
    def __init__(self):
        """Initialize geographic proxy builder."""
        logger.info("Initialized GeographicProxyBuilder")
    
    def build_from_census_data(
        self,
        census_df: pd.DataFrame,
        geography_col: str,
        demographic_cols: List[str],
        threshold: float = 0.5
    ) -> ProxyMapping:
        """
        Build proxy mapping from census demographic data.
        
        Args:
            census_df: Census data with geographic identifiers
            geography_col: Column with geography (ZIP, tract, etc.)
            demographic_cols: Columns with demographic percentages
            threshold: Threshold for classification (e.g., >50% = group 1)
        
        Returns:
            ProxyMapping configuration
        """
        proxy_to_group = {}
        
        for _, row in census_df.iterrows():
            geo_id = str(row[geography_col])
            
            # Simple majority-based classification
            # For more sophisticated: use clustering or multiple thresholds
            minority_pct = row.get('pct_minority', 0)
            
            if minority_pct >= threshold:
                proxy_to_group[geo_id] = 1  # Minority-majority area
            else:
                proxy_to_group[geo_id] = 0  # Majority area
        
        # Estimate confidence based on data quality
        confidence = self._estimate_mapping_confidence(census_df, demographic_cols)
        
        mapping = ProxyMapping(
            proxy_column=geography_col,
            proxy_to_group=proxy_to_group,
            confidence_level=confidence,
            metadata={
                'source': 'census_data',
                'threshold': threshold,
                'n_geographies': len(proxy_to_group)
            }
        )
        
        logger.info(f"Built proxy mapping with {len(proxy_to_group)} geographies")
        
        return mapping
    
    def _estimate_mapping_confidence(
        self,
        census_df: pd.DataFrame,
        demographic_cols: List[str]
    ) -> float:
        """
        Estimate confidence in proxy mapping.
        
        Based on:
        - Data completeness
        - Geographic granularity
        - Within-geography homogeneity
        """
        # Completeness check
        completeness = 1 - census_df[demographic_cols].isna().mean().mean()
        
        # Simple confidence model
        # In production, this would be more sophisticated
        confidence = min(0.95, 0.7 + 0.2 * completeness)
        
        return confidence


class PrivacyPreservingReporter:
    """
    Generate fairness reports with privacy preservation.
    
    Implements:
    - K-anonymity for small groups
    - Differential privacy noise addition
    - Aggregation thresholds
    """
    
    def __init__(
        self,
        k_threshold: int = 10,
        add_noise: bool = True,
        epsilon: float = 1.0
    ):
        """
        Initialize privacy-preserving reporter.
        
        Args:
            k_threshold: Minimum group size for reporting
            add_noise: Whether to add differential privacy noise
            epsilon: Privacy budget (lower = more private)
        """
        self.k_threshold = k_threshold
        self.add_noise = add_noise
        self.epsilon = epsilon
        
        logger.info(
            f"Initialized PrivacyPreservingReporter "
            f"(k={k_threshold}, ε={epsilon})"
        )
    
    def safe_report_metrics(
        self,
        group_metrics: Dict[str, float],
        group_sizes: Dict[str, int]
    ) -> Dict[str, any]:
        """
        Generate privacy-safe metric report.
        
        Args:
            group_metrics: Metrics per group
            group_sizes: Sample sizes per group
        
        Returns:
            Privacy-safe report with suppressed/noised values
        """
        safe_report = {}
        
        for group, metric_value in group_metrics.items():
            group_size = group_sizes.get(group, 0)
            
            # K-anonymity: suppress small groups
            if group_size < self.k_threshold:
                safe_report[group] = {
                    'value': None,
                    'suppressed': True,
                    'reason': f'Group size ({group_size}) below threshold ({self.k_threshold})'
                }
            else:
                # Add differential privacy noise if enabled
                if self.add_noise:
                    noise = self._laplace_noise(metric_value)
                    noised_value = metric_value + noise
                else:
                    noised_value = metric_value
                
                safe_report[group] = {
                    'value': noised_value,
                    'suppressed': False,
                    'sample_size_range': self._size_range(group_size),
                    'privacy_applied': self.add_noise
                }
        
        return safe_report
    
    def _laplace_noise(self, value: float) -> float:
        """Add Laplace noise for differential privacy."""
        # Sensitivity for bounded metrics (0-1 range)
        sensitivity = 1.0
        
        # Scale parameter
        scale = sensitivity / self.epsilon
        
        # Sample from Laplace distribution
        noise = np.random.laplace(0, scale)
        
        return noise
    
    def _size_range(self, size: int) -> str:
        """Convert exact size to privacy-safe range."""
        if size < 50:
            return "10-50"
        elif size < 100:
            return "50-100"
        elif size < 500:
            return "100-500"
        elif size < 1000:
            return "500-1000"
        else:
            return "1000+"


# Convenience function
def create_geographic_proxy_monitor(
    census_df: pd.DataFrame,
    geography_col: str = 'zip_code',
    demographic_threshold: float = 0.5
) -> ProxyBasedMonitor:
    """
    Convenience function to create a geographic proxy monitor.
    
    Args:
        census_df: Census data with demographics by geography
        geography_col: Geographic identifier column
        demographic_threshold: Threshold for group classification
    
    Returns:
        Configured ProxyBasedMonitor
    """
    builder = GeographicProxyBuilder()
    
    mapping = builder.build_from_census_data(
        census_df=census_df,
        geography_col=geography_col,
        demographic_cols=['pct_minority'],
        threshold=demographic_threshold
    )
    
    monitor = ProxyBasedMonitor(
        proxy_mapping=mapping,
        uncertainty_adjustment=True
    )
    
    return monitor