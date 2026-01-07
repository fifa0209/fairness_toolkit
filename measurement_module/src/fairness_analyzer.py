"""
Fairness Analyzer - Unified API for fairness measurement.

High-level interface that combines metrics computation with statistical validation.
This is the main entry point for the measurement module.

Usage:
    analyzer = FairnessAnalyzer()
    result = analyzer.compute_metric(
        y_true, y_pred, sensitive_features,
        metric='demographic_parity'
    )
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from datetime import datetime

from shared.schemas import FairnessMetricResult, DatasetMetadata
from shared.constants import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_BOOTSTRAP_SAMPLES,
    MIN_GROUP_SIZE,
)
from shared.validation import (
    validate_dataframe,
    validate_protected_attribute,
    validate_predictions,
)
from shared.logging import get_logger, log_fairness_result

from measurement_module.src.metrics_engine import (
    compute_metric,
    interpret_metric,
    compute_group_metrics,
)
from measurement_module.src.statistical_validation import (
    bootstrap_confidence_interval,
    compute_effect_size_cohens_d,
)

logger = get_logger(__name__)


class FairnessAnalyzer:
    """
    Unified interface for fairness measurement with statistical validation.
    
    Example:
        >>> analyzer = FairnessAnalyzer()
        >>> result = analyzer.compute_metric(
        ...     y_true=y_test,
        ...     y_pred=y_pred,
        ...     sensitive_features=X_test['gender'],
        ...     metric='demographic_parity',
        ...     threshold=0.1
        ... )
        >>> print(f"Fair: {result.is_fair}, CI: {result.confidence_interval}")
    """
    
    def __init__(
        self,
        confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
        bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
        min_group_size: int = MIN_GROUP_SIZE,
    ):
        """
        Initialize FairnessAnalyzer.
        
        Args:
            confidence_level: Confidence level for intervals (e.g., 0.95)
            bootstrap_samples: Number of bootstrap samples
            min_group_size: Minimum samples per group
        """
        self.confidence_level = confidence_level
        self.bootstrap_samples = bootstrap_samples
        self.min_group_size = min_group_size
        
        logger.info(
            f"FairnessAnalyzer initialized: "
            f"CI={confidence_level}, bootstrap={bootstrap_samples}"
        )
    
    def compute_metric(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
        metric: str = "demographic_parity",
        threshold: float = 0.1,
        compute_ci: bool = True,
        compute_effect_size: bool = True,
    ) -> FairnessMetricResult:
        """
        Compute fairness metric with full statistical validation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute
            metric: Metric name ('demographic_parity', 'equalized_odds', 'equal_opportunity')
            threshold: Fairness threshold for pass/fail
            compute_ci: Whether to compute bootstrap confidence interval
            compute_effect_size: Whether to compute effect size
            
        Returns:
            FairnessMetricResult with metric, CI, interpretation
            
        Raises:
            ValidationError: If inputs are invalid
        """
        logger.info(f"Computing {metric}...")
        
        # Convert to numpy
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)
        
        # Validate inputs
        validate_predictions(y_true, y_pred)
        
        # Check group sizes
        groups, counts = np.unique(sensitive_features, return_counts=True)
        group_sizes = {f"Group_{g}": int(c) for g, c in zip(groups, counts)}
        
        min_size = min(counts)
        if min_size < self.min_group_size:
            logger.warning(
                f"Small group detected: {min_size} < {self.min_group_size}. "
                f"Results may be unreliable."
            )
        
        # Compute point estimate
        metric_value, group_metrics, _ = compute_metric(
            metric, y_true, y_pred, sensitive_features
        )
        
        # Compute confidence interval
        if compute_ci:
            try:
                _, ci = bootstrap_confidence_interval(
                    lambda yt, yp, sf: compute_metric(metric, yt, yp, sf),
                    y_true,
                    y_pred,
                    sensitive_features,
                    n_bootstrap=self.bootstrap_samples,
                    confidence_level=self.confidence_level,
                )
            except Exception as e:
                logger.error(f"Bootstrap CI failed: {e}")
                ci = (metric_value, metric_value)  # Fallback
        else:
            ci = (metric_value, metric_value)
        
        # Compute effect size (for demographic parity)
        effect_size = None
        if compute_effect_size and metric == "demographic_parity":
            try:
                # Effect size based on group positive rates
                groups_list = sorted(groups)
                mask_0 = sensitive_features == groups_list[0]
                mask_1 = sensitive_features == groups_list[1]
                
                # Use predictions as outcomes for effect size
                effect_size = compute_effect_size_cohens_d(
                    y_pred[mask_0].astype(float),
                    y_pred[mask_1].astype(float)
                )
            except Exception as e:
                logger.error(f"Effect size computation failed: {e}")
        
        # Determine fairness
        is_fair = metric_value <= threshold
        
        # Generate interpretation
        interpretation = interpret_metric(metric, metric_value, threshold, group_metrics)
        
        # Create result object
        result = FairnessMetricResult(
            metric_name=metric,
            value=metric_value,
            confidence_interval=ci,
            group_metrics=group_metrics,
            group_sizes=group_sizes,
            interpretation=interpretation,
            is_fair=is_fair,
            threshold=threshold,
            effect_size=effect_size,
            timestamp=datetime.now(),
        )
        
        # Log result
        log_fairness_result(
            logger,
            metric,
            metric_value,
            is_fair,
            threshold,
            group_metrics,
        )
        
        return result
    
    def compute_all_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
        metrics: Optional[List[str]] = None,
        threshold: float = 0.1,
    ) -> Dict[str, FairnessMetricResult]:
        """
        Compute multiple fairness metrics at once.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute
            metrics: List of metric names (None = all available)
            threshold: Fairness threshold
            
        Returns:
            Dictionary mapping metric name -> FairnessMetricResult
        """
        if metrics is None:
            metrics = ["demographic_parity", "equalized_odds", "equal_opportunity"]
        
        results = {}
        
        for metric in metrics:
            try:
                result = self.compute_metric(
                    y_true, y_pred, sensitive_features,
                    metric=metric,
                    threshold=threshold,
                )
                results[metric] = result
            except Exception as e:
                logger.error(f"Failed to compute {metric}: {e}")
                continue
        
        return results
    
    def create_dataset_metadata(
        self,
        df: pd.DataFrame,
        protected_attribute: str,
        target_column: str,
        dataset_name: str = "dataset",
    ) -> DatasetMetadata:
        """
        Create metadata object for a dataset.
        
        Args:
            df: DataFrame containing data
            protected_attribute: Name of protected attribute column
            target_column: Name of target column
            dataset_name: Name for the dataset
            
        Returns:
            DatasetMetadata object
        """
        # Validate
        validate_dataframe(df, required_columns=[protected_attribute, target_column])
        validate_protected_attribute(df, protected_attribute, self.min_group_size)
        
        # Extract metadata
        group_dist = df[protected_attribute].value_counts().to_dict()
        group_dist = {f"Group_{k}": int(v) for k, v in group_dist.items()}
        
        class_balance = None
        if target_column in df.columns:
            class_balance = df[target_column].value_counts().to_dict()
            class_balance = {f"Class_{k}": int(v) for k, v in class_balance.items()}
        
        metadata = DatasetMetadata(
            name=dataset_name,
            n_samples=len(df),
            n_features=len(df.columns) - 2,  # Exclude protected + target
            task_type="binary_classification",
            protected_attribute=protected_attribute,
            protected_groups=list(group_dist.keys()),
            group_distribution=group_dist,
            class_balance=class_balance,
        )
        
        logger.info(
            f"Dataset metadata created: {metadata.n_samples} samples, "
            f"{len(metadata.protected_groups)} groups"
        )
        
        return metadata
    
    def analyze_group_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute detailed per-group classification metrics.
        
        Returns metrics like TPR, FPR, precision for each group.
        Useful for deep-dive analysis beyond single fairness metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features: Protected attribute
            
        Returns:
            Dictionary mapping group -> metrics dict
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        sensitive_features = np.asarray(sensitive_features)
        
        validate_predictions(y_true, y_pred)
        
        group_metrics = compute_group_metrics(y_true, y_pred, sensitive_features)
        
        logger.info("Computed detailed group metrics")
        for group, metrics in group_metrics.items():
            logger.info(f"  {group}: TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")
        
        return group_metrics
    
    def compute_intersectional_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        sensitive_features_1: Union[np.ndarray, pd.Series],
        sensitive_features_2: Union[np.ndarray, pd.Series],
        metric: str = "demographic_parity",
        threshold: float = 0.1,
    ) -> Dict[str, FairnessMetricResult]:
        """
        Compute fairness metrics for intersectional groups.
        
        Example: Analyze fairness across gender × race combinations.
        
        Note: This is a stretch goal for 48-hour demo.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_features_1: First protected attribute (e.g., gender)
            sensitive_features_2: Second protected attribute (e.g., race)
            metric: Fairness metric to compute
            threshold: Fairness threshold
            
        Returns:
            Dictionary of results for each intersectional group pair
        """
        logger.info("Computing intersectional fairness metrics...")
        
        # Convert to numpy
        sf1 = np.asarray(sensitive_features_1)
        sf2 = np.asarray(sensitive_features_2)
        
        # Create intersectional groups
        intersectional = np.char.add(
            sf1.astype(str),
            np.char.add('_', sf2.astype(str))
        )
        
        # Compute metric for intersectional groups
        result = self.compute_metric(
            y_true, y_pred, intersectional,
            metric=metric,
            threshold=threshold,
            compute_ci=False,  # Skip CI for speed
        )
        
        logger.warning(
            "Intersectional analysis requires larger samples. "
            "Interpret with caution if groups are small."
        )
        
        return {"intersectional": result}