"""
Validation utilities for fairness pipeline.
Input validation, data quality checks, and error handling.
"""

import numpy as np
import pandas as pd
from typing import Any, List, Optional, Union
from shared.constants import MIN_GROUP_SIZE, FAIRNESS_METRICS, TASK_TYPES


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
) -> None:
    """
    Validate DataFrame structure and content.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"Expected DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValidationError(
            f"DataFrame has {len(df)} rows, minimum {min_rows} required"
        )
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")
    
    # Check for empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        raise ValidationError(f"Columns contain only NaN values: {empty_cols}")


def validate_protected_attribute(
    df: pd.DataFrame,
    attribute: str,
    min_group_size: int = MIN_GROUP_SIZE,
) -> None:
    """
    Validate protected attribute column.
    
    Args:
        df: DataFrame containing the attribute
        attribute: Name of protected attribute column
        min_group_size: Minimum samples per group
        
    Raises:
        ValidationError: If validation fails
    """
    if attribute not in df.columns:
        raise ValidationError(f"Protected attribute '{attribute}' not found in DataFrame")
    
    # Check for missing values
    if df[attribute].isna().any():
        n_missing = df[attribute].isna().sum()
        raise ValidationError(
            f"Protected attribute '{attribute}' contains {n_missing} missing values"
        )
    
    # Check group sizes
    group_counts = df[attribute].value_counts()
    small_groups = group_counts[group_counts < min_group_size]
    
    if len(small_groups) > 0:
        raise ValidationError(
            f"Groups with insufficient samples (< {min_group_size}): "
            f"{small_groups.to_dict()}"
        )
    
    # Binary classification scope for 48-hour demo
    if len(group_counts) > 2:
        raise ValidationError(
            f"Protected attribute has {len(group_counts)} groups. "
            f"48-hour demo supports binary attributes only."
        )


def validate_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    task_type: str = "binary_classification",
) -> None:
    """
    Validate prediction arrays.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels or probabilities
        task_type: Type of task
        
    Raises:
        ValidationError: If validation fails
    """
    if task_type not in TASK_TYPES:
        raise ValidationError(f"Invalid task_type: {task_type}. Must be one of {TASK_TYPES}")
    
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if len(y_true) != len(y_pred):
        raise ValidationError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )
    
    if len(y_true) == 0:
        raise ValidationError("Empty prediction arrays")
    
    # Check for NaN/inf
    if np.any(np.isnan(y_true)) or np.any(np.isnan(y_pred)):
        raise ValidationError("Predictions contain NaN values")
    
    if np.any(np.isinf(y_true)) or np.any(np.isinf(y_pred)):
        raise ValidationError("Predictions contain infinite values")
    
    # Task-specific validation
    if task_type == "binary_classification":
        unique_true = np.unique(y_true)
        if not np.all(np.isin(unique_true, [0, 1])):
            raise ValidationError(
                f"Binary classification requires labels in {{0, 1}}, got {unique_true}"
            )
        
        # Check if y_pred is probabilities or labels
        if np.all(np.isin(y_pred, [0, 1])):
            # Hard predictions
            pass
        elif np.all((y_pred >= 0) & (y_pred <= 1)):
            # Probabilities (OK)
            pass
        else:
            raise ValidationError(
                "Binary predictions must be in {0, 1} or probabilities in [0, 1]"
            )


def validate_config(config: Any) -> List[str]:
    """
    Validate pipeline configuration.
    
    Args:
        config: PipelineConfig object
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Check required attributes
    required_attrs = ["target_column", "protected_attribute"]
    for attr in required_attrs:
        if not hasattr(config, attr) or not getattr(config, attr):
            errors.append(f"Missing required config: {attr}")
    
    # Validate fairness metrics
    if hasattr(config, "fairness_metrics"):
        invalid_metrics = [
            m for m in config.fairness_metrics 
            if m not in FAIRNESS_METRICS
        ]
        if invalid_metrics:
            errors.append(
                f"Invalid fairness metrics: {invalid_metrics}. "
                f"Valid options: {list(FAIRNESS_METRICS.keys())}"
            )
    
    # Validate thresholds
    if hasattr(config, "fairness_threshold"):
        if not 0 < config.fairness_threshold < 1:
            errors.append("fairness_threshold must be between 0 and 1")
    
    if hasattr(config, "confidence_level"):
        if not 0 < config.confidence_level < 1:
            errors.append("confidence_level must be between 0 and 1")
    
    # Validate bootstrap samples
    if hasattr(config, "bootstrap_samples"):
        if config.bootstrap_samples < 100:
            errors.append("bootstrap_samples should be at least 100")
    
    return errors


def validate_sample_weights(
    sample_weights: Union[np.ndarray, pd.Series],
    n_samples: int,
) -> None:
    """
    Validate sample weights array.
    
    Args:
        sample_weights: Array of sample weights
        n_samples: Expected number of samples
        
    Raises:
        ValidationError: If validation fails
    """
    sample_weights = np.asarray(sample_weights)
    
    if len(sample_weights) != n_samples:
        raise ValidationError(
            f"Sample weights length {len(sample_weights)} != n_samples {n_samples}"
        )
    
    if np.any(sample_weights < 0):
        raise ValidationError("Sample weights must be non-negative")
    
    if np.any(np.isnan(sample_weights)) or np.any(np.isinf(sample_weights)):
        raise ValidationError("Sample weights contain NaN or infinite values")
    
    if np.sum(sample_weights) == 0:
        raise ValidationError("Sum of sample weights is zero")


def validate_group_metrics(
    group_metrics: dict,
    protected_groups: List[str],
) -> None:
    """
    Validate per-group metric dictionary.
    
    Args:
        group_metrics: Dictionary of group -> metric value
        protected_groups: Expected list of group names
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(group_metrics, dict):
        raise ValidationError(f"group_metrics must be dict, got {type(group_metrics)}")
    
    missing_groups = set(protected_groups) - set(group_metrics.keys())
    if missing_groups:
        raise ValidationError(f"Missing metrics for groups: {missing_groups}")
    
    for group, value in group_metrics.items():
        if not isinstance(value, (int, float)):
            raise ValidationError(
                f"Invalid metric value for group '{group}': {value} ({type(value)})"
            )
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(f"Invalid metric value for group '{group}': {value}")


def validate_confidence_interval(
    ci: tuple,
    metric_value: float,
) -> None:
    """
    Validate confidence interval.
    
    Args:
        ci: Tuple of (lower, upper) bounds
        metric_value: Point estimate
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(ci, (tuple, list)) or len(ci) != 2:
        raise ValidationError(f"Confidence interval must be (lower, upper), got {ci}")
    
    lower, upper = ci
    
    if not all(isinstance(x, (int, float)) for x in [lower, upper]):
        raise ValidationError(f"CI bounds must be numeric: {ci}")
    
    if np.isnan(lower) or np.isnan(upper) or np.isinf(lower) or np.isinf(upper):
        raise ValidationError(f"CI contains invalid values: {ci}")
    
    if lower > upper:
        raise ValidationError(f"CI lower bound > upper bound: {ci}")
    
    # Metric value should typically be within CI (sanity check)
    if not (lower <= metric_value <= upper):
        # This is a warning, not an error - bootstrap can be weird
        import warnings
        warnings.warn(
            f"Metric value {metric_value} outside CI {ci}. "
            f"This may indicate estimation issues."
        )


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator