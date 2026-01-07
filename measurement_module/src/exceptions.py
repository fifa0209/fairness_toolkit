"""
Custom Exceptions - Error handling for the Measurement Module.

Provides specific exception types for different failure modes to enable
better error handling and debugging.
"""


class FairnessModuleError(Exception):
    """Base exception for all fairness module errors."""
    pass


class ValidationError(FairnessModuleError):
    """Raised when input validation fails."""
    pass


class InsufficientDataError(FairnessModuleError):
    """
    Raised when group sizes are too small for reliable analysis.
    
    Example:
        >>> if group_size < min_group_size:
        ...     raise InsufficientDataError(
        ...         f"Group size {group_size} is below minimum {min_group_size}"
        ...     )
    """
    pass


class MetricComputationError(FairnessModuleError):
    """
    Raised when metric computation fails.
    
    This could be due to numerical issues, invalid configurations, or
    incompatible data.
    """
    pass


class UnsupportedMetricError(FairnessModuleError):
    """
    Raised when an unsupported metric is requested.
    
    Example:
        >>> if metric not in SUPPORTED_METRICS:
        ...     raise UnsupportedMetricError(
        ...         f"Metric '{metric}' is not supported. "
        ...         f"Choose from: {SUPPORTED_METRICS}"
        ...     )
    """
    pass


class ConfigurationError(FairnessModuleError):
    """
    Raised when configuration parameters are invalid or incompatible.
    
    Example:
        >>> if bootstrap_samples < 10:
        ...     raise ConfigurationError(
        ...         "bootstrap_samples must be at least 10"
        ...     )
    """
    pass


class StatisticalValidationError(FairnessModuleError):
    """
    Raised when statistical validation fails or produces invalid results.
    
    This includes bootstrap failures, CI computation issues, etc.
    """
    pass


class MLflowIntegrationError(FairnessModuleError):
    """
    Raised when MLflow logging or integration fails.
    
    Example:
        >>> try:
        ...     mlflow.log_metric("fairness", value)
        ... except Exception as e:
        ...     raise MLflowIntegrationError(f"Failed to log metric: {e}")
    """
    pass


class IntersectionalAnalysisError(FairnessModuleError):
    """
    Raised when intersectional analysis encounters issues.
    
    Common causes:
    - Too many intersectional groups with small sizes
    - Invalid group combinations
    - Missing data in certain subgroups
    """
    pass


# Convenience function for validation
def validate_binary_array(arr, name="array"):
    """
    Validate that an array contains only 0s and 1s.
    
    Args:
        arr: Array to validate
        name: Name of the array (for error messages)
        
    Raises:
        ValidationError: If array is not binary
    """
    import numpy as np
    
    arr = np.asarray(arr)
    unique_values = np.unique(arr)
    
    if not np.all(np.isin(unique_values, [0, 1])):
        raise ValidationError(
            f"{name} must contain only 0 and 1. "
            f"Found values: {unique_values}"
        )


def validate_group_sizes(sensitive_features, min_size):
    """
    Validate that all groups meet minimum size requirements.
    
    Args:
        sensitive_features: Protected attribute array
        min_size: Minimum required group size
        
    Raises:
        InsufficientDataError: If any group is too small
    """
    import numpy as np
    
    groups, counts = np.unique(sensitive_features, return_counts=True)
    
    for group, count in zip(groups, counts):
        if count < min_size:
            raise InsufficientDataError(
                f"Group {group} has only {count} samples, "
                f"but minimum required is {min_size}"
            )


def validate_equal_length(*arrays):
    """
    Validate that all arrays have the same length.
    
    Args:
        *arrays: Variable number of arrays to check
        
    Raises:
        ValidationError: If arrays have different lengths
    """
    import numpy as np
    
    arrays = [np.asarray(arr) for arr in arrays]
    lengths = [len(arr) for arr in arrays]
    
    if len(set(lengths)) > 1:
        raise ValidationError(
            f"All arrays must have the same length. "
            f"Got lengths: {lengths}"
        )