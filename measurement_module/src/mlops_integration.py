"""
MLOps Integration - MLflow logging and CI/CD testing utilities.

Provides helpers for:
- Logging fairness metrics to MLflow
- Custom pytest assertions for fairness testing
- Integration with experiment tracking
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import warnings

# Try to import mlflow, but make it optional
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn(
        "MLflow not available. Install with: pip install mlflow",
        ImportWarning
    )


def log_fairness_metrics_to_mlflow(
    metric_results: Dict[str, Any],
    prefix: str = "fairness",
    log_confidence_intervals: bool = True,
    log_group_metrics: bool = True,
) -> None:
    """
    Log fairness metric results to an active MLflow run.
    
    Args:
        metric_results: Dictionary of metric name -> MetricResult objects
        prefix: Prefix for metric names in MLflow (default: "fairness")
        log_confidence_intervals: Whether to log CI bounds
        log_group_metrics: Whether to log per-group metrics
        
    Raises:
        MLflowIntegrationError: If MLflow is not available or logging fails
        
    Example:
        >>> import mlflow
        >>> from measurement_module import FairnessAnalyzer
        >>> 
        >>> analyzer = FairnessAnalyzer()
        >>> results = analyzer.compute_all_metrics(y_true, y_pred, sensitive)
        >>> 
        >>> with mlflow.start_run():
        ...     mlflow.log_param("model_type", "LogisticRegression")
        ...     mlflow.log_metric("accuracy", 0.85)
        ...     log_fairness_metrics_to_mlflow(results)
    """
    if not MLFLOW_AVAILABLE:
        from .exceptions import MLflowIntegrationError
        raise MLflowIntegrationError(
            "MLflow is not installed. Install with: pip install mlflow"
        )
    
    try:
        # Check if there's an active run
        active_run = mlflow.active_run()
        if active_run is None:
            raise ValueError("No active MLflow run. Start a run with mlflow.start_run()")
        
        for metric_name, result in metric_results.items():
            # Log main metric value
            mlflow.log_metric(f"{prefix}_{metric_name}", result.value)
            
            # Log fairness status as a tag
            mlflow.set_tag(f"{prefix}_{metric_name}_is_fair", str(result.is_fair))
            
            # Log threshold
            mlflow.log_param(f"{prefix}_{metric_name}_threshold", result.threshold)
            
            # Log confidence intervals if available and requested
            if log_confidence_intervals and result.confidence_interval is not None:
                ci_lower, ci_upper = result.confidence_interval
                mlflow.log_metric(f"{prefix}_{metric_name}_ci_lower", ci_lower)
                mlflow.log_metric(f"{prefix}_{metric_name}_ci_upper", ci_upper)
                mlflow.log_metric(f"{prefix}_{metric_name}_ci_width", ci_upper - ci_lower)
            
            # Log effect size if available
            if hasattr(result, 'effect_size') and result.effect_size is not None:
                mlflow.log_metric(f"{prefix}_{metric_name}_effect_size", result.effect_size)
            
            # Log p-value if available
            if hasattr(result, 'p_value') and result.p_value is not None:
                mlflow.log_metric(f"{prefix}_{metric_name}_p_value", result.p_value)
            
            # Log group metrics if requested
            if log_group_metrics and result.group_metrics:
                for group_name, group_value in result.group_metrics.items():
                    safe_group_name = group_name.replace(" ", "_").replace(":", "_")
                    mlflow.log_metric(
                        f"{prefix}_{metric_name}_{safe_group_name}",
                        group_value
                    )
            
            # Log group sizes
            if hasattr(result, 'group_sizes') and result.group_sizes:
                for group_name, size in result.group_sizes.items():
                    safe_group_name = group_name.replace(" ", "_").replace(":", "_")
                    mlflow.log_metric(
                        f"{prefix}_{metric_name}_{safe_group_name}_size",
                        size
                    )
        
        print(f"✅ Successfully logged {len(metric_results)} fairness metrics to MLflow")
        
    except Exception as e:
        from .exceptions import MLflowIntegrationError
        raise MLflowIntegrationError(f"Failed to log fairness metrics to MLflow: {e}")


def log_fairness_report(
    metric_results: Dict[str, Any],
    report_path: str = "fairness_report.txt"
) -> None:
    """
    Generate and log a fairness audit report as an MLflow artifact.
    
    Args:
        metric_results: Dictionary of metric name -> MetricResult objects
        report_path: Path where report will be saved
        
    Example:
        >>> with mlflow.start_run():
        ...     results = analyzer.compute_all_metrics(y_true, y_pred, sensitive)
        ...     log_fairness_report(results)
    """
    if not MLFLOW_AVAILABLE:
        warnings.warn("MLflow not available. Saving report locally only.")
    
    # Generate report content
    report_lines = [
        "=" * 80,
        # FIX: Changed "FAIRNESS AUDIT REPORT" to "FAIRNESS VALIDATION REPORT"
        # to match test expectations and project consistency.
        "FAIRNESS VALIDATION REPORT",
        "=" * 80,
        ""
    ]
    
    for metric_name, result in metric_results.items():
        status = "✅ FAIR" if result.is_fair else "❌ UNFAIR"
        
        report_lines.extend([
            f"Metric: {metric_name.replace('_', ' ').title()}",
            f"Value: {result.value:.4f}",
            f"Threshold: {result.threshold}",
            f"Status: {status}",
        ])
        
        if result.confidence_interval:
            ci_lower, ci_upper = result.confidence_interval
            report_lines.append(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        if hasattr(result, 'interpretation'):
            report_lines.append(f"Interpretation: {result.interpretation}")
        
        report_lines.append("")
    
    # Overall assessment
    all_fair = all(r.is_fair for r in metric_results.values())
    report_lines.extend([
        "=" * 80,
        f"OVERALL: {'✅ ALL METRICS PASSED' if all_fair else '❌ FAIRNESS ISSUES DETECTED'}",
        "=" * 80
    ])
    
    # Write report
    report_content = "\n".join(report_lines)
    
    # Write with UTF-8 encoding to handle special chars like emojis
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # Log to MLflow if available
    if MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_artifact(report_path)
        print(f"✅ Fairness report logged to MLflow: {report_path}")
    else:
        print(f"📄 Fairness report saved locally: {report_path}")


# ============================================================================
# Pytest Assertions for CI/CD
# ============================================================================

def assert_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    metric: str = "demographic_parity",
    threshold: float = 0.1,
    require_statistical_significance: bool = False,
    analyzer: Optional[Any] = None
) -> None:
    """
    Custom pytest assertion for fairness testing.
    
    This function can be used in pytest test suites to automatically
    fail tests if fairness criteria are not met.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        metric: Fairness metric to check
        threshold: Maximum acceptable difference
        require_statistical_significance: If True, also check if CI excludes zero
        analyzer: Optional FairnessAnalyzer instance (creates new one if None)
        
    Raises:
        AssertionError: If fairness criteria are not met
        
    Example:
        >>> # In test_model_fairness.py
        >>> def test_loan_model_fairness():
        ...     y_test, y_pred, gender = load_test_data()
        ...     
        ...     # This will fail the test if fairness threshold is exceeded
        ...     assert_fairness(
        ...         y_true=y_test,
        ...         y_pred=y_pred,
        ...         sensitive_features=gender,
        ...         metric='demographic_parity',
        ...         threshold=0.1
        ...     )
    """
    # Import here to avoid circular imports
    if analyzer is None:
        from .fairness_analyzer_simple import FairnessAnalyzer
        analyzer = FairnessAnalyzer(bootstrap_samples=100)  # Faster for CI
    
    # Compute metric
    result = analyzer.compute_metric(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        metric=metric,
        threshold=threshold,
        compute_ci=require_statistical_significance
    )
    
    # Check fairness
    if not result.is_fair:
        error_msg = (
            f"Fairness assertion failed for {metric}!\n"
            f"  Measured value: {result.value:.4f}\n"
            f"  Threshold: {threshold}\n"
            f"  Difference: {result.value - threshold:.4f} over threshold\n"
        )
        
        if result.confidence_interval:
            ci_lower, ci_upper = result.confidence_interval
            error_msg += f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
        
        if result.group_metrics:
            error_msg += "  Group metrics:\n"
            for group, value in result.group_metrics.items():
                error_msg += f"    {group}: {value:.4f}\n"
        
        raise AssertionError(error_msg)
    
    # Check statistical significance if required
    if require_statistical_significance and result.confidence_interval:
        ci_lower, ci_upper = result.confidence_interval
        if ci_lower <= 0 <= ci_upper:
            raise AssertionError(
                f"Fairness difference is not statistically significant!\n"
                f"  95% CI [{ci_lower:.4f}, {ci_upper:.4f}] includes zero\n"
                f"  May need more data or effect is too small to detect reliably"
            )
    
    print(f"✅ Fairness assertion passed: {metric} = {result.value:.4f} <= {threshold}")


def assert_all_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    threshold: float = 0.1,
    metrics: Optional[list] = None,
    analyzer: Optional[Any] = None
) -> None:
    """
    Assert that all fairness metrics pass threshold.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        threshold: Maximum acceptable difference for all metrics
        metrics: List of metric names (None = all standard metrics)
        analyzer: Optional FairnessAnalyzer instance
        
    Raises:
        AssertionError: If any fairness metric fails
        
    Example:
        >>> def test_model_comprehensive_fairness():
        ...     y_test, y_pred, gender = load_test_data()
        ...     assert_all_fairness_metrics(y_test, y_pred, gender, threshold=0.1)
    """
    if metrics is None:
        metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
    
    if analyzer is None:
        from .fairness_analyzer_simple import FairnessAnalyzer
        analyzer = FairnessAnalyzer(bootstrap_samples=100)
    
    failed_metrics = []
    
    for metric in metrics:
        try:
            assert_fairness(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive_features,
                metric=metric,
                threshold=threshold,
                analyzer=analyzer
            )
        except AssertionError as e:
            failed_metrics.append((metric, str(e)))
    
    if failed_metrics:
        error_msg = f"Failed {len(failed_metrics)} of {len(metrics)} fairness checks:\n\n"
        for metric, msg in failed_metrics:
            error_msg += f"❌ {metric}:\n{msg}\n"
        raise AssertionError(error_msg)
    
    print(f"✅ All {len(metrics)} fairness metrics passed!")


def create_fairness_test_fixture(
    test_data_loader: callable,
    threshold: float = 0.1
):
    """
    Create a pytest fixture for fairness testing.
    
    Args:
        test_data_loader: Function that returns (y_true, y_pred, sensitive_features)
        threshold: Fairness threshold
        
    Returns:
        Pytest fixture function
        
    Example:
        >>> # In conftest.py
        >>> def load_test_data():
        ...     # Load your test data
        ...     return y_test, y_pred, gender
        >>> 
        >>> fairness_fixture = create_fairness_test_fixture(load_test_data)
        >>> 
        >>> # In test file
        >>> def test_my_model(fairness_fixture):
        ...     # Test automatically uses the fixture
        ...     pass
    """
    import pytest
    
    @pytest.fixture
    def fairness_checker():
        y_true, y_pred, sensitive_features = test_data_loader()
        
        from .fairness_analyzer_simple import FairnessAnalyzer
        analyzer = FairnessAnalyzer()
        
        results = analyzer.compute_all_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            threshold=threshold,
            compute_ci=False  # Fast for CI
        )
        
        return results
    
    return fairness_checker