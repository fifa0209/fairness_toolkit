"""
Tests for MLflow integration and pytest assertions.

Tests mlops_integration.py module functionality.
Focuses on:
1. File I/O (Report generation with UTF-8).
2. Mocking external dependencies (MLflow).
3. Custom pytest assertions (assert_fairness).
4. Edge case handling.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from measurement_module.src import FairnessAnalyzer
from measurement_module.src.mlops_integration import (
    log_fairness_metrics_to_mlflow,
    log_fairness_report,
    assert_fairness,
    assert_all_fairness_metrics,
)


# Fixtures
@pytest.fixture
def sample_data():
    """
    Generate sample data for testing.
    
    Returns:
        tuple: (y_true, y_pred, sensitive)
    """
    np.random.seed(42)
    n = 200
    
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    sensitive = np.random.binomial(1, 0.5, n)
    
    return y_true, y_pred, sensitive


@pytest.fixture
def fairness_results(sample_data):
    """
    Generate fairness analysis results.
    
    Uses the FairnessAnalyzer to create a real result object
    that mimics the output of the pipeline.
    """
    y_true, y_pred, sensitive = sample_data
    
    analyzer = FairnessAnalyzer()
    results = analyzer.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
        threshold=0.1,
        compute_ci=False  # Faster for testing
    )
    
    return results


class TestMLflowLogging:
    """Test MLflow logging functionality."""
    
    def test_log_fairness_report_local(self, fairness_results):
        """
        Test generating fairness report without MLflow.
        
        Validates:
        1. Report file creation in temp directory.
        2. File encoding (UTF-8) to handle special chars.
        3. Content validation (contains expected headers).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Cast tmpdir to string for robust path joining on Windows
            report_path = os.path.join(str(tmpdir), "fairness_report.txt")
            
            # Should work without MLflow enabled
            log_fairness_report(fairness_results, report_path)
            
            # Check file was created successfully
            assert os.path.exists(report_path)
            
            # Check content with explicit UTF-8 encoding
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verify headers match the expected format from mlops_integration.py
            assert "FAIRNESS VALIDATION REPORT" in content
            assert "demographic_parity" in content.lower() or "Demographic Parity" in content
    
    @patch('measurement_module.src.mlops_integration.MLFLOW_AVAILABLE', True)
    @patch('measurement_module.src.mlops_integration.mlflow')
    def test_log_fairness_metrics_with_mlflow(self, mock_mlflow, fairness_results):
        """
        Test logging to MLflow when available.
        
        Strategy:
        1. Patch MLflow availability to True.
        2. Mock the mlflow module to avoid real network calls.
        3. Verify that log_metric/set_tag are called.
        """
        # Mock active run context
        mock_mlflow.active_run.return_value = MagicMock()
        
        # This function should call MLflow internally
        log_fairness_metrics_to_mlflow(fairness_results, prefix="test")
        
        # Verify MLflow methods were called with correct data
        assert mock_mlflow.log_metric.called
        assert mock_mlflow.set_tag.called
    
    @patch('measurement_module.src.mlops_integration.MLFLOW_AVAILABLE', False)
    def test_log_fairness_metrics_without_mlflow(self, fairness_results):
        """
        Test error handling when MLflow is not available.
        
        Validates that the function raises an informative exception
        if MLflow is not installed.
        """
        # Expect an exception to be raised
        with pytest.raises(Exception) as exc_info:
            log_fairness_metrics_to_mlflow(fairness_results)
        
        # Verify the error message mentions MLflow
        assert "MLflow" in str(exc_info.value)


class TestPytestAssertions:
    """Test custom pytest assertions for fairness."""
    
    def test_assert_fairness_pass(self, sample_data):
        """
        Test assertion passes when model is fair.
        
        Setup:
        1. Generate random predictions.
        2. Use lenient threshold (0.2) to ensure pass.
        """
        y_true, y_pred, sensitive = sample_data
        
        n = len(y_true)
        # Create fair predictions (random, not biased)
        y_pred_fair = np.random.binomial(1, 0.5, n)
        
        # Should not raise an assertion error
        assert_fairness(
            y_true=y_true,
            y_pred=y_pred_fair,
            sensitive_features=sensitive,
            metric='demographic_parity',
            threshold=0.2  # Lenient threshold
        )
    
    def test_assert_fairness_fail(self):
        """
        Test assertion fails when model is unfair.
        
        Setup:
        1. Create highly biased predictions based on group membership.
        2. Use strict threshold (0.1) to ensure fail.
        """
        # Create biased scenario
        n = 200
        sensitive = np.random.binomial(1, 0.5, n)
        y_true = np.random.binomial(1, 0.5, n)
        
        # Force predictions to be 1 if sensitive=1, else 0
        # This guarantees high bias
        y_pred = np.where(sensitive == 1, 1, 0)
        
        # Expect AssertionError
        with pytest.raises(AssertionError) as exc_info:
            assert_fairness(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
                metric='demographic_parity',
                threshold=0.1  # Strict threshold
            )
        
        # Check error message content
        assert "Fairness assertion failed" in str(exc_info.value)
    
    def test_assert_all_fairness_metrics(self, sample_data):
        """
        Test assertion for all metrics simultaneously.
        
        Validates that the wrapper iterates through all metrics
        and applies the threshold correctly.
        """
        y_true, y_pred, sensitive = sample_data
        
        # With very lenient threshold (0.5), this should pass
        # even with random noise
        assert_all_fairness_metrics(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            threshold=0.5  # Very lenient
        )
    
    def test_assert_fairness_with_custom_analyzer(self, sample_data):
        """
        Test assertion with a custom analyzer instance.
        
        Validates that the function accepts an external analyzer object
        instead of creating a default one.
        """
        y_true, y_pred, sensitive = sample_data
        
        # Create analyzer with specific settings
        analyzer = FairnessAnalyzer(bootstrap_samples=50)
        
        # Pass the analyzer to the assertion function
        assert_fairness(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive,
            metric='equal_opportunity',
            threshold=0.3,
            analyzer=analyzer
        )


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_results_dict(self):
        """
        Test handling of empty results dictionary.
        
        Ensures the report generator doesn't crash when
        no metrics are provided (empty dict).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(str(tmpdir), "empty_report.txt")
            
            # Should handle empty dict gracefully without error
            log_fairness_report({}, report_path)
            
            # File should still be created
            assert os.path.exists(report_path)
    
    def test_invalid_metric_name(self, sample_data):
        """
        Test assertion with invalid metric name.
        
        Verifies error handling for typos or unsupported metrics.
        """
        y_true, y_pred, sensitive = sample_data
        
        # Expect an exception because 'invalid_metric' is not implemented
        with pytest.raises(Exception):
            assert_fairness(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
                metric='invalid_metric',
                threshold=0.1
            )
    
    def test_mismatched_lengths(self):
        """
        Test assertion with mismatched array lengths.
        
        Validates that input validation catches shape mismatches.
        """
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0])  # Length 3 vs 4
        sensitive = np.array([0, 0, 1, 1])
        
        # Expect validation error
        with pytest.raises(Exception):
            assert_fairness(
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=sensitive,
                metric='demographic_parity',
                threshold=0.1
            )


def test_integration_workflow():
    """
    Test complete integration workflow.
    
    Simulates:
    1. Data generation.
    2. Metric computation (via Analyzer).
    3. Report generation (via mlops_integration).
    4. File content verification.
    """
    # Generate data
    np.random.seed(42)
    n = 300
    y_true = np.random.binomial(1, 0.5, n)
    y_pred = np.random.binomial(1, 0.5, n)
    sensitive = np.random.binomial(1, 0.5, n)
    
    # Compute metrics using the real Analyzer
    analyzer = FairnessAnalyzer()
    results = analyzer.compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
        threshold=0.15,
        compute_ci=False
    )
    
    # Generate report to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(str(tmpdir), "integration_report.txt")
        log_fairness_report(results, report_path)
        
        # Verify file creation
        assert os.path.exists(report_path)
        
        # Verify report content
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for generic header string to match report structure
        assert "VALIDATION REPORT" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])