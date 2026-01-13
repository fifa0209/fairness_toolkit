"""
Tests for regression fairness metrics.

Tests the regression_metrics.py module functionality.
"""

import pytest
import numpy as np
from measurement_module.src.regression_metrics import (
    mean_absolute_error_difference,
    root_mean_squared_error_difference,
    r2_score_difference,
    mean_residual_difference,
    compute_regression_metric,
    compute_all_regression_metrics,
    interpret_regression_metric,
)


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n = 200
    
    # Generate continuous target
    y_true = np.random.normal(100, 20, n)
    
    # Generate predictions with some error
    y_pred = y_true + np.random.normal(0, 5, n)
    
    # Binary sensitive attribute
    sensitive = np.random.binomial(1, 0.5, n)
    
    return y_true, y_pred, sensitive


@pytest.fixture
def biased_regression_data():
    """Generate regression data with systematic bias."""
    np.random.seed(42)
    n = 200
    
    sensitive = np.random.binomial(1, 0.5, n)
    
    # True values
    y_true = np.random.normal(100, 20, n)
    
    # Predictions with bias: worse for group 1
    y_pred = np.where(
        sensitive == 0,
        y_true + np.random.normal(0, 3, n),  # Small error for group 0
        y_true + np.random.normal(0, 10, n)  # Large error for group 1
    )
    
    return y_true, y_pred, sensitive


class TestMAEDifference:
    """Test MAE difference metric."""
    
    def test_mae_difference_calculation(self, regression_data):
        """Test basic MAE difference calculation."""
        y_true, y_pred, sensitive = regression_data
        
        diff, group_maes, group_sizes = mean_absolute_error_difference(
            y_true, y_pred, sensitive
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
        assert len(group_maes) == 2
        assert len(group_sizes) == 2
        assert all(mae >= 0 for mae in group_maes.values())
    
    def test_mae_difference_with_bias(self, biased_regression_data):
        """Test MAE difference detects bias."""
        y_true, y_pred, sensitive = biased_regression_data
        
        diff, group_maes, _ = mean_absolute_error_difference(
            y_true, y_pred, sensitive
        )
        
        # Should detect significant difference
        assert diff > 1.0  # Expect noticeable difference
        
        # Group 1 should have higher MAE
        maes = list(group_maes.values())
        assert max(maes) > min(maes)
    
    def test_mae_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0])  # Perfect
        sensitive = np.array([0, 0, 1, 1])
        
        diff, group_maes, _ = mean_absolute_error_difference(
            y_true, y_pred, sensitive
        )
        
        assert diff == 0.0
        assert all(mae == 0.0 for mae in group_maes.values())


class TestRMSEDifference:
    """Test RMSE difference metric."""
    
    def test_rmse_difference_calculation(self, regression_data):
        """Test basic RMSE difference calculation."""
        y_true, y_pred, sensitive = regression_data
        
        diff, group_rmses, group_sizes = root_mean_squared_error_difference(
            y_true, y_pred, sensitive
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
        assert len(group_rmses) == 2
        assert all(rmse >= 0 for rmse in group_rmses.values())
    
    def test_rmse_higher_than_mae(self, regression_data):
        """Test that RMSE is typically higher than MAE."""
        y_true, y_pred, sensitive = regression_data
        
        _, group_maes, _ = mean_absolute_error_difference(
            y_true, y_pred, sensitive
        )
        _, group_rmses, _ = root_mean_squared_error_difference(
            y_true, y_pred, sensitive
        )
        
        # RMSE should be >= MAE (due to squared errors)
        for group in group_maes.keys():
            assert group_rmses[group] >= group_maes[group] - 0.001  # Allow small numerical errors


class TestR2Difference:
    """Test R² difference metric."""
    
    def test_r2_difference_calculation(self, regression_data):
        """Test basic R² difference calculation."""
        y_true, y_pred, sensitive = regression_data
        
        diff, group_r2s, _ = r2_score_difference(
            y_true, y_pred, sensitive
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
        assert len(group_r2s) == 2
    
    def test_r2_bounds(self, regression_data):
        """Test R² is in valid range."""
        y_true, y_pred, sensitive = regression_data
        
        _, group_r2s, _ = r2_score_difference(
            y_true, y_pred, sensitive
        )
        
        # R² can be negative for very bad models, but typically between 0 and 1
        for r2 in group_r2s.values():
            assert r2 <= 1.0  # Cannot exceed 1
    
    def test_r2_perfect_predictions(self):
        """
        Test R² with perfect predictions.
        R² should be 1.0 for both groups.
        """
        n = 20
        np.random.seed(42)
        y_true = np.random.rand(n) * 10
        y_pred = y_true.copy()  # Perfect prediction
        sensitive = np.array([0]*10 + [1]*10)
        
        _, group_r2s, _ = r2_score_difference(
            y_true, y_pred, sensitive
        )
        
        # All R² should be 1.0
        for r2 in group_r2s.values():
            assert r2 == 1.0, f"R² {r2} should be 1.0 for perfect prediction"

    def test_r2_biased_predictions(self):
        """
        Test R² detection of performance difference.
        Group 1 has worse R² than Group 0.
        """
        n = 20
        np.random.seed(42)
        
        # True values
        y_true = np.linspace(1.0, 4.0, n)
        
        # Biased predictions
        # Group 0 (indices 0-9): Good predictions (low error)
        y_pred_0 = y_true[0:10] + np.random.normal(0, 0.1, 10)
        
        # Group 1 (indices 10-19): Bad predictions (high error)
        y_pred_1 = y_true[10:20] + np.random.normal(0, 2.0, 10)
        
        y_pred = np.concatenate([y_pred_0, y_pred_1])
        sensitive = np.array([0]*10 + [1]*10)
        
        _, group_r2s, _ = r2_score_difference(
            y_true, y_pred, sensitive
        )
        
        # Group 1 R² should be worse than Group 0
        assert group_r2s['Group_0'] > group_r2s['Group_1'], "Group 0 should be more fair (lower R²)"
class TestResidualBias:
    """Test residual bias metric."""
    
    def test_residual_bias_calculation(self, regression_data):
        """Test basic residual bias calculation."""
        y_true, y_pred, sensitive = regression_data
        
        diff, group_residuals, _ = mean_residual_difference(
            y_true, y_pred, sensitive
        )
        
        assert isinstance(diff, float)
        assert diff >= 0
        assert len(group_residuals) == 2
    
    def test_residual_bias_systematic(self):
        """Test detection of systematic bias."""
        n = 100
        sensitive = np.random.binomial(1, 0.5, n)
        
        y_true = np.random.normal(100, 20, n)
        
        # Systematic over-prediction for group 1, under-prediction for group 0
        y_pred = np.where(
            sensitive == 0,
            y_true - 10,  # Under-predict group 0
            y_true + 10   # Over-predict group 1
        )
        
        diff, group_residuals, _ = mean_residual_difference(
            y_true, y_pred, sensitive
        )
        
        # Should detect large systematic difference
        assert diff > 15.0
    
    def test_residual_bias_unbiased(self):
        """Test with unbiased predictions."""
        n = 100
        y_true = np.random.normal(100, 20, n)
        y_pred = y_true + np.random.normal(0, 5, n)  # Unbiased noise
        sensitive = np.random.binomial(1, 0.5, n)
        
        diff, group_residuals, _ = mean_residual_difference(
            y_true, y_pred, sensitive
        )
        
        # Mean residuals should be close to 0 for both groups
        for residual in group_residuals.values():
            assert abs(residual) < 5.0  # Small random variation


class TestComputeRegressionMetric:
    """Test unified metric computation function."""
    
    def test_compute_all_metrics(self, regression_data):
        """Test computing all regression metrics."""
        y_true, y_pred, sensitive = regression_data
        
        metrics = ["mae_difference", "rmse_difference", "r2_difference", "residual_bias"]
        
        for metric in metrics:
            diff, group_metrics, group_sizes = compute_regression_metric(
                metric, y_true, y_pred, sensitive
            )
            
            assert isinstance(diff, float)
            assert len(group_metrics) == 2
            assert len(group_sizes) == 2
    
    def test_invalid_metric_name(self, regression_data):
        """Test error handling for invalid metric."""
        y_true, y_pred, sensitive = regression_data
        
        with pytest.raises(ValueError) as exc_info:
            compute_regression_metric(
                "invalid_metric", y_true, y_pred, sensitive
            )
        
        assert "Unknown regression metric" in str(exc_info.value)
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched lengths."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1])  # Wrong length
        sensitive = np.array([0, 0, 1])
        
        with pytest.raises(ValueError) as exc_info:
            compute_regression_metric(
                "mae_difference", y_true, y_pred, sensitive
            )
        
        assert "Length mismatch" in str(exc_info.value)


class TestAllRegressionMetrics:
    """Test comprehensive metric computation."""
    
    def test_compute_all_regression_metrics(self, regression_data):
        """Test computing all metrics at once."""
        y_true, y_pred, sensitive = regression_data
        
        results = compute_all_regression_metrics(
            y_true, y_pred, sensitive, threshold=1.0
        )
        
        assert isinstance(results, dict)
        assert len(results) == 4  # All 4 metrics
        
        for metric, result in results.items():
            assert 'value' in result
            assert 'is_fair' in result
            assert 'interpretation' in result
            assert 'group_metrics' in result
    
    def test_fairness_assessment(self, biased_regression_data):
        """Test fairness assessment with biased data."""
        y_true, y_pred, sensitive = biased_regression_data
        
        results = compute_all_regression_metrics(
            y_true, y_pred, sensitive, threshold=1.0
        )
        
        # At least some metrics should fail
        unfair_count = sum(1 for r in results.values() if not r['is_fair'])
        assert unfair_count > 0


class TestInterpretation:
    """Test metric interpretation."""
    
    def test_interpret_mae_difference(self):
        """Test MAE interpretation."""
        interpretation = interpret_regression_metric(
            "mae_difference",
            0.5,
            1.0,
            {"Group_0": 2.5, "Group_1": 3.0}
        )
        
        assert isinstance(interpretation, str)
        assert "MAE" in interpretation
        assert "FAIR" in interpretation
    
    def test_interpret_unfair_metric(self):
        """Test interpretation of unfair metric."""
        interpretation = interpret_regression_metric(
            "rmse_difference",
            2.0,
            1.0,
            {"Group_0": 3.0, "Group_1": 5.0}
        )
        
        assert "UNFAIR" in interpretation
        assert "different" in interpretation.lower()


def test_integration_regression_workflow():
    """Test complete regression fairness workflow."""
    # Generate data
    np.random.seed(42)
    n = 300
    
    y_true = np.random.normal(100, 20, n)
    y_pred = y_true + np.random.normal(0, 5, n)
    sensitive = np.random.binomial(1, 0.5, n)
    
    # Compute all metrics
    results = compute_all_regression_metrics(
        y_true, y_pred, sensitive, threshold=2.0
    )
    
    # Verify structure
    assert len(results) > 0
    
    # All should pass with lenient threshold
    for result in results.values():
        assert result['is_fair'] == True  # With threshold=2.0 and small error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])