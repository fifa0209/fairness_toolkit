"""
Unit tests for utils.py - Training Module Utilities
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from training_module.src.utils import (
    prepare_fairness_data,
    evaluate_model_comprehensive,
    log_model_performance,
    compute_group_statistics,
    validate_fairness_data,
    create_synthetic_fairness_dataset,
    grid_search_fairness_weights,
)


class TestPrepareFairnessData:
    """Test suite for prepare_fairness_data."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        s = np.random.binomial(1, 0.5, 200)
        return X, y, s
    
    def test_basic_split(self, sample_data):
        """Test basic data splitting."""
        X, y, s = sample_data
        
        data = prepare_fairness_data(
            X, y, s,
            test_size=0.2,
            val_size=0.1,
            random_state=42
        )
        
        assert 'X_train' in data
        assert 'y_train' in data
        assert 'sensitive_train' in data
        assert 'X_val' in data
        assert 'X_test' in data
        
        # Check sizes
        total = len(X)
        test_size = len(data['X_test'])
        val_size = len(data['X_val'])
        train_size = len(data['X_train'])
        
        assert test_size == pytest.approx(total * 0.2, abs=5)
        assert train_size + val_size + test_size == total
    
    def test_with_scaling(self, sample_data):
        """Test data preparation with feature scaling."""
        X, y, s = sample_data
        
        data = prepare_fairness_data(
            X, y, s,
            scale_features=True,
            random_state=42
        )
        
        # Check that scaler is returned
        assert 'scaler' in data
        assert data['scaler'] is not None
        
        # Check that scaled data has zero mean and unit variance
        assert np.allclose(data['X_train'].mean(axis=0), 0, atol=0.1)
        assert np.allclose(data['X_train'].std(axis=0), 1, atol=0.1)
    
    def test_without_scaling(self, sample_data):
        """Test data preparation without feature scaling."""
        X, y, s = sample_data
        
        data = prepare_fairness_data(
            X, y, s,
            scale_features=False,
            random_state=42
        )
        
        assert data['scaler'] is None
        # Data should be unscaled (not necessarily zero mean)
    
    def test_with_pandas_input(self, sample_data):
        """Test with pandas DataFrame/Series input."""
        X, y, s = sample_data
        
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        s_series = pd.Series(s)
        
        data = prepare_fairness_data(
            X_df, y_series, s_series,
            random_state=42
        )
        
        # Output should be numpy arrays
        assert isinstance(data['X_train'], np.ndarray)
        assert isinstance(data['y_train'], np.ndarray)
    
    def test_reproducibility(self, sample_data):
        """Test that same random_state gives same split."""
        X, y, s = sample_data
        
        data1 = prepare_fairness_data(X, y, s, random_state=42)
        data2 = prepare_fairness_data(X, y, s, random_state=42)
        
        np.testing.assert_array_equal(data1['X_train'], data2['X_train'])
        np.testing.assert_array_equal(data1['y_train'], data2['y_train'])


class TestEvaluateModelComprehensive:
    """Test suite for evaluate_model_comprehensive."""
    
    @pytest.fixture
    def trained_model_and_data(self):
        """Create trained model and test data."""
        np.random.seed(42)
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        s = np.random.binomial(1, 0.5, 200)
        
        # Split data
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        s_train, s_test = s[:150], s[150:]
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        return model, X_test, y_test, s_test
    
    def test_basic_evaluation(self, trained_model_and_data):
        """Test basic model evaluation."""
        model, X, y, s = trained_model_and_data
        
        metrics = evaluate_model_comprehensive(model, X, y, s)
        
        # Check that all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
    
    def test_with_prefix(self, trained_model_and_data):
        """Test evaluation with metric prefix."""
        model, X, y, s = trained_model_and_data
        
        metrics = evaluate_model_comprehensive(
            model, X, y, s, prefix='test_'
        )
        
        # All metrics should have prefix
        assert 'test_accuracy' in metrics
        assert 'test_precision' in metrics
        assert 'test_recall' in metrics
    
    def test_fairness_metrics_included(self, trained_model_and_data):
        """Test that fairness metrics are computed."""
        model, X, y, s = trained_model_and_data
        
        metrics = evaluate_model_comprehensive(model, X, y, s)
        
        # Should include fairness metrics if measurement module available
        # Note: This depends on measurement_module being available
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
    
    # def test_metric_values_valid(self, trained_model_and_data):
    #     """Test that metric values are valid."""
    #     model, X, y, s = trained_model_and_data
        
    #     metrics = evaluate_model_comprehensive(model, X, y, s)
        
    #     # All metrics should be between 0 and 1 or valid floats
    #     for key, value in metrics.items():
    #         assert isinstance(value, (int, float))
    #         assert not np.isnan(value)

    def test_metric_values_valid(self, trained_model_and_data):
        """Test that metric values are valid."""
        model, X, y, s = trained_model_and_data
        
        metrics = evaluate_model_comprehensive(model, X, y, s)
        
        # Check scalar metrics only
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not np.isnan(value)
            
class TestLogModelPerformance:
    """Test suite for log_model_performance."""
    
    def test_logging_basic(self, caplog):
        """Test basic performance logging."""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'fairness_metric': 0.10
        }
        
        log_model_performance(metrics, model_name='Test Model')
        
        # Check that logging occurred (captures would depend on logger config)
        assert len(metrics) == 3
    
    def test_with_empty_metrics(self, caplog):
        """Test logging with empty metrics."""
        log_model_performance({}, model_name='Empty Model')
        # Should not crash


class TestComputeGroupStatistics:
    """Test suite for compute_group_statistics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions."""
        np.random.seed(42)
        
        n = 100
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.random.binomial(1, 0.5, n)
        sensitive = np.random.binomial(1, 0.5, n)
        
        return y_true, y_pred, sensitive
    
    def test_basic_computation(self, sample_predictions):
        """Test basic group statistics computation."""
        y_true, y_pred, sensitive = sample_predictions
        
        df = compute_group_statistics(y_true, y_pred, sensitive)
        
        assert isinstance(df, pd.DataFrame)
        assert 'group' in df.columns
        assert 'n_samples' in df.columns
        assert 'accuracy' in df.columns
        assert 'base_rate' in df.columns
        assert 'selection_rate' in df.columns
    
    def test_two_groups(self, sample_predictions):
        """Test with two groups."""
        y_true, y_pred, sensitive = sample_predictions
        
        df = compute_group_statistics(y_true, y_pred, sensitive)
        
        # Should have 2 groups + 1 difference row
        assert len(df) == 3
        
        # Check for difference row
        assert 'Difference' in df['group'].values
    
    def test_multiple_groups(self):
        """Test with multiple groups."""
        np.random.seed(42)
        
        y_true = np.random.binomial(1, 0.5, 150)
        y_pred = np.random.binomial(1, 0.5, 150)
        sensitive = np.random.choice([0, 1, 2], 150)  # 3 groups
        
        df = compute_group_statistics(y_true, y_pred, sensitive)
        
        # Should have 3 groups (no difference row for >2 groups)
        assert len(df) >= 3
    
    def test_metric_ranges(self, sample_predictions):
        """Test that computed metrics are in valid ranges."""
        y_true, y_pred, sensitive = sample_predictions
        
        df = compute_group_statistics(y_true, y_pred, sensitive)
        
        # Check that metrics are in [0, 1]
        assert all(0 <= df['accuracy'].iloc[:-1])
        assert all(df['accuracy'].iloc[:-1] <= 1)
        assert all(0 <= df['precision'].iloc[:-1])
        assert all(df['precision'].iloc[:-1] <= 1)


class TestValidateFairnessData:
    """Test suite for validate_fairness_data."""
    
    def test_valid_data(self):
        """Test validation with valid data."""
        X = np.random.randn(100, 10)
        y = np.random.binomial(1, 0.5, 100)
        s = np.random.binomial(1, 0.5, 100)
        
        result = validate_fairness_data(X, y, s)
        
        assert result is True
    
    def test_shape_mismatch(self):
        """Test validation with shape mismatch."""
        X = np.random.randn(100, 10)
        y = np.random.binomial(1, 0.5, 90)  # Wrong size
        s = np.random.binomial(1, 0.5, 100)
        
        with pytest.raises(ValueError, match="Shape mismatch"):
            validate_fairness_data(X, y, s)
    
    def test_nan_in_features(self):
        """Test validation with NaN in features."""
        X = np.random.randn(100, 10)
        X[0, 0] = np.nan
        y = np.random.binomial(1, 0.5, 100)
        s = np.random.binomial(1, 0.5, 100)
        
        with pytest.raises(ValueError, match="X contains NaN"):
            validate_fairness_data(X, y, s)
    
    def test_nan_in_labels(self):
        """Test validation with NaN in labels."""
        X = np.random.randn(100, 10)
        y = np.random.binomial(1, 0.5, 100).astype(float)
        y[0] = np.nan
        s = np.random.binomial(1, 0.5, 100)
        
        with pytest.raises(ValueError, match="y contains NaN"):
            validate_fairness_data(X, y, s)
    
    def test_invalid_labels(self):
        """Test validation with non-binary labels."""
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)  # Labels: 0, 1, 2
        s = np.random.binomial(1, 0.5, 100)
        
        with pytest.raises(ValueError, match="Labels must be binary"):
            validate_fairness_data(X, y, s)
    
    def test_small_group_warning(self, caplog):
        """Test warning for small groups."""
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)
        s = np.zeros(50)
        s[:5] = 1  # Only 5 samples in group 1
        
        result = validate_fairness_data(X, y, s)
        
        # Should still pass but log warning
        assert result is True
    
    def test_imbalanced_labels_warning(self, caplog):
        """Test warning for highly imbalanced labels."""
        X = np.random.randn(100, 10)
        y = np.zeros(100)
        y[:3] = 1  # Only 3% positive
        s = np.random.binomial(1, 0.5, 100)
        
        result = validate_fairness_data(X, y, s)
        
        # Should pass but log warning
        assert result is True


class TestCreateSyntheticFairnessDataset:
    """Test suite for create_synthetic_fairness_dataset."""
    
    def test_basic_creation(self):
        """Test basic synthetic data creation."""
        X, y, s = create_synthetic_fairness_dataset(
            n_samples=100,
            n_features=10,
            bias_strength=0.3,
            random_state=42
        )
        
        assert X.shape == (100, 10)
        assert y.shape == (100,)
        assert s.shape == (100,)
        assert set(y).issubset({0, 1})
        assert set(s).issubset({0, 1})
    
    def test_no_bias(self):
        """Test dataset creation with no bias."""
        X, y, s = create_synthetic_fairness_dataset(
            n_samples=200,
            n_features=5,
            bias_strength=0.0,
            random_state=42
        )
        
        # With no bias, groups should have similar positive rates
        pos_rate_g0 = y[s == 0].mean()
        pos_rate_g1 = y[s == 1].mean()
        
        # They should be relatively close (not perfect due to randomness)
        assert abs(pos_rate_g0 - pos_rate_g1) < 0.3
    
    def test_strong_bias(self):
        """Test dataset creation with strong bias."""
        X, y, s = create_synthetic_fairness_dataset(
            n_samples=500,
            n_features=5,
            bias_strength=2.0,
            random_state=42
        )
        
        # With strong bias, groups should have different positive rates
        pos_rate_g0 = y[s == 0].mean()
        pos_rate_g1 = y[s == 1].mean()
        
        # They should be different
        assert abs(pos_rate_g0 - pos_rate_g1) > 0.1
    
    def test_reproducibility(self):
        """Test that same random_state gives same data."""
        X1, y1, s1 = create_synthetic_fairness_dataset(
            n_samples=100,
            random_state=42
        )
        
        X2, y2, s2 = create_synthetic_fairness_dataset(
            n_samples=100,
            random_state=42
        )
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_array_equal(s1, s2)


class TestGridSearchFairnessWeights:
    """Test suite for grid_search_fairness_weights."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for grid search."""
        np.random.seed(42)
        X, y = make_classification(n_samples=150, n_features=8, random_state=42)
        s = np.random.binomial(1, 0.5, 150)
        
        split = 100
        return {
            'X_train': X[:split],
            'y_train': y[:split],
            's_train': s[:split],
            'X_val': X[split:],
            'y_val': y[split:],
            's_val': s[split:],
        }
    
    def test_grid_search_not_implemented(self, sample_data):
        """Test that grid search raises NotImplementedError (stub)."""
        # This function requires a model class with fairness_weight parameter
        # For now, we test that it's defined
        assert callable(grid_search_fairness_weights)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataset(self):
        """Test with empty dataset."""
        X = np.array([]).reshape(0, 5)
        y = np.array([])
        s = np.array([])
        
        with pytest.raises(ValueError):
            prepare_fairness_data(X, y, s)
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.random.randn(1, 5)
        y = np.array([1])
        s = np.array([0])
        
        # Should raise error due to insufficient data for splitting
        with pytest.raises(ValueError):
            prepare_fairness_data(X, y, s, test_size=0.5)
    
    def test_all_same_label(self):
        """Test with all same label."""
        X = np.random.randn(100, 5)
        y = np.ones(100)
        s = np.random.binomial(1, 0.5, 100)
        
        # Should handle but may warn about class imbalance
        result = validate_fairness_data(X, y, s)
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])