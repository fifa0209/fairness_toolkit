"""
Unit tests for visualization.py - Fairness Visualization Tools
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from training_module.src.visualization import (
    plot_pareto_frontier,
    plot_fairness_comparison,
    plot_group_metrics,
    generate_pareto_frontier_data,
)


class TestPlotParetoFrontier:
    """Test suite for plot_pareto_frontier."""
    
    @pytest.fixture
    def sample_results(self):
        """Generate sample results for plotting."""
        return [
            {'accuracy': 0.90, 'fairness': 0.20, 'param': 0.0},
            {'accuracy': 0.87, 'fairness': 0.15, 'param': 0.3},
            {'accuracy': 0.84, 'fairness': 0.10, 'param': 0.5},
            {'accuracy': 0.80, 'fairness': 0.05, 'param': 0.8},
            {'accuracy': 0.75, 'fairness': 0.02, 'param': 1.0},
        ]
    
    def test_basic_plot(self, sample_results):
        """Test basic Pareto frontier plotting."""
        fig = plot_pareto_frontier(sample_results)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_custom_keys(self, sample_results):
        """Test plotting with custom key names."""
        fig = plot_pareto_frontier(
            sample_results,
            accuracy_key='accuracy',
            fairness_key='fairness',
            param_key='param'
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_save_path(self, sample_results, tmp_path):
        """Test saving plot to file."""
        save_path = tmp_path / "pareto_test.png"
        
        fig = plot_pareto_frontier(
            sample_results,
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)
    
    def test_single_point(self):
        """Test plotting with single point."""
        results = [{'accuracy': 0.85, 'fairness': 0.10, 'param': 0.5}]
        
        fig = plot_pareto_frontier(results)
        
        assert fig is not None
        plt.close(fig)
    
    def test_empty_results(self):
        """Test with empty results."""
        with pytest.raises((IndexError, ValueError)):
            plot_pareto_frontier([])


class TestPlotFairnessComparison:
    """Test suite for plot_fairness_comparison."""
    
    @pytest.fixture
    def sample_models(self):
        """Generate sample model metrics for comparison."""
        return {
            'Baseline': {
                'demographic_parity': 0.25,
                'equalized_odds': 0.18,
                'calibration_error': 0.12
            },
            'Fair Model': {
                'demographic_parity': 0.08,
                'equalized_odds': 0.10,
                'calibration_error': 0.08
            },
            'Very Fair Model': {
                'demographic_parity': 0.03,
                'equalized_odds': 0.05,
                'calibration_error': 0.15
            }
        }
    
    def test_basic_comparison(self, sample_models):
        """Test basic fairness comparison plot."""
        fig = plot_fairness_comparison(sample_models)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_specific_metrics(self, sample_models):
        """Test plotting specific metrics."""
        fig = plot_fairness_comparison(
            sample_models,
            metric_names=['demographic_parity', 'equalized_odds']
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_save_path(self, sample_models, tmp_path):
        """Test saving comparison plot."""
        save_path = tmp_path / "comparison_test.png"
        
        fig = plot_fairness_comparison(
            sample_models,
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)
    
    def test_single_model(self):
        """Test with single model."""
        models = {
            'Model': {'metric1': 0.1, 'metric2': 0.2}
        }
        
        fig = plot_fairness_comparison(models)
        
        assert fig is not None
        plt.close(fig)
    
    def test_empty_models(self):
        """Test with empty models dict."""
        with pytest.raises((StopIteration, KeyError, ValueError)):
            plot_fairness_comparison({})


class TestPlotGroupMetrics:
    """Test suite for plot_group_metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Generate sample predictions and ground truth."""
        np.random.seed(42)
        
        n = 200
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.random.binomial(1, 0.5, n)
        sensitive = np.random.binomial(1, 0.5, n)
        
        return y_true, y_pred, sensitive
    
    def test_basic_group_metrics(self, sample_predictions):
        """Test basic group metrics plotting."""
        y_true, y_pred, sensitive = sample_predictions
        
        fig = plot_group_metrics(y_true, y_pred, sensitive)
        
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_with_group_names(self, sample_predictions):
        """Test with custom group names."""
        y_true, y_pred, sensitive = sample_predictions
        
        group_names = {0: 'Group A', 1: 'Group B'}
        
        fig = plot_group_metrics(
            y_true, y_pred, sensitive,
            group_names=group_names
        )
        
        assert fig is not None
        plt.close(fig)
    
    def test_with_save_path(self, sample_predictions, tmp_path):
        """Test saving group metrics plot."""
        y_true, y_pred, sensitive = sample_predictions
        save_path = tmp_path / "group_metrics_test.png"
        
        fig = plot_group_metrics(
            y_true, y_pred, sensitive,
            save_path=str(save_path)
        )
        
        assert fig is not None
        assert save_path.exists()
        plt.close(fig)
    
    def test_multiple_groups(self):
        """Test with more than 2 groups."""
        np.random.seed(42)
        
        n = 300
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.random.binomial(1, 0.5, n)
        sensitive = np.random.choice([0, 1, 2], n)  # 3 groups
        
        fig = plot_group_metrics(y_true, y_pred, sensitive)
        
        assert fig is not None
        plt.close(fig)
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        n = 100
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = y_true.copy()  # Perfect predictions
        sensitive = np.random.binomial(1, 0.5, n)
        
        fig = plot_group_metrics(y_true, y_pred, sensitive)
        
        assert fig is not None
        plt.close(fig)


class TestGenerateParetoFrontierData:
    """Test suite for generate_pareto_frontier_data."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        s = np.random.binomial(1, 0.5, 200)
        
        split1, split2 = 100, 150
        
        return {
            'X_train': X[:split1],
            'y_train': y[:split1],
            's_train': s[:split1],
            'X_val': X[split1:split2],
            'y_val': y[split1:split2],
            's_val': s[split1:split2],
        }
    
    @pytest.mark.slow
    def test_basic_generation(self, sample_data):
        """Test basic Pareto frontier data generation."""
        from sklearn.linear_model import LogisticRegression
        
        results = generate_pareto_frontier_data(
            base_estimator=LogisticRegression(max_iter=1000),
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            sensitive_train=sample_data['s_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            sensitive_val=sample_data['s_val'],
            fairness_weights=[0.0, 0.5, 1.0]
        )
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all('accuracy' in r for r in results)
        assert all('fairness' in r for r in results)
    
    @pytest.mark.slow
    def test_custom_weights(self, sample_data):
        """Test with custom fairness weights."""
        from sklearn.linear_model import LogisticRegression
        
        custom_weights = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        results = generate_pareto_frontier_data(
            base_estimator=LogisticRegression(max_iter=1000),
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            sensitive_train=sample_data['s_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            sensitive_val=sample_data['s_val'],
            fairness_weights=custom_weights
        )
        
        # Should attempt to train len(custom_weights) models
        assert len(results) <= len(custom_weights)


class TestVisualizationEdgeCases:
    """Test edge cases and error handling."""
    
    def test_pareto_with_nan_values(self):
        """Test Pareto plot with NaN values."""
        results = [
            {'accuracy': 0.9, 'fairness': 0.1, 'param': 0.0},
            {'accuracy': np.nan, 'fairness': 0.2, 'param': 0.5},
            {'accuracy': 0.8, 'fairness': 0.15, 'param': 1.0},
        ]
        
        # Should handle or raise appropriate error
        try:
            fig = plot_pareto_frontier(results)
            plt.close(fig)
        except (ValueError, TypeError):
            pass  # Acceptable to raise error for invalid data
    
    def test_comparison_with_missing_metrics(self):
        """Test comparison plot with missing metrics."""
        models = {
            'Model1': {'metric1': 0.1, 'metric2': 0.2},
            'Model2': {'metric1': 0.15}  # Missing metric2
        }
        
        # Should handle gracefully
        fig = plot_fairness_comparison(models)
        assert fig is not None
        plt.close(fig)
    
    def test_group_metrics_single_group(self):
        """Test group metrics with only one group."""
        n = 50
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.random.binomial(1, 0.5, n)
        sensitive = np.zeros(n)  # All same group
        
        fig = plot_group_metrics(y_true, y_pred, sensitive)
        
        assert fig is not None
        plt.close(fig)
    
    def test_all_zero_predictions(self):
        """Test with all zero predictions."""
        n = 100
        y_true = np.random.binomial(1, 0.5, n)
        y_pred = np.zeros(n)  # All negative predictions
        sensitive = np.random.binomial(1, 0.5, n)
        
        fig = plot_group_metrics(y_true, y_pred, sensitive)
        
        assert fig is not None
        plt.close(fig)


class TestPlotConfiguration:
    """Test plot configuration and styling."""
    
    def test_figure_size(self):
        """Test that figures have reasonable sizes."""
        results = [
            {'accuracy': 0.9, 'fairness': 0.1, 'param': 0.0},
            {'accuracy': 0.85, 'fairness': 0.05, 'param': 0.5},
        ]
        
        fig = plot_pareto_frontier(results)
        
        # Check figure size
        assert fig.get_figwidth() > 0
        assert fig.get_figheight() > 0
        plt.close(fig)
    
    def test_axes_labels(self):
        """Test that axes have labels."""
        results = [
            {'accuracy': 0.9, 'fairness': 0.1, 'param': 0.0},
            {'accuracy': 0.85, 'fairness': 0.05, 'param': 0.5},
        ]
        
        fig = plot_pareto_frontier(results)
        ax = fig.axes[0]
        
        # Check that labels are set
        assert ax.get_xlabel() != ''
        assert ax.get_ylabel() != ''
        plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])