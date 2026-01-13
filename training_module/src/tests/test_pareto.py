"""
Unit tests for pareto.py - Pareto Frontier Analysis
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from training_module.src.pareto import (
    ParetoPoint,
    ParetoFrontierExplorer,
    quick_pareto_analysis,
)


class TestParetoPoint:
    """Test suite for ParetoPoint dataclass."""
    
    def test_initialization(self):
        """Test ParetoPoint initialization."""
        point = ParetoPoint(
            accuracy=0.85,
            fairness_violation=0.12,
            hyperparameter=0.5,
            hyperparameter_name='fairness_weight'
        )
        
        assert point.accuracy == 0.85
        assert point.fairness_violation == 0.12
        assert point.hyperparameter == 0.5
        assert point.hyperparameter_name == 'fairness_weight'
        assert point.model is None
        assert point.metadata is None
    
    def test_with_optional_fields(self):
        """Test ParetoPoint with optional fields."""
        model = LogisticRegression()
        metadata = {'constraint': 'demographic_parity'}
        
        point = ParetoPoint(
            accuracy=0.9,
            fairness_violation=0.05,
            hyperparameter=0.1,
            hyperparameter_name='eps',
            model=model,
            metadata=metadata
        )
        
        assert point.model is model
        assert point.metadata == metadata


class TestParetoFrontierExplorer:
    """Test suite for ParetoFrontierExplorer."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            random_state=42
        )
        
        sensitive = np.random.binomial(1, 0.5, size=300)
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.5, random_state=42
        )
        
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_train, y_train, s_train, test_size=0.3, random_state=42
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            's_train': s_train,
            'X_val': X_val,
            'y_val': y_val,
            's_val': s_val,
            'X_test': X_test,
            'y_test': y_test,
            's_test': s_test,
        }
    
    def test_initialization(self):
        """Test explorer initialization."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions',
            constraint_type='demographic_parity'
        )
        
        assert explorer.base_estimator is not None
        assert explorer.training_method == 'reductions'
        assert explorer.constraint_type == 'demographic_parity'
        assert len(explorer.results_) == 0
        assert len(explorer.pareto_optimal_) == 0
    
    @pytest.mark.slow
    def test_explore_basic(self, sample_data):
        """Test basic exploration."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions',
            constraint_type='demographic_parity'
        )
        
        results = explorer.explore(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            param_range=np.array([0.0, 0.5, 1.0])
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ParetoPoint) for r in results)
        assert len(explorer.results_) == len(results)
    
    @pytest.mark.slow
    def test_explore_multiple_points(self, sample_data):
        """Test exploration with multiple parameter values."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions',
            constraint_type='demographic_parity'
        )
        
        n_points = 5
        results = explorer.explore(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            param_range=np.linspace(0.0, 1.0, n_points)
        )
        
        # Should have tried to train n_points models
        assert len(results) <= n_points
        assert len(results) > 0
    
    @pytest.mark.slow
    def test_pareto_optimal_identification(self, sample_data):
        """Test identification of Pareto-optimal points."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions',
            constraint_type='demographic_parity'
        )
        
        results = explorer.explore(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            param_range=np.linspace(0.0, 1.0, 5)
        )
        
        pareto_optimal = explorer.get_pareto_optimal()
        
        assert len(pareto_optimal) > 0
        assert len(pareto_optimal) <= len(results)
        assert all(isinstance(p, ParetoPoint) for p in pareto_optimal)
    
    def test_get_results_dataframe(self):
        """Test conversion of results to DataFrame."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # Manually add some results
        explorer.results_ = [
            ParetoPoint(0.9, 0.1, 0.5, 'weight'),
            ParetoPoint(0.85, 0.05, 0.7, 'weight'),
        ]
        explorer.pareto_optimal_ = [explorer.results_[1]]
        
        df = explorer.get_results_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert 'accuracy' in df.columns
        assert 'fairness_violation' in df.columns
        assert 'is_pareto_optimal' in df.columns
    
    def test_get_results_dataframe_empty(self):
        """Test DataFrame conversion with no results."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        df = explorer.get_results_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
    
    @pytest.mark.slow
    def test_recommend_model(self, sample_data):
        """Test model recommendation."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions',
            constraint_type='demographic_parity'
        )
        
        explorer.explore(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            param_range=np.linspace(0.0, 1.0, 5)
        )
        
        recommended = explorer.recommend_model(
            max_fairness_violation=0.5,
            min_accuracy=0.0
        )
        
        # Should return a model if any meet constraints
        if recommended:
            assert isinstance(recommended, ParetoPoint)
            assert recommended.fairness_violation <= 0.5
            assert recommended.accuracy >= 0.0
    
    def test_recommend_model_no_results(self):
        """Test recommendation with no results."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        recommended = explorer.recommend_model()
        
        assert recommended is None
    
    def test_recommend_model_strict_constraints(self):
        """Test recommendation with impossible constraints."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # Add results that don't meet strict constraints
        explorer.results_ = [
            ParetoPoint(0.8, 0.2, 0.5, 'weight'),
            ParetoPoint(0.75, 0.15, 0.7, 'weight'),
        ]
        explorer.pareto_optimal_ = explorer.results_
        
        recommended = explorer.recommend_model(
            max_fairness_violation=0.01,  # Very strict
            min_accuracy=0.95  # Very high
        )
        
        assert recommended is None


class TestParetoOptimalityLogic:
    """Test Pareto optimality detection logic."""
    
    def test_simple_pareto_frontier(self):
        """Test with clear Pareto frontier."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # Point A: high accuracy, high fairness violation
        # Point B: medium accuracy, medium fairness violation
        # Point C: low accuracy, low fairness violation
        # All three are Pareto-optimal
        explorer.results_ = [
            ParetoPoint(0.90, 0.20, 0.0, 'weight'),  # A
            ParetoPoint(0.85, 0.10, 0.5, 'weight'),  # B
            ParetoPoint(0.80, 0.05, 1.0, 'weight'),  # C
        ]
        
        explorer._identify_pareto_optimal()
        
        # All three should be Pareto-optimal
        assert len(explorer.pareto_optimal_) == 3
    
    def test_dominated_points(self):
        """Test with dominated points."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # Point A: 0.9 acc, 0.1 fair - Pareto optimal
        # Point B: 0.85 acc, 0.15 fair - Dominated by A (worse in both)
        # Point C: 0.95 acc, 0.05 fair - Pareto optimal (best in both, dominates A and B)
        explorer.results_ = [
            ParetoPoint(0.90, 0.10, 0.0, 'weight'),  # A - dominated by C
            ParetoPoint(0.85, 0.15, 0.5, 'weight'),  # B - dominated by A and C
            ParetoPoint(0.95, 0.05, 1.0, 'weight'),  # C - optimal (dominates both)
        ]
        
        explorer._identify_pareto_optimal()
        
        # Only C should be Pareto-optimal (it dominates both A and B)
        # C has higher accuracy AND lower fairness violation than both
        assert len(explorer.pareto_optimal_) == 1
        assert explorer.pareto_optimal_[0].accuracy == 0.95
        assert explorer.pareto_optimal_[0].fairness_violation == 0.05
    
    def test_all_same_performance(self):
        """Test with all points having same performance."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # All points identical
        explorer.results_ = [
            ParetoPoint(0.85, 0.10, 0.0, 'weight'),
            ParetoPoint(0.85, 0.10, 0.5, 'weight'),
            ParetoPoint(0.85, 0.10, 1.0, 'weight'),
        ]
        
        explorer._identify_pareto_optimal()
        
        # All should be considered Pareto-optimal (none dominates another)
        assert len(explorer.pareto_optimal_) == 3


class TestQuickParetoAnalysis:
    """Test suite for quick_pareto_analysis convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        
        X, y = make_classification(n_samples=200, n_features=8, random_state=42)
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
    def test_quick_analysis(self, sample_data):
        """Test quick Pareto analysis."""
        results, fig = quick_pareto_analysis(
            LogisticRegression(max_iter=1000),
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            n_points=3
        )
        
        assert len(results) > 0
        assert all(isinstance(r, ParetoPoint) for r in results)
        assert fig is not None
    
    @pytest.mark.slow
    def test_quick_analysis_custom_points(self, sample_data):
        """Test quick analysis with custom number of points."""
        results, fig = quick_pareto_analysis(
            LogisticRegression(max_iter=1000),
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            sample_data['X_val'],
            sample_data['y_val'],
            sample_data['s_val'],
            n_points=5
        )
        
        assert len(results) <= 5
        assert len(results) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_param_range(self):
        """Test with empty parameter range."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        X = np.random.randn(50, 5)
        y = np.random.binomial(1, 0.5, 50)
        s = np.random.binomial(1, 0.5, 50)
        
        results = explorer.explore(
            X, y, s, X, y, s,
            param_range=np.array([])
        )
        
        assert len(results) == 0
    
    @pytest.mark.slow
    def test_single_parameter_value(self):
        """Test with single parameter value."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions'
        )
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        s = np.random.binomial(1, 0.5, 100)
        
        results = explorer.explore(
            X, y, s, X, y, s,
            param_range=np.array([0.5])
        )
        
        assert len(results) == 1
        assert len(explorer.pareto_optimal_) == 1
    
    def test_invalid_training_method(self):
        """Test with invalid training method.
        
        Note: The explorer catches exceptions during exploration and logs them,
        so it doesn't raise an error but returns empty results instead.
        """
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='invalid_method'
        )
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.binomial(1, 0.5, 50)
        s = np.random.binomial(1, 0.5, 50)
        
        # The explore method catches exceptions and returns empty results
        # instead of raising them (graceful error handling)
        results = explorer.explore(X, y, s, X, y, s, param_range=np.array([0.5]))
        
        # Should return empty results since training failed
        assert len(results) == 0
        assert len(explorer.results_) == 0
        assert len(explorer.pareto_optimal_) == 0
    
    def test_direct_train_model_invalid_method(self):
        """Test that _train_model raises ValueError for invalid method."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='invalid_method'
        )
        
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.binomial(1, 0.5, 50)
        s = np.random.binomial(1, 0.5, 50)
        
        # Directly calling _train_model should raise ValueError
        with pytest.raises(ValueError, match="Unknown training method"):
            explorer._train_model(X, y, s, 0.5)
    
    @pytest.mark.slow
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(max_iter=1000),
            training_method='reductions'
        )
        
        X = np.random.randn(30, 5)
        y = np.random.binomial(1, 0.5, 30)
        s = np.random.binomial(1, 0.5, 30)
        
        results = explorer.explore(
            X, y, s, X, y, s,
            param_range=np.array([0.5])
        )
        
        # Should handle small datasets
        assert len(results) <= 1


class TestVisualization:
    """Test visualization methods."""
    
    def test_plot_frontier_with_results(self):
        """Test plotting with results."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        explorer.results_ = [
            ParetoPoint(0.9, 0.1, 0.0, 'weight'),
            ParetoPoint(0.85, 0.05, 0.5, 'weight'),
            ParetoPoint(0.8, 0.02, 1.0, 'weight'),
        ]
        
        fig = explorer.plot_frontier()
        
        assert fig is not None
    
    def test_plot_frontier_empty(self):
        """Test plotting with no results."""
        explorer = ParetoFrontierExplorer(
            base_estimator=LogisticRegression(),
            training_method='reductions'
        )
        
        # Should handle empty results gracefully
        # Implementation may raise error or return empty plot
        try:
            fig = explorer.plot_frontier()
            assert fig is not None
        except (ValueError, IndexError):
            # Acceptable to raise error for empty results
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'not slow'])