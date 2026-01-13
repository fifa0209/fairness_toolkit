"""
Unit tests for sklearn_wrappers.py - Fairlearn Reductions Wrappers
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Check if fairlearn is available
try:
    from fairlearn.reductions import DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from training_module.src.sklearn_wrappers import (
    ReductionsWrapper,
    GridSearchReductions,
)


@pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
class TestReductionsWrapper:
    """Test suite for ReductionsWrapper."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        
        X, y = make_classification(
            n_samples=400,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        sensitive = np.random.binomial(1, 0.5, size=400)
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=42
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            's_train': s_train,
            'X_test': X_test,
            'y_test': y_test,
            's_test': s_test,
        }
    
    def test_initialization_with_string_constraint(self):
        """Test initialization with string constraint."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint='demographic_parity',
            eps=0.05
        )
        
        assert model.base_estimator is not None
        assert model.constraint == 'demographic_parity'
        assert model.eps == 0.05
        assert not model.fitted_
    
    def test_initialization_with_constraint_object(self):
        """Test initialization with constraint object."""
        constraint = DemographicParity()
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint=constraint,
            eps=0.05
        )
        
        assert model.constraint is constraint
    
    def test_initialization_invalid_constraint_string(self):
        """Test initialization with invalid constraint string."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint='invalid_constraint',
            eps=0.05
        )
        
        # Error should occur at fit, not init
        assert model.constraint == 'invalid_constraint'
    
    def test_fit_demographic_parity(self, sample_data):
        """Test fitting with demographic parity constraint."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.05,
            max_iter=20
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        assert model.fitted_
        assert model.model_ is not None
    
    def test_fit_equalized_odds(self, sample_data):
        """Test fitting with equalized odds constraint."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='equalized_odds',
            eps=0.05,
            max_iter=20
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        assert model.fitted_
    
    def test_fit_equal_opportunity(self, sample_data):
        """Test fitting with equal opportunity constraint."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='equal_opportunity',
            eps=0.05,
            max_iter=20
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        assert model.fitted_
    
    def test_fit_error_rate_parity(self, sample_data):
        """Test fitting with error rate parity constraint."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='error_rate_parity',
            eps=0.05,
            max_iter=20
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        assert model.fitted_
    
    def test_fit_with_pandas(self, sample_data):
        """Test fitting with pandas DataFrames/Series."""
        X_df = pd.DataFrame(sample_data['X_train'])
        y_series = pd.Series(sample_data['y_train'])
        s_series = pd.Series(sample_data['s_train'])
        
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.05
        )
        
        model.fit(X_df, y_series, sensitive_features=s_series)
        
        assert model.fitted_
    
    def test_predict(self, sample_data):
        """Test predict method."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.05
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        predictions = model.predict(sample_data['X_test'])
        
        assert predictions.shape == (len(sample_data['X_test']),)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_before_fit(self, sample_data):
        """Test predict before fitting raises error."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint='demographic_parity'
        )
        
        with pytest.raises(ValueError, match="Must call fit"):
            model.predict(sample_data['X_test'])
    
    def test_predict_proba(self, sample_data):
        """Test predict_proba method."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.05
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        # Note: Fairlearn reductions may not have predict_proba
        # This test checks the fallback behavior
        proba = model.predict_proba(sample_data['X_test'])
        
        assert proba.shape == (len(sample_data['X_test']), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_score(self, sample_data):
        """Test score method."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.05
        )
        
        model.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        accuracy = model.score(
            sample_data['X_test'],
            sample_data['y_test']
        )
        
        assert 0 <= accuracy <= 1
    
    def test_get_params(self):
        """Test get_params method."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint='demographic_parity',
            eps=0.05,
            max_iter=50
        )
        
        params = model.get_params()
        
        assert params['constraint'] == 'demographic_parity'
        assert params['eps'] == 0.05
        assert params['max_iter'] == 50
    
    def test_set_params(self):
        """Test set_params method."""
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(),
            constraint='demographic_parity',
            eps=0.05
        )
        
        model.set_params(eps=0.1, max_iter=100)
        
        assert model.eps == 0.1
        assert model.max_iter == 100
    
    def test_different_base_estimators(self, sample_data):
        """Test with different base estimators."""
        estimators = [
            LogisticRegression(max_iter=1000),
            RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42),
        ]
        
        for estimator in estimators:
            model = ReductionsWrapper(
                base_estimator=estimator,
                constraint='demographic_parity',
                eps=0.05,
                max_iter=10
            )
            
            model.fit(
                sample_data['X_train'],
                sample_data['y_train'],
                sensitive_features=sample_data['s_train']
            )
            
            predictions = model.predict(sample_data['X_test'])
            
            assert predictions.shape == (len(sample_data['X_test']),)
            assert model.fitted_


@pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
class TestGridSearchReductions:
    """Test suite for GridSearchReductions."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        
        X, y = make_classification(
            n_samples=300,
            n_features=8,
            n_informative=4,
            random_state=42
        )
        
        sensitive = np.random.binomial(1, 0.5, size=300)
        
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=42
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            's_train': s_train,
            'X_test': X_test,
            'y_test': y_test,
            's_test': s_test,
        }
    
    def test_initialization(self):
        """Test initialization."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(),
            constraints=['demographic_parity', 'equalized_odds'],
            eps_values=[0.01, 0.05, 0.1]
        )
        
        assert grid_search.base_estimator is not None
        assert len(grid_search.constraints) == 2
        assert len(grid_search.eps_values) == 3
        assert grid_search.best_estimator_ is None
    
    def test_fit(self, sample_data):
        """Test fitting grid search."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(max_iter=1000),
            constraints=['demographic_parity'],
            eps_values=[0.05, 0.1],
            max_iter=10
        )
        
        grid_search.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        assert grid_search.best_estimator_ is not None
        assert grid_search.best_params_ is not None
        assert grid_search.best_score_ is not None
        assert len(grid_search.cv_results_) > 0
    
    def test_fit_multiple_constraints(self, sample_data):
        """Test fitting with multiple constraints."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(max_iter=1000),
            constraints=['demographic_parity', 'equalized_odds'],
            eps_values=[0.05],
            max_iter=10
        )
        
        grid_search.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        # Should have tried 2 constraints × 1 eps = 2 models
        assert len(grid_search.cv_results_) == 2
        assert grid_search.best_estimator_ is not None
    
    def test_predict(self, sample_data):
        """Test predict method."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(max_iter=1000),
            constraints=['demographic_parity'],
            eps_values=[0.05],
            max_iter=10
        )
        
        grid_search.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        predictions = grid_search.predict(sample_data['X_test'])
        
        assert predictions.shape == (len(sample_data['X_test']),)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_before_fit(self, sample_data):
        """Test predict before fitting raises error."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(),
            constraints=['demographic_parity'],
            eps_values=[0.05]
        )
        
        with pytest.raises(ValueError, match="Must call fit"):
            grid_search.predict(sample_data['X_test'])
    
    def test_score(self, sample_data):
        """Test score method."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(max_iter=1000),
            constraints=['demographic_parity'],
            eps_values=[0.05],
            max_iter=10
        )
        
        grid_search.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        accuracy = grid_search.score(
            sample_data['X_test'],
            sample_data['y_test']
        )
        
        assert 0 <= accuracy <= 1
    
    def test_best_model_selection(self, sample_data):
        """Test that best model is selected correctly."""
        grid_search = GridSearchReductions(
            base_estimator=LogisticRegression(max_iter=1000),
            constraints=['demographic_parity'],
            eps_values=[0.01, 0.05, 0.1],
            max_iter=10
        )
        
        grid_search.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sensitive_features=sample_data['s_train']
        )
        
        # Best score should be from results
        scores = [r['score'] for r in grid_search.cv_results_]
        assert grid_search.best_score_ == max(scores)
        
        # Best params should match best result
        best_result = max(grid_search.cv_results_, key=lambda r: r['score'])
        assert grid_search.best_params_['eps'] == best_result['eps']


@pytest.mark.skipif(FAIRLEARN_AVAILABLE, reason="Test for when Fairlearn is not installed")
class TestWithoutFairlearn:
    """Test behavior when Fairlearn is not installed."""
    
    def test_import_error(self):
        """Test that appropriate error is raised when Fairlearn not available."""
        # This test only runs when FAIRLEARN_AVAILABLE is False
        # The actual import should have already raised warning
        assert not FAIRLEARN_AVAILABLE


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(50, 5)
        y = np.random.binomial(1, 0.5, 50)
        s = np.random.binomial(1, 0.5, 50)
        
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.1,
            max_iter=10
        )
        
        model.fit(X, y, sensitive_features=s)
        predictions = model.predict(X)
        
        assert predictions.shape == (50,)
    
    @pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
    def test_imbalanced_groups(self):
        """Test with highly imbalanced groups."""
        X = np.random.randn(200, 5)
        y = np.random.binomial(1, 0.5, 200)
        s = np.zeros(200)
        s[:20] = 1  # Only 10% in group 1
        
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.1,
            max_iter=10
        )
        
        model.fit(X, y, sensitive_features=s)
        predictions = model.predict(X)
        
        assert predictions.shape == (200,)
    
    @pytest.mark.skipif(not FAIRLEARN_AVAILABLE, reason="Fairlearn not installed")
    def test_all_same_label(self):
        """Test with all same label."""
        X = np.random.randn(100, 5)
        y = np.ones(100)  # All positive
        s = np.random.binomial(1, 0.5, 100)
        
        model = ReductionsWrapper(
            base_estimator=LogisticRegression(max_iter=1000),
            constraint='demographic_parity',
            eps=0.1,
            max_iter=10
        )
        
        # Should handle gracefully or raise informative error
        try:
            model.fit(X, y, sensitive_features=s)
            predictions = model.predict(X)
            assert predictions.shape == (100,)
        except Exception as e:
            # If it fails, it should be a clear error
            assert "label" in str(e).lower() or "class" in str(e).lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])