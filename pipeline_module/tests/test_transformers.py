"""
Unit tests for transformers.

Run: pytest pipeline_module/tests/test_transformers.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from pipeline_module.src.transformers import (
    InstanceReweighting,
    GroupBalancer,
    SimpleOversampler,
    SimpleUndersampler,
)


@pytest.fixture
def imbalanced_data():
    """Create imbalanced dataset."""
    np.random.seed(42)
    
    n_minority = 100
    n_majority = 300
    
    X = np.vstack([
        np.random.randn(n_minority, 5),
        np.random.randn(n_majority, 5)
    ])
    
    y = np.concatenate([
        np.random.choice([0, 1], n_minority),
        np.random.choice([0, 1], n_majority)
    ])
    
    sensitive = np.array([0] * n_minority + [1] * n_majority)
    
    return X, y, sensitive


def test_instance_reweighting_fit(imbalanced_data):
    """Test InstanceReweighting fit method."""
    X, y, sensitive = imbalanced_data
    
    reweighter = InstanceReweighting(method='inverse_propensity', alpha=1.0)
    reweighter.fit(X, y, sensitive_features=sensitive)
    
    assert reweighter.group_weights_ is not None
    assert len(reweighter.group_weights_) == 2


def test_instance_reweighting_weights(imbalanced_data):
    """Test that weights balance groups."""
    X, y, sensitive = imbalanced_data
    
    reweighter = InstanceReweighting(alpha=1.0)
    reweighter.fit(X, y, sensitive_features=sensitive)
    weights = reweighter.get_sample_weights(sensitive)
    
    # Check weights exist
    assert len(weights) == len(X)
    assert np.all(weights > 0)
    
    # Check effective balance
    minority_weight = weights[sensitive == 0].sum()
    majority_weight = weights[sensitive == 1].sum()
    
    # Should be approximately balanced
    ratio = minority_weight / majority_weight
    assert 0.8 < ratio < 1.2


def test_instance_reweighting_fit_transform(imbalanced_data):
    """Test fit_transform returns correct tuple."""
    X, y, sensitive = imbalanced_data
    
    reweighter = InstanceReweighting()
    X_out, y_out, weights = reweighter.fit_transform(X, y, sensitive_features=sensitive)
    
    assert X_out.shape == X.shape
    assert y_out.shape == y.shape
    assert len(weights) == len(X)


def test_group_balancer_oversample(imbalanced_data):
    """Test GroupBalancer oversampling."""
    X, y, sensitive = imbalanced_data
    
    balancer = GroupBalancer(strategy='oversample', random_state=42)
    X_balanced, y_balanced = balancer.fit_resample(X, y, sensitive_features=sensitive)
    
    # Size should increase (oversampling minority)
    assert len(X_balanced) >= len(X)
    assert X_balanced.shape[1] == X.shape[1]


def test_group_balancer_undersample(imbalanced_data):
    """Test GroupBalancer undersampling."""
    X, y, sensitive = imbalanced_data
    
    balancer = GroupBalancer(strategy='undersample', random_state=42)
    X_balanced, y_balanced = balancer.fit_resample(X, y, sensitive_features=sensitive)
    
    # Size should decrease (undersampling majority)
    assert len(X_balanced) <= len(X)
    assert X_balanced.shape[1] == X.shape[1]


def test_simple_oversampler(imbalanced_data):
    """Test SimpleOversampler."""
    X, y, sensitive = imbalanced_data
    
    oversampler = SimpleOversampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y, sensitive_features=sensitive)
    
    assert len(X_resampled) >= len(X)
    assert X_resampled.shape[1] == X.shape[1]


def test_simple_undersampler(imbalanced_data):
    """Test SimpleUndersampler."""
    X, y, sensitive = imbalanced_data
    
    undersampler = SimpleUndersampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y, sensitive_features=sensitive)
    
    assert len(X_resampled) <= len(X)
    assert X_resampled.shape[1] == X.shape[1]


def test_reweighting_with_sklearn_model(imbalanced_data):
    """Test InstanceReweighting works with sklearn models."""
    X, y, sensitive = imbalanced_data
    
    # Split data
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    s_train = sensitive[:split]
    
    # Train with reweighting
    reweighter = InstanceReweighting()
    reweighter.fit(X_train, y_train, sensitive_features=s_train)
    weights = reweighter.get_sample_weights(s_train)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Should train successfully
    score = model.score(X_test, y_test)
    assert 0 <= score <= 1


def test_transformer_sklearn_compatibility(imbalanced_data):
    """Test transformers are sklearn-compatible."""
    from sklearn.base import is_classifier, is_regressor, BaseEstimator, TransformerMixin
    
    reweighter = InstanceReweighting()
    
    assert isinstance(reweighter, BaseEstimator)
    assert isinstance(reweighter, TransformerMixin)
    assert hasattr(reweighter, 'fit')
    assert hasattr(reweighter, 'transform')
    assert hasattr(reweighter, 'get_params')
    assert hasattr(reweighter, 'set_params')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])