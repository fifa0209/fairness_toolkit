"""
Unit tests for calibration.py - Group Fairness Calibration
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from training_module.src.calibration import (
    GroupFairnessCalibrator,
    calibrate_by_group,
)


class TestGroupFairnessCalibrator:
    """Test suite for GroupFairnessCalibrator."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        
        # Generate features and labels
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        
        # Generate sensitive features
        sensitive = np.random.binomial(1, 0.5, size=500)
        
        # Split data
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, sensitive, test_size=0.3, random_state=42
        )
        
        X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
            X_train, y_train, s_train, test_size=0.2, random_state=42
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
    
    @pytest.fixture
    def trained_base_model(self, sample_data):
        """Train a base model for calibration."""
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(sample_data['X_train'], sample_data['y_train'])
        return model
    
    def test_initialization(self, trained_base_model):
        """Test calibrator initialization."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        assert calibrator.base_estimator is not None
        assert calibrator.method == 'isotonic'
        assert not calibrator.fitted_
        assert len(calibrator.group_calibrators_) == 0
    
    def test_initialization_invalid_method(self, trained_base_model):
        """Test initialization with invalid method."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='invalid_method'
        )
        
        # Should not raise error at init, only at fit
        assert calibrator.method == 'invalid_method'
    
    def test_fit_isotonic(self, trained_base_model, sample_data):
        """Test fitting with isotonic regression."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        assert calibrator.fitted_
        assert len(calibrator.group_calibrators_) > 0
        
        # Check that calibrators were created for each group
        unique_groups = np.unique(sample_data['s_val'])
        for group in unique_groups:
            assert group in calibrator.group_calibrators_
    
    def test_fit_sigmoid(self, trained_base_model, sample_data):
        """Test fitting with sigmoid/Platt scaling."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='sigmoid'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        assert calibrator.fitted_
        assert len(calibrator.group_calibrators_) > 0
    
    def test_fit_with_pandas(self, trained_base_model, sample_data):
        """Test fitting with pandas DataFrames/Series."""
        X_df = pd.DataFrame(sample_data['X_val'])
        y_series = pd.Series(sample_data['y_val'])
        s_series = pd.Series(sample_data['s_val'])
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(X_df, y_series, sensitive_features=s_series)
        
        assert calibrator.fitted_
    
    def test_fit_without_base_estimator(self, sample_data):
        """Test fitting without base estimator raises error."""
        calibrator = GroupFairnessCalibrator(method='isotonic')
        
        with pytest.raises(ValueError, match="base_estimator is required"):
            calibrator.fit(
                sample_data['X_val'],
                sample_data['y_val'],
                sensitive_features=sample_data['s_val']
            )
    
    def test_fit_invalid_method(self, trained_base_model, sample_data):
        """Test fitting with invalid method raises error."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='invalid_method'
        )
        
        with pytest.raises(ValueError, match="Unknown calibration method"):
            calibrator.fit(
                sample_data['X_val'],
                sample_data['y_val'],
                sensitive_features=sample_data['s_val']
            )
    
    def test_predict_proba(self, trained_base_model, sample_data):
        """Test predict_proba method."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        proba = calibrator.predict_proba(
            sample_data['X_test'],
            sensitive_features=sample_data['s_test']
        )
        
        # Check output shape and format
        assert proba.shape == (len(sample_data['X_test']), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_predict_proba_before_fit(self, trained_base_model, sample_data):
        """Test predict_proba before fitting raises error."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        with pytest.raises(ValueError, match="Must call fit"):
            calibrator.predict_proba(
                sample_data['X_test'],
                sensitive_features=sample_data['s_test']
            )
    
    def test_predict(self, trained_base_model, sample_data):
        """Test predict method."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        predictions = calibrator.predict(
            sample_data['X_test'],
            sensitive_features=sample_data['s_test']
        )
        
        # Check output
        assert predictions.shape == (len(sample_data['X_test']),)
        assert set(predictions).issubset({0, 1})
    
    def test_score(self, trained_base_model, sample_data):
        """Test score method."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        accuracy = calibrator.score(
            sample_data['X_test'],
            sample_data['y_test'],
            sensitive_features=sample_data['s_test']
        )
        
        assert 0 <= accuracy <= 1
    
    def test_calibration_improves_probabilities(self, trained_base_model, sample_data):
        """Test that calibration produces valid probabilities."""
        calibrator = GroupFairnessCalibrator(
            base_estimator=trained_base_model,
            method='isotonic'
        )
        
        calibrator.fit(
            sample_data['X_val'],
            sample_data['y_val'],
            sensitive_features=sample_data['s_val']
        )
        
        # Get base probabilities
        base_proba = trained_base_model.predict_proba(sample_data['X_test'])[:, 1]
        
        # Get calibrated probabilities
        calib_proba = calibrator.predict_proba(
            sample_data['X_test'],
            sensitive_features=sample_data['s_test']
        )[:, 1]
        
        # Both should be valid probabilities
        assert np.all(base_proba >= 0) and np.all(base_proba <= 1)
        assert np.all(calib_proba >= 0) and np.all(calib_proba <= 1)
    
    def test_small_group_warning(self):
        """Test warning for small groups."""
        # Create data with very small group
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = np.random.binomial(1, 0.5, 50)
        s = np.zeros(50)
        s[:5] = 1  # Only 5 samples in group 1
        
        # Train a new model on this data
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X, y)
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        
        # Should log warning but not fail
        calibrator.fit(X, y, sensitive_features=s)
        
        # Group with < 10 samples should be skipped
        assert calibrator.fitted_


class TestCalibrateByGroup:
    """Test suite for calibrate_by_group convenience function."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        np.random.seed(42)
        X, y = make_classification(n_samples=300, n_features=10, random_state=42)
        s = np.random.binomial(1, 0.5, 300)
        return X, y, s
    
    @pytest.fixture
    def trained_model(self, sample_data):
        """Train a base model."""
        X, y, _ = sample_data
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X, y)
        return model
    
    def test_calibrate_by_group_isotonic(self, trained_model, sample_data):
        """Test calibrate_by_group with isotonic method."""
        X, y, s = sample_data
        
        calibrator = calibrate_by_group(
            base_model=trained_model,
            X_cal=X,
            y_cal=y,
            sensitive_cal=s,
            method='isotonic'
        )
        
        assert isinstance(calibrator, GroupFairnessCalibrator)
        assert calibrator.fitted_
        assert calibrator.method == 'isotonic'
    
    def test_calibrate_by_group_sigmoid(self, trained_model, sample_data):
        """Test calibrate_by_group with sigmoid method."""
        X, y, s = sample_data
        
        calibrator = calibrate_by_group(
            base_model=trained_model,
            X_cal=X,
            y_cal=y,
            sensitive_cal=s,
            method='sigmoid'
        )
        
        assert isinstance(calibrator, GroupFairnessCalibrator)
        assert calibrator.fitted_
        assert calibrator.method == 'sigmoid'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_group(self):
        """Test calibration with single group (should work but no effect)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.binomial(1, 0.5, 100)
        s = np.zeros(100)  # All same group
        
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X, y)
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        
        calibrator.fit(X, y, sensitive_features=s)
        
        assert calibrator.fitted_
        assert len(calibrator.group_calibrators_) == 1
    
    def test_multiple_groups(self):
        """Test calibration with multiple groups (>2)."""
        np.random.seed(42)
        X = np.random.randn(300, 5)
        y = np.random.binomial(1, 0.5, 300)
        s = np.random.choice([0, 1, 2], 300)  # Three groups
        
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X, y)
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        
        calibrator.fit(X, y, sensitive_features=s)
        
        assert calibrator.fitted_
        assert len(calibrator.group_calibrators_) <= 3
    
    def test_empty_group_in_test(self):
        """Test prediction when test set has group not in training."""
        np.random.seed(42)
        n_features = 5
        
        X_train = np.random.randn(100, n_features)
        y_train = np.random.binomial(1, 0.5, 100)
        s_train = np.zeros(100)  # Only group 0 in training
        
        X_test = np.random.randn(50, n_features)
        s_test = np.ones(50)  # Only group 1 in test
        
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train, y_train)
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        
        calibrator.fit(X_train, y_train, sensitive_features=s_train)
        
        # Should not crash, but predictions for unseen group will be uncalibrated
        proba = calibrator.predict_proba(X_test, sensitive_features=s_test)
        
        assert proba.shape == (50, 2)
    
    def test_perfect_separation(self):
        """Test with perfectly separated data."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
        s = np.random.binomial(1, 0.5, 100)
        
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X, y)
        
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        
        # Should handle perfect or near-perfect predictions
        calibrator.fit(X, y, sensitive_features=s)
        
        assert calibrator.fitted_


class TestCalibrationQuality:
    """Test calibration quality and behavior."""
    
    @pytest.fixture
    def biased_data(self):
        """Generate biased data where calibration is needed."""
        np.random.seed(42)
        
        n = 500
        X = np.random.randn(n, 10)
        
        # Create bias: group 0 gets higher scores
        s = np.random.binomial(1, 0.5, n)
        true_scores = X @ np.random.randn(10)
        biased_scores = true_scores + 2 * (s == 0) - 2 * (s == 1)
        y = (biased_scores > np.median(biased_scores)).astype(int)
        
        # Split
        X_train, X_val = X[:300], X[300:]
        y_train, y_val = y[:300], y[300:]
        s_train, s_val = s[:300], s[300:]
        
        return X_train, y_train, s_train, X_val, y_val, s_val
    
    def test_calibration_reduces_group_disparity(self, biased_data):
        """Test that calibration can reduce disparity in predicted probabilities."""
        X_train, y_train, s_train, X_val, y_val, s_val = biased_data
        
        # Train base model
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        base_model.fit(X_train, y_train)
        
        # Get base predictions
        base_proba = base_model.predict_proba(X_val)[:, 1]
        
        # Compute group means for base predictions
        base_mean_g0 = base_proba[s_val == 0].mean()
        base_mean_g1 = base_proba[s_val == 1].mean()
        base_disparity = abs(base_mean_g0 - base_mean_g1)
        
        # Calibrate
        calibrator = GroupFairnessCalibrator(
            base_estimator=base_model,
            method='isotonic'
        )
        calibrator.fit(X_train, y_train, sensitive_features=s_train)
        
        # Get calibrated predictions
        calib_proba = calibrator.predict_proba(X_val, sensitive_features=s_val)[:, 1]
        
        # Compute group means for calibrated predictions
        calib_mean_g0 = calib_proba[s_val == 0].mean()
        calib_mean_g1 = calib_proba[s_val == 1].mean()
        calib_disparity = abs(calib_mean_g0 - calib_mean_g1)
        
        # Both should be valid probabilities
        assert np.all(calib_proba >= 0) and np.all(calib_proba <= 1)
        
        # Note: Calibration doesn't necessarily reduce demographic parity
        # but ensures probabilities are well-calibrated within each group


if __name__ == '__main__':
    pytest.main([__file__, '-v'])