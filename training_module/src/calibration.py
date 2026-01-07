"""
Group Fairness Calibration - Post-training calibration per group.

Calibrates model predictions separately for each protected group
to ensure fair confidence scores.
"""

import numpy as np
from typing import Dict, Optional, Union, Any
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from shared.logging import get_logger

logger = get_logger(__name__)


class GroupFairnessCalibrator(BaseEstimator, ClassifierMixin):
    """
    Calibrate predictions separately per protected group.
    
    Ensures that predicted probabilities are well-calibrated
    within each group, improving fairness of confidence scores.
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> # Train base model
        >>> base_model = LogisticRegression()
        >>> base_model.fit(X_train, y_train)
        >>> 
        >>> # Calibrate per group
        >>> calibrator = GroupFairnessCalibrator(
        ...     base_estimator=base_model,
        ...     method='isotonic'
        ... )
        >>> calibrator.fit(
        ...     X_val, y_val, sensitive_features=s_val
        ... )
        >>> 
        >>> # Get calibrated predictions
        >>> proba = calibrator.predict_proba(X_test, sensitive_features=s_test)
    """
    
    def __init__(
        self,
        base_estimator: Any = None,
        method: str = 'isotonic',
        cv: int = 5,
    ):
        """
        Initialize group calibrator.
        
        Args:
            base_estimator: Pre-trained base model
            method: Calibration method ('isotonic' or 'sigmoid')
            cv: Number of CV folds (for sklearn CalibratedClassifierCV)
        """
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv
        
        self.group_calibrators_ = {}
        self.fitted_ = False
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> 'GroupFairnessCalibrator':
        """
        Fit group-specific calibrators.
        
        Args:
            X: Validation features
            y: Validation labels
            sensitive_features: Protected attribute
            
        Returns:
            self (fitted calibrator)
        """
        if self.base_estimator is None:
            raise ValueError("base_estimator is required")
        
        # Convert to arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        sf_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else sensitive_features
        
        # Get base predictions
        if hasattr(self.base_estimator, 'predict_proba'):
            base_proba = self.base_estimator.predict_proba(X_arr)
            if base_proba.shape[1] == 2:
                base_proba = base_proba[:, 1]  # Probability of positive class
        else:
            raise ValueError("base_estimator must have predict_proba method")
        
        # Fit calibrator for each group
        groups = np.unique(sf_arr)
        
        for group in groups:
            mask = sf_arr == group
            X_group = X_arr[mask]
            y_group = y_arr[mask]
            proba_group = base_proba[mask]
            
            if len(y_group) < 10:
                logger.warning(f"Group {group} has only {len(y_group)} samples. Skipping calibration.")
                continue
            
            # Fit calibrator
            if self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(proba_group, y_group)
            
            elif self.method == 'sigmoid':
                # Platt scaling: fit logistic regression to log-odds
                from sklearn.linear_model import LogisticRegression
                
                # Convert probabilities to log-odds
                epsilon = 1e-10
                proba_clipped = np.clip(proba_group, epsilon, 1 - epsilon)
                log_odds = np.log(proba_clipped / (1 - proba_clipped)).reshape(-1, 1)
                
                calibrator = LogisticRegression()
                calibrator.fit(log_odds, y_group)
            
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            self.group_calibrators_[group] = calibrator
            logger.info(f"Calibrated group {group} ({len(y_group)} samples)")
        
        self.fitted_ = True
        return self
    
    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """
        Get calibrated probability predictions.
        
        Args:
            X: Features
            sensitive_features: Protected attribute
            
        Returns:
            Calibrated probabilities (n_samples, 2)
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before predict_proba()")
        
        # Convert to arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        sf_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else sensitive_features
        
        # Get base predictions
        base_proba = self.base_estimator.predict_proba(X_arr)
        if base_proba.shape[1] == 2:
            base_proba_pos = base_proba[:, 1]
        else:
            base_proba_pos = base_proba
        
        # Apply group-specific calibration
        calibrated_proba = np.zeros(len(X_arr))
        
        for group, calibrator in self.group_calibrators_.items():
            mask = sf_arr == group
            
            if not mask.any():
                continue
            
            if self.method == 'isotonic':
                calibrated_proba[mask] = calibrator.predict(base_proba_pos[mask])
            
            elif self.method == 'sigmoid':
                # Convert to log-odds, apply calibration
                epsilon = 1e-10
                proba_clipped = np.clip(base_proba_pos[mask], epsilon, 1 - epsilon)
                log_odds = np.log(proba_clipped / (1 - proba_clipped)).reshape(-1, 1)
                calibrated_proba[mask] = calibrator.predict_proba(log_odds)[:, 1]
        
        # Return in sklearn format (n_samples, 2)
        proba_output = np.zeros((len(X_arr), 2))
        proba_output[:, 1] = calibrated_proba
        proba_output[:, 0] = 1 - calibrated_proba
        
        return proba_output
    
    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Get binary predictions (threshold=0.5)."""
        proba = self.predict_proba(X, sensitive_features)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy."""
        y_arr = y.values if isinstance(y, pd.Series) else y
        y_pred = self.predict(X, sensitive_features)
        return (y_pred == y_arr).mean()


def calibrate_by_group(
    base_model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    sensitive_cal: np.ndarray,
    method: str = 'isotonic',
) -> GroupFairnessCalibrator:
    """
    Convenience function to calibrate a trained model.
    
    Args:
        base_model: Trained model with predict_proba
        X_cal: Calibration features
        y_cal: Calibration labels
        sensitive_cal: Calibration sensitive features
        method: 'isotonic' or 'sigmoid'
        
    Returns:
        Fitted GroupFairnessCalibrator
    """
    calibrator = GroupFairnessCalibrator(
        base_estimator=base_model,
        method=method
    )
    
    calibrator.fit(X_cal, y_cal, sensitive_features=sensitive_cal)
    
    return calibrator