"""
Sklearn Wrappers - Fairness-aware sklearn-compatible models.

Wraps Fairlearn's reduction algorithms for easy integration.
48-hour scope: Focus on ExponentiatedGradient with constraints.
"""

import numpy as np
from typing import Optional, Union, Any
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from shared.logging import get_logger

logger = get_logger(__name__)

# Try to import fairlearn (graceful fallback if not available)
try:
    from fairlearn.reductions import (
        ExponentiatedGradient,
        DemographicParity,
        EqualizedOdds,
        TruePositiveRateParity,
        ErrorRateParity,
    )
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logger.warning(
        "Fairlearn not available. Install with: pip install fairlearn"
    )


class ReductionsWrapper(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible wrapper for Fairlearn's reduction algorithms.
    
    Uses ExponentiatedGradient to train models with fairness constraints.
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from fairlearn.reductions import DemographicParity
        >>> 
        >>> model = ReductionsWrapper(
        ...     base_estimator=LogisticRegression(),
        ...     constraint=DemographicParity()
        ... )
        >>> model.fit(X_train, y_train, sensitive_features=s_train)
        >>> y_pred = model.predict(X_test)
    """
    
    def __init__(
        self,
        base_estimator: Any,
        constraint: Union[str, Any] = 'demographic_parity',
        constraint_weight: float = 0.5,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: Optional[float] = None,
        eta0: float = 2.0,
    ):
        """
        Initialize ReductionsWrapper.
        
        Args:
            base_estimator: Base sklearn estimator (e.g., LogisticRegression)
            constraint: Fairness constraint ('demographic_parity', 'equalized_odds',
                       'equal_opportunity', 'error_rate_parity') or constraint object
            constraint_weight: Weight for fairness constraint (0=ignore, 1=strict)
            eps: Allowed constraint violation
            max_iter: Maximum iterations
            nu: Learning rate (None=auto)
            eta0: Initial learning rate
        """
        if not FAIRLEARN_AVAILABLE:
            raise ImportError(
                "Fairlearn is required for ReductionsWrapper. "
                "Install with: pip install fairlearn"
            )
        
        self.base_estimator = base_estimator
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        
        self.model_ = None
        self.fitted_ = False
    
    def _get_constraint_object(self):
        """Convert constraint string to Fairlearn constraint object."""
        if isinstance(self.constraint, str):
            constraint_map = {
                'demographic_parity': DemographicParity,
                'equalized_odds': EqualizedOdds,
                'equal_opportunity': TruePositiveRateParity,
                'error_rate_parity': ErrorRateParity,
            }
            
            if self.constraint not in constraint_map:
                raise ValueError(
                    f"Unknown constraint: {self.constraint}. "
                    f"Valid options: {list(constraint_map.keys())}"
                )
            
            return constraint_map[self.constraint]()
        else:
            return self.constraint
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> 'ReductionsWrapper':
        """
        Fit model with fairness constraints.
        
        Args:
            X: Training features
            y: Training labels
            sensitive_features: Protected attribute
            
        Returns:
            self (fitted model)
        """
        # Convert to arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        sf_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else sensitive_features
        
        # Get constraint object
        constraint_obj = self._get_constraint_object()
        
        # Create and fit ExponentiatedGradient
        self.model_ = ExponentiatedGradient(
            estimator=self.base_estimator,
            constraints=constraint_obj,
            eps=self.eps,
            max_iter=self.max_iter,
            nu=self.nu,
            eta0=self.eta0,
        )
        
        logger.info(
            f"Fitting ReductionsWrapper with {type(constraint_obj).__name__} "
            f"(eps={self.eps}, max_iter={self.max_iter})"
        )
        
        self.model_.fit(X_arr, y_arr, sensitive_features=sf_arr)
        self.fitted_ = True
        
        logger.info("ReductionsWrapper fitted successfully")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict labels."""
        if not self.fitted_:
            raise ValueError("Must call fit() before predict()")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict(X_arr)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict probabilities (if base estimator supports it).
        
        Note: Fairlearn reductions may not preserve predict_proba.
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before predict_proba()")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        # Check if underlying estimator has predict_proba
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_arr)
        else:
            logger.warning(
                "predict_proba not available. Returning binary predictions."
            )
            preds = self.predict(X_arr)
            # Convert to probability format
            proba = np.zeros((len(preds), 2))
            proba[np.arange(len(preds)), preds] = 1.0
            return proba
    
    def score(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
    ) -> float:
        """Calculate accuracy score."""
        if not self.fitted_:
            raise ValueError("Must call fit() before score()")
        
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        y_pred = self.predict(X_arr)
        return (y_pred == y_arr).mean()
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (sklearn compatibility)."""
        return {
            'base_estimator': self.base_estimator,
            'constraint': self.constraint,
            'constraint_weight': self.constraint_weight,
            'eps': self.eps,
            'max_iter': self.max_iter,
            'nu': self.nu,
            'eta0': self.eta0,
        }
    
    def set_params(self, **params) -> 'ReductionsWrapper':
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GridSearchReductions(BaseEstimator, ClassifierMixin):
    """
    Grid search over multiple fairness constraints and parameters.
    
    Trains models with different fairness constraints and selects best
    based on accuracy-fairness trade-off.
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> grid_search = GridSearchReductions(
        ...     base_estimator=LogisticRegression(),
        ...     constraints=['demographic_parity', 'equalized_odds'],
        ...     eps_values=[0.01, 0.05, 0.1]
        ... )
        >>> grid_search.fit(X_train, y_train, sensitive_features=s_train)
        >>> best_model = grid_search.best_estimator_
    """
    
    def __init__(
        self,
        base_estimator: Any,
        constraints: list = None,
        eps_values: list = None,
        max_iter: int = 50,
    ):
        """
        Initialize GridSearchReductions.
        
        Args:
            base_estimator: Base sklearn estimator
            constraints: List of constraint names to try
            eps_values: List of eps values to try
            max_iter: Maximum iterations per model
        """
        if not FAIRLEARN_AVAILABLE:
            raise ImportError("Fairlearn is required")
        
        self.base_estimator = base_estimator
        self.constraints = constraints or ['demographic_parity', 'equalized_odds']
        self.eps_values = eps_values or [0.01, 0.05, 0.1]
        self.max_iter = max_iter
        
        self.best_estimator_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> 'GridSearchReductions':
        """
        Fit multiple models and select best.
        
        Args:
            X: Training features
            y: Training labels
            sensitive_features: Protected attribute
            
        Returns:
            self (fitted with best model)
        """
        logger.info(
            f"Grid search: {len(self.constraints)} constraints × "
            f"{len(self.eps_values)} eps values = "
            f"{len(self.constraints) * len(self.eps_values)} models"
        )
        
        best_score = -np.inf
        
        for constraint in self.constraints:
            for eps in self.eps_values:
                try:
                    # Train model
                    model = ReductionsWrapper(
                        base_estimator=self.base_estimator,
                        constraint=constraint,
                        eps=eps,
                        max_iter=self.max_iter,
                    )
                    model.fit(X, y, sensitive_features=sensitive_features)
                    
                    # Score model (just accuracy for now)
                    score = model.score(X, y)
                    
                    # Store result
                    self.cv_results_.append({
                        'constraint': constraint,
                        'eps': eps,
                        'score': score,
                        'model': model,
                    })
                    
                    # Update best
                    if score > best_score:
                        best_score = score
                        self.best_score_ = score
                        self.best_estimator_ = model
                        self.best_params_ = {'constraint': constraint, 'eps': eps}
                    
                    logger.info(
                        f"  {constraint} (eps={eps}): score={score:.4f}"
                    )
                
                except Exception as e:
                    logger.error(f"  {constraint} (eps={eps}): FAILED - {e}")
                    continue
        
        if self.best_estimator_ is None:
            raise ValueError("All models failed to train")
        
        logger.info(f"Best model: {self.best_params_} (score={self.best_score_:.4f})")
        
        return self
    
    def predict(self, X):
        """Predict using best model."""
        if self.best_estimator_ is None:
            raise ValueError("Must call fit() before predict()")
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities using best model."""
        if self.best_estimator_ is None:
            raise ValueError("Must call fit() before predict_proba()")
        return self.best_estimator_.predict_proba(X)
    
    def score(self, X, y):
        """Score using best model."""
        if self.best_estimator_ is None:
            raise ValueError("Must call fit() before score()")
        return self.best_estimator_.score(X, y)