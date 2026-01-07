"""
Base Transformer - Abstract base class for fairness transformers.

Provides common interface for all fairness transformers.
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FairnessTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Abstract base class for fairness transformers.
    
    All fairness transformers should inherit from this class
    and implement the required methods.
    """
    
    @abstractmethod
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> 'FairnessTransformer':
        """
        Fit the transformer to training data.
        
        Args:
            X: Training features
            y: Training labels (optional)
            sensitive_features: Protected attribute
            
        Returns:
            self (fitted transformer)
        """
        pass
    
    @abstractmethod
    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> np.ndarray:
        """
        Transform the data.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        pass
    
    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Fit and transform in one step.
        
        Args:
            X: Training features
            y: Training labels (optional)
            sensitive_features: Protected attribute (optional)
            
        Returns:
            Transformed data (may be tuple if transformer modifies multiple arrays)
        """
        self.fit(X, y, sensitive_features)
        return self.transform(X)
    
    def _validate_inputs(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Validate and convert inputs to numpy arrays.
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Protected attribute
            
        Returns:
            Tuple of (X, y, sensitive_features) as numpy arrays
        """
        # Convert to numpy
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else (np.asarray(y) if y is not None else None)
        sf_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else (
            np.asarray(sensitive_features) if sensitive_features is not None else None
        )
        
        # Check shapes
        n_samples = len(X_arr)
        
        if y_arr is not None and len(y_arr) != n_samples:
            raise ValueError(f"X and y have different lengths: {n_samples} vs {len(y_arr)}")
        
        if sf_arr is not None and len(sf_arr) != n_samples:
            raise ValueError(
                f"X and sensitive_features have different lengths: {n_samples} vs {len(sf_arr)}"
            )
        
        return X_arr, y_arr, sf_arr