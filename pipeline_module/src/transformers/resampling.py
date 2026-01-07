"""
Resampling Transformers - Modify dataset size to balance groups.

Placeholder for future SMOTE and advanced resampling methods.
48-hour scope: Simple over/under sampling in GroupBalancer.
"""

import numpy as np
from typing import Union, Tuple
import pandas as pd

from pipeline_module.src.transformers.base import FairnessTransformer
from shared.logging import get_logger

logger = get_logger(__name__)


class SimpleOversampler(FairnessTransformer):
    """
    Simple oversampling - duplicate minority group samples.
    
    Note: This is a simplified version. For production use SMOTE.
    
    Example:
        >>> oversampler = SimpleOversampler(random_state=42)
        >>> X_balanced, y_balanced = oversampler.fit_resample(
        ...     X_train, y_train, sensitive_features=s_train
        ... )
    """
    
    def __init__(self, random_state: int = None):
        """
        Initialize oversampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.fitted_ = False
    
    def fit(self, X, y=None, sensitive_features=None):
        """Fit (no-op for this transformer)."""
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Transform returns X unchanged. Use fit_resample instead."""
        return X
    
    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority groups to match majority size.
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Protected attribute
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X_arr, y_arr, sf_arr = self._validate_inputs(X, y, sensitive_features)
        
        # Find majority group size
        groups, counts = np.unique(sf_arr, return_counts=True)
        target_size = max(counts)
        
        # Oversample each group
        X_resampled = []
        y_resampled = []
        
        for group in groups:
            mask = sf_arr == group
            X_group = X_arr[mask]
            y_group = y_arr[mask]
            n_current = len(X_group)
            
            if n_current < target_size:
                # Need to oversample
                n_to_add = target_size - n_current
                indices = np.random.choice(n_current, size=n_to_add, replace=True)
                X_resampled.append(X_group)
                X_resampled.append(X_group[indices])
                y_resampled.append(y_group)
                y_resampled.append(y_group[indices])
            else:
                X_resampled.append(X_group)
                y_resampled.append(y_group)
        
        X_result = np.vstack(X_resampled)
        y_result = np.concatenate(y_resampled)
        
        logger.info(f"Oversampled: {len(X_arr)} -> {len(X_result)} samples")
        
        return X_result, y_result


class GroupBalancer(FairnessTransformer):
    """
    Unified group balancer - supports multiple resampling strategies.
    
    This is a convenience wrapper that provides a single interface for
    different resampling strategies to balance protected groups.
    
    Strategies:
        - 'oversample': Duplicate minority samples to match majority group size
        - 'undersample': Remove majority samples to match minority group size
    
    Example:
        >>> # Oversample minority groups
        >>> balancer = GroupBalancer(strategy='oversample', random_state=42)
        >>> X_balanced, y_balanced = balancer.fit_resample(
        ...     X_train, y_train, sensitive_features=s_train
        ... )
        >>> 
        >>> # Undersample majority groups
        >>> balancer = GroupBalancer(strategy='undersample', random_state=42)
        >>> X_balanced, y_balanced = balancer.fit_resample(
        ...     X_train, y_train, sensitive_features=s_train
        ... )
    """
    
    def __init__(self, strategy: str = 'oversample', random_state: int = None):
        """
        Initialize group balancer.
        
        Args:
            strategy: Resampling strategy ('oversample' or 'undersample')
            random_state: Random seed for reproducibility
            
        Raises:
            ValueError: If strategy is not 'oversample' or 'undersample'
        """
        if strategy not in ['oversample', 'undersample']:
            raise ValueError(
                f"strategy must be 'oversample' or 'undersample', got '{strategy}'"
            )
        
        self.strategy = strategy
        self.random_state = random_state
        self.fitted_ = False
        
        # Initialize appropriate underlying sampler
        if strategy == 'oversample':
            self._sampler = SimpleOversampler(random_state=random_state)
        else:
            self._sampler = SimpleUndersampler(random_state=random_state)
        
        logger.info(f"Initialized GroupBalancer with strategy='{strategy}'")
    
    def fit(self, X, y=None, sensitive_features=None):
        """
        Fit the balancer (delegates to underlying sampler).
        
        Args:
            X: Features
            y: Labels (optional)
            sensitive_features: Protected attribute (optional)
            
        Returns:
            self
        """
        self._sampler.fit(X, y, sensitive_features)
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """
        Transform returns X unchanged. Use fit_resample instead.
        
        Note: Resampling cannot be applied in a standard transform() call
        because it changes the number of samples. Use fit_resample() instead.
        
        Args:
            X: Features
            
        Returns:
            X unchanged
        """
        return X
    
    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance groups using the specified resampling strategy.
        
        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)
            sensitive_features: Protected attribute values (n_samples,)
            
        Returns:
            Tuple of (X_resampled, y_resampled) with balanced group sizes
            
        Example:
            >>> balancer = GroupBalancer(strategy='oversample')
            >>> X_new, y_new = balancer.fit_resample(X, y, sensitive_features=gender)
            >>> # X_new and y_new now have equal representation of all groups
        """
        logger.info(f"Resampling data using strategy='{self.strategy}'")
        return self._sampler.fit_resample(X, y, sensitive_features)


class SimpleUndersampler(FairnessTransformer):
    """
    Simple undersampling - randomly remove majority group samples.
    
    Example:
        >>> undersampler = SimpleUndersampler(random_state=42)
        >>> X_balanced, y_balanced = undersampler.fit_resample(
        ...     X_train, y_train, sensitive_features=s_train
        ... )
    """
    
    def __init__(self, random_state: int = None):
        """
        Initialize undersampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.fitted_ = False
    
    def fit(self, X, y=None, sensitive_features=None):
        """Fit (no-op for this transformer)."""
        self.fitted_ = True
        return self
    
    def transform(self, X):
        """Transform returns X unchanged. Use fit_resample instead."""
        return X
    
    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Undersample majority groups to match minority size.
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Protected attribute
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X_arr, y_arr, sf_arr = self._validate_inputs(X, y, sensitive_features)
        
        # Find minority group size
        groups, counts = np.unique(sf_arr, return_counts=True)
        target_size = min(counts)
        
        # Undersample each group
        X_resampled = []
        y_resampled = []
        
        for group in groups:
            mask = sf_arr == group
            X_group = X_arr[mask]
            y_group = y_arr[mask]
            n_current = len(X_group)
            
            if n_current > target_size:
                # Need to undersample
                indices = np.random.choice(n_current, size=target_size, replace=False)
                X_resampled.append(X_group[indices])
                y_resampled.append(y_group[indices])
            else:
                X_resampled.append(X_group)
                y_resampled.append(y_group)
        
        X_result = np.vstack(X_resampled)
        y_result = np.concatenate(y_resampled)
        
        logger.info(f"Undersampled: {len(X_arr)} -> {len(X_result)} samples")
        
        return X_result, y_result