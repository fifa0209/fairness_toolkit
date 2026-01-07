"""
Reweighting Transformer - sklearn-compatible transformer for bias mitigation.

Implements instance reweighting to balance representation across protected groups.
Compatible with sklearn.pipeline.Pipeline.

48-hour scope: Simple inverse propensity weighting.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple
from sklearn.base import BaseEstimator, TransformerMixin

from shared.validation import validate_dataframe, validate_sample_weights
from shared.logging import get_logger

logger = get_logger(__name__)


class InstanceReweighting(BaseEstimator, TransformerMixin):
    """
    Reweight samples to balance representation across protected groups.
    
    Sklearn-compatible transformer that assigns weights to samples based
    on their protected group membership, promoting demographic parity.
    
    Example:
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> pipeline = Pipeline([
        ...     ('reweight', InstanceReweighting()),
        ...     ('model', LogisticRegression())
        ... ])
        >>> pipeline.fit(X_train, y_train, 
        ...              reweight__sensitive_features=sensitive_train)
    """
    
    def __init__(
        self,
        method: str = 'inverse_propensity',
        alpha: float = 1.0,
    ):
        """
        Initialize InstanceReweighting transformer.
        
        Args:
            method: Weighting method ('inverse_propensity' or 'uniform')
            alpha: Smoothing parameter (1.0 = no smoothing, 0.0 = uniform weights)
        """
        self.method = method
        self.alpha = alpha
        self.group_weights_ = None
        self.n_groups_ = None
        
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> 'InstanceReweighting':
        """
        Learn group weights from training data.
        
        Args:
            X: Training features (not used, for sklearn compatibility)
            y: Training labels (not used, for sklearn compatibility)
            sensitive_features: Protected attribute for each sample
            
        Returns:
            self (fitted transformer)
        """
        if sensitive_features is None:
            raise ValueError("sensitive_features is required for fit()")
        
        sensitive_features = np.asarray(sensitive_features)
        
        # Count samples per group
        groups, counts = np.unique(sensitive_features, return_counts=True)
        n_total = len(sensitive_features)
        
        self.n_groups_ = len(groups)
        self.group_weights_ = {}
        
        if self.method == 'inverse_propensity':
            # Weight inversely proportional to group size
            for group, count in zip(groups, counts):
                # Inverse propensity: n_total / (n_groups * n_group)
                weight = n_total / (self.n_groups_ * count)
                # Apply smoothing
                weight = self.alpha * weight + (1 - self.alpha) * 1.0
                self.group_weights_[group] = weight
        
        elif self.method == 'uniform':
            # Equal weight for all groups
            for group in groups:
                self.group_weights_[group] = 1.0
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        logger.info(f"Fitted InstanceReweighting: {self.group_weights_}")
        
        return self
    
    def transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """
        Transform does nothing - returns X unchanged.
        Use get_sample_weights() to get weights.
        
        Args:
            X: Features
            sensitive_features: Not used in transform
            
        Returns:
            X unchanged (transformer doesn't modify features)
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        return X
    
    def fit_transform(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit and return features, labels, and sample weights.
        
        Args:
            X: Training features
            y: Training labels
            sensitive_features: Protected attribute
            
        Returns:
            Tuple of (X, y, sample_weights)
        """
        self.fit(X, y, sensitive_features)
        
        sample_weights = self.get_sample_weights(sensitive_features)
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if y is not None and isinstance(y, pd.Series):
            y = y.values
        
        return X, y, sample_weights
    
    def get_sample_weights(
        self,
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> np.ndarray:
        """
        Get sample weights for given sensitive features.
        
        Args:
            sensitive_features: Protected attribute for each sample
            
        Returns:
            Array of sample weights
        """
        if self.group_weights_ is None:
            raise ValueError("Must call fit() before get_sample_weights()")
        
        sensitive_features = np.asarray(sensitive_features)
        
        # Assign weight based on group membership
        sample_weights = np.array([
            self.group_weights_[group]
            for group in sensitive_features
        ])
        
        # Validate weights
        validate_sample_weights(sample_weights, len(sensitive_features))
        
        logger.info(
            f"Generated sample weights: "
            f"mean={sample_weights.mean():.3f}, "
            f"min={sample_weights.min():.3f}, "
            f"max={sample_weights.max():.3f}"
        )
        
        return sample_weights
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters (sklearn compatibility)."""
        return {'method': self.method, 'alpha': self.alpha}
    
    def set_params(self, **params) -> 'InstanceReweighting':
        """Set parameters (sklearn compatibility)."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class GroupBalancer(BaseEstimator, TransformerMixin):
    """
    Balance dataset by resampling to achieve target group distribution.
    
    More aggressive than reweighting - actually modifies the dataset.
    Use when you want exact demographic parity in training data.
    
    Example:
        >>> balancer = GroupBalancer(strategy='oversample')
        >>> X_balanced, y_balanced = balancer.fit_resample(
        ...     X_train, y_train, sensitive_features=sensitive_train
        ... )
    """
    
    def __init__(
        self,
        strategy: str = 'oversample',
        target_distribution: Optional[dict] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize GroupBalancer.
        
        Args:
            strategy: 'oversample' (duplicate minority) or 'undersample' (reduce majority)
            target_distribution: Target distribution (None = uniform)
            random_state: Random seed
        """
        self.strategy = strategy
        self.target_distribution = target_distribution
        self.random_state = random_state
        
    def fit(self, X, y=None, sensitive_features=None):
        """Fit is no-op for this transformer."""
        return self
    
    def fit_resample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        sensitive_features: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample data to balance protected groups.
        
        Args:
            X: Features
            y: Labels
            sensitive_features: Protected attribute
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Convert to arrays
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        sf_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else sensitive_features
        
        # Get current distribution
        groups, counts = np.unique(sf_arr, return_counts=True)
        n_groups = len(groups)
        
        # Determine target counts
        if self.target_distribution is None:
            # Uniform distribution
            if self.strategy == 'oversample':
                target_count = max(counts)  # Oversample to largest group
            else:
                target_count = min(counts)  # Undersample to smallest group
            target_counts = {g: target_count for g in groups}
        else:
            # Custom distribution
            n_total = len(sf_arr)
            target_counts = {
                g: int(n_total * self.target_distribution.get(g, 1.0/n_groups))
                for g in groups
            }
        
        # Resample each group
        X_resampled = []
        y_resampled = []
        
        for group in groups:
            mask = sf_arr == group
            X_group = X_arr[mask]
            y_group = y_arr[mask]
            n_current = len(X_group)
            n_target = target_counts[group]
            
            if n_target > n_current:
                # Oversample (with replacement)
                indices = np.random.choice(n_current, size=n_target, replace=True)
            elif n_target < n_current:
                # Undersample (without replacement)
                indices = np.random.choice(n_current, size=n_target, replace=False)
            else:
                # Keep as is
                indices = np.arange(n_current)
            
            X_resampled.append(X_group[indices])
            y_resampled.append(y_group[indices])
        
        # Concatenate
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.concatenate(y_resampled)
        
        logger.info(
            f"Resampled data: {len(X_arr)} -> {len(X_resampled)} samples "
            f"(strategy={self.strategy})"
        )
        
        return X_resampled, y_resampled
    
    def transform(self, X):
        """Transform is no-op (use fit_resample instead)."""
        return X