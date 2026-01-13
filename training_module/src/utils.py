"""
Training Module Utilities - Helper functions for fair model training.

Provides common utilities for data preprocessing, model evaluation,
and hyperparameter tuning in the context of fairness-aware ML.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from shared.logging import get_logger

logger = get_logger(__name__)


def prepare_fairness_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    sensitive_features: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    scale_features: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Prepare data for fairness-aware training with train/val/test splits.
    
    Args:
        X: Features
        y: Labels
        sensitive_features: Protected attributes
        test_size: Proportion for test set
        val_size: Proportion of remaining data for validation
        random_state: Random seed
        scale_features: Whether to standardize features
        
    Returns:
        Dictionary with train/val/test splits
        
    Example:
        >>> data = prepare_fairness_data(X, y, sensitive_features)
        >>> X_train = data['X_train']
        >>> y_train = data['y_train']
        >>> s_train = data['sensitive_train']
    """
    # Convert to arrays
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    y_arr = y.values if isinstance(y, pd.Series) else np.array(y)
    s_arr = sensitive_features.values if isinstance(sensitive_features, pd.Series) else np.array(sensitive_features)
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, s_temp, s_test = train_test_split(
        X_arr, y_arr, s_arr,
        test_size=test_size,
        random_state=random_state,
        stratify=y_arr,
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(
        X_temp, y_temp, s_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp,
    )
    
    # Scale features if requested
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    
    logger.info(
        f"Data splits: train={len(X_train)}, "
        f"val={len(X_val)}, test={len(X_test)}"
    )
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'sensitive_train': s_train,
        'X_val': X_val,
        'y_val': y_val,
        'sensitive_val': s_val,
        'X_test': X_test,
        'y_test': y_test,
        'sensitive_test': s_test,
        'scaler': scaler if scale_features else None,
    }


# def evaluate_model_comprehensive(
#     model: Any,
#     X: np.ndarray,
#     y: np.ndarray,
#     sensitive_features: np.ndarray,
#     prefix: str = '',
# ) -> Dict[str, float]:
#     """
#     Comprehensive model evaluation including accuracy and fairness metrics.
    
#     Args:
#         model: Trained model with predict() method
#         X: Features
#         y: True labels
#         sensitive_features: Protected attributes
#         prefix: Prefix for metric names (e.g., 'train_', 'test_')
        
#     Returns:
#         Dictionary of metrics
#     """
#     from measurement_module.src.metrics_engine import (
#         demographic_parity_difference,
#         equalized_odds_difference,
#     )
    
#     # Get predictions
#     y_pred = model.predict(X)
    
#     # Standard ML metrics
#     metrics = {
#         f'{prefix}accuracy': accuracy_score(y, y_pred),
#         f'{prefix}precision': precision_score(y, y_pred, zero_division=0),
#         f'{prefix}recall': recall_score(y, y_pred, zero_division=0),
#         f'{prefix}f1': f1_score(y, y_pred, zero_division=0),
#     }
    
#     # Fairness metrics
#     try:
#         dp_diff, dp_g0, dp_g1 = demographic_parity_difference(
#             y, y_pred, sensitive_features
#         )
#         metrics[f'{prefix}demographic_parity'] = abs(dp_diff)
#         metrics[f'{prefix}selection_rate_group0'] = dp_g0
#         metrics[f'{prefix}selection_rate_group1'] = dp_g1
#     except Exception as e:
#         logger.warning(f"Could not compute demographic parity: {e}")
    
#     try:
#         eo_diff = equalized_odds_difference(y, y_pred, sensitive_features)
#         metrics[f'{prefix}equalized_odds'] = abs(eo_diff)
#     except Exception as e:
#         logger.warning(f"Could not compute equalized odds: {e}")
    
#     return metrics

def evaluate_model_comprehensive(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
    prefix: str = '',
    requires_sensitive_features: bool = None,
) -> Dict[str, float]:
    """
    Comprehensive evaluation with explicit control over sensitive features.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: True labels
        sensitive_features: Protected attribute values
        prefix: Prefix for metric names
        requires_sensitive_features: If None, auto-detect; if True/False, force behavior
    
    Returns:
        Dictionary of all metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    )
    from measurement_module.src.metrics_engine import (
        demographic_parity_difference,
        equalized_odds_difference,
    )
    
    # Auto-detect if model requires sensitive features
    if requires_sensitive_features is None:
        # Check if this is a GroupFairnessCalibrator or similar
        model_class_name = model.__class__.__name__
        requires_sensitive_features = 'Calibrator' in model_class_name
    
    # Get predictions
    if requires_sensitive_features:
        y_pred = model.predict(X, sensitive_features=sensitive_features)
    else:
        y_pred = model.predict(X)
    
    # Standard ML metrics
    metrics = {
        f'{prefix}accuracy': accuracy_score(y, y_pred),
        f'{prefix}precision': precision_score(y, y_pred, zero_division=0),
        f'{prefix}recall': recall_score(y, y_pred, zero_division=0),
        f'{prefix}f1': f1_score(y, y_pred, zero_division=0),
    }
    
    # Add AUC if model supports predict_proba
    try:
        if requires_sensitive_features and hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X, sensitive_features=sensitive_features)
        elif hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
        else:
            y_proba = None
            
        if y_proba is not None:
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 1]
            metrics[f'{prefix}auc'] = roc_auc_score(y, y_proba)
    except Exception:
        pass
    
    # Fairness metrics
    try:
        dp_diff, dp_ratio, dp_dict = demographic_parity_difference(
            y, y_pred, sensitive_features
        )
        metrics[f'{prefix}demographic_parity'] = abs(dp_diff)
        metrics[f'{prefix}demographic_parity_ratio'] = dp_ratio
    except Exception as e:
        logger.warning(f"Could not compute demographic parity: {e}")
    
    # try:
    #     eo_diff, eo_ratio, eo_dict = equalized_odds_difference(
    #         y, y_pred, sensitive_features
    #     )
    #     metrics[f'{prefix}equalized_odds'] = abs(eo_diff)
    #     metrics[f'{prefix}equalized_odds_ratio'] = eo_ratio
    # except Exception as e:
    #     logger.warning(f"Could not compute equalized odds: {e}")
    try:
        eo_diff, eo_ratio, eo_dict = equalized_odds_difference(
            y, y_pred, sensitive_features
        )
        # Handle potential tuple structure in the difference
        if isinstance(eo_diff, tuple):
            # If it's a tuple (e.g., (diff, tpr_diff, fpr_diff)), take the first element (diff)
            metrics[f'{prefix}equalized_odds'] = abs(eo_diff[0])
        else:
            metrics[f'{prefix}equalized_odds'] = abs(eo_diff)
            
        metrics[f'{prefix}equalized_odds_ratio'] = eo_ratio
    except Exception as e:
        logger.warning(f"Could not compute equalized odds: {e}")
    return metrics

# def log_model_performance(
#     metrics: Dict[str, float],
#     model_name: str = 'Model',
# ):
#     """
#     Pretty-print model performance metrics.
    
#     Args:
#         metrics: Dictionary of metric names and values
#         model_name: Name of the model for logging
#     """
#     logger.info(f"\n{'='*60}")
#     logger.info(f"{model_name} Performance")
#     logger.info(f"{'='*60}")
    
#     # Group metrics
#     accuracy_metrics = {}
#     fairness_metrics = {}
#     other_metrics = {}
    
#     for key, value in metrics.items():
#         if any(m in key.lower() for m in ['accuracy', 'precision', 'recall', 'f1']):
#             accuracy_metrics[key] = value
#         elif any(m in key.lower() for m in ['parity', 'odds', 'fairness']):
#             fairness_metrics[key] = value
#         else:
#             other_metrics[key] = value
    
#     # Print grouped metrics
#     if accuracy_metrics:
#         logger.info("\nAccuracy Metrics:")
#         for key, value in accuracy_metrics.items():
#             logger.info(f"  {key:30s}: {value:.4f}")
    
#     if fairness_metrics:
#         logger.info("\nFairness Metrics:")
#         for key, value in fairness_metrics.items():
#             logger.info(f"  {key:30s}: {value:.4f}")
    
#     if other_metrics:
#         logger.info("\nOther Metrics:")
#         for key, value in other_metrics.items():
#             logger.info(f"  {key:30s}: {value:.4f}")
    
#     logger.info(f"{'='*60}\n")
def log_model_performance(
    metrics: Dict[str, float],
    model_name: str = 'Model',
):
    """
    Pretty-print model performance metrics.
    
    Args:
        metrics: Dictionary of metric names and values
        model_name: Name of the model for logging
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"{model_name} Performance")
    logger.info(f"{'='*60}")
    
    # Group metrics
    accuracy_metrics = {}
    fairness_metrics = {}
    other_metrics = {}
    
    for key, value in metrics.items():
        # Skip nested dictionaries and non-numeric values
        if not isinstance(value, (int, float, np.number)):
            continue
            
        if any(m in key.lower() for m in ['accuracy', 'precision', 'recall', 'f1']):
            accuracy_metrics[key] = value
        elif any(m in key.lower() for m in ['parity', 'odds', 'fairness']):
            fairness_metrics[key] = value
        else:
            other_metrics[key] = value
    
    # Print grouped metrics with additional safety check
    if accuracy_metrics:
        logger.info("\nAccuracy Metrics:")
        for key, value in accuracy_metrics.items():
            if isinstance(value, (int, float, np.number)):
                logger.info(f"  {key:30s}: {value:.4f}")
    
    if fairness_metrics:
        logger.info("\nFairness Metrics:")
        for key, value in fairness_metrics.items():
            if isinstance(value, (int, float, np.number)):
                logger.info(f"  {key:30s}: {value:.4f}")
    
    if other_metrics:
        logger.info("\nOther Metrics:")
        for key, value in other_metrics.items():
            if isinstance(value, (int, float, np.number)):
                logger.info(f"  {key:30s}: {value:.4f}")
    
    logger.info(f"{'='*60}\n")
        
def compute_group_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-group statistics for model predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attributes
        
    Returns:
        DataFrame with per-group statistics
    """
    groups = np.unique(sensitive_features)
    stats = []
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        group_stats = {
            'group': group,
            'n_samples': len(y_true_group),
            'base_rate': y_true_group.mean(),
            'selection_rate': y_pred_group.mean(),
            'accuracy': accuracy_score(y_true_group, y_pred_group),
            'precision': precision_score(y_true_group, y_pred_group, zero_division=0),
            'recall': recall_score(y_true_group, y_pred_group, zero_division=0),
            'f1': f1_score(y_true_group, y_pred_group, zero_division=0),
        }
        
        stats.append(group_stats)
    
    df = pd.DataFrame(stats)
    
    # Add difference row
    if len(stats) == 2:
        diff_row = {
            'group': 'Difference',
            'n_samples': None,
            'base_rate': abs(stats[0]['base_rate'] - stats[1]['base_rate']),
            'selection_rate': abs(stats[0]['selection_rate'] - stats[1]['selection_rate']),
            'accuracy': abs(stats[0]['accuracy'] - stats[1]['accuracy']),
            'precision': abs(stats[0]['precision'] - stats[1]['precision']),
            'recall': abs(stats[0]['recall'] - stats[1]['recall']),
            'f1': abs(stats[0]['f1'] - stats[1]['f1']),
        }
        df = pd.concat([df, pd.DataFrame([diff_row])], ignore_index=True)
    
    return df


def validate_fairness_data(
    X: np.ndarray,
    y: np.ndarray,
    sensitive_features: np.ndarray,
) -> bool:
    """
    Validate data for fairness training.
    
    Args:
        X: Features
        y: Labels
        sensitive_features: Protected attributes
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    # Check shapes
    if len(X) != len(y) or len(X) != len(sensitive_features):
        raise ValueError(
            f"Shape mismatch: X={len(X)}, y={len(y)}, "
            f"sensitive_features={len(sensitive_features)}"
        )
    
    # Check for NaN
    if np.any(np.isnan(X)):
        raise ValueError("X contains NaN values")
    if np.any(np.isnan(y)):
        raise ValueError("y contains NaN values")
    if np.any(np.isnan(sensitive_features)):
        raise ValueError("sensitive_features contains NaN values")
    
    # Check label values
    unique_labels = np.unique(y)
    if not np.array_equal(unique_labels, [0, 1]):
        raise ValueError(f"Labels must be binary (0, 1). Got: {unique_labels}")
    
    # Check group sizes
    groups = np.unique(sensitive_features)
    for group in groups:
        n_group = np.sum(sensitive_features == group)
        if n_group < 10:
            logger.warning(f"Group {group} has only {n_group} samples (< 10)")
    
    # Check class balance per group
    for group in groups:
        mask = sensitive_features == group
        y_group = y[mask]
        pos_rate = y_group.mean()
        
        if pos_rate < 0.05 or pos_rate > 0.95:
            logger.warning(
                f"Group {group} has imbalanced labels: "
                f"{pos_rate:.1%} positive"
            )
    
    logger.info("Data validation passed")
    return True


def create_synthetic_fairness_dataset(
    n_samples: int = 1000,
    n_features: int = 10,
    bias_strength: float = 0.3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create synthetic dataset with built-in bias for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        bias_strength: Strength of bias (0 = no bias, 1 = strong bias)
        random_state: Random seed
        
    Returns:
        X, y, sensitive_features
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate sensitive feature (binary)
    sensitive_features = np.random.binomial(1, 0.5, n_samples)
    
    # Generate labels with bias
    # True function: weighted sum of features
    true_scores = X @ np.random.randn(n_features)
    
    # Add bias: Group 1 gets higher scores
    biased_scores = true_scores + bias_strength * sensitive_features
    
    # Convert to binary labels
    y = (biased_scores > np.median(biased_scores)).astype(int)
    
    logger.info(
        f"Generated synthetic dataset: "
        f"n={n_samples}, features={n_features}, "
        f"bias={bias_strength:.2f}"
    )
    
    return X, y, sensitive_features


def grid_search_fairness_weights(
    model_class: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sensitive_val: np.ndarray,
    weight_range: np.ndarray = None,
    metric: str = 'balanced',
) -> Tuple[float, Dict[str, float]]:
    """
    Grid search to find optimal fairness weight.
    
    Args:
        model_class: Model class to instantiate
        X_train, y_train, sensitive_train: Training data
        X_val, y_val, sensitive_val: Validation data
        weight_range: Array of weights to try
        metric: Selection criterion ('accuracy', 'fairness', or 'balanced')
        
    Returns:
        (best_weight, metrics_dict)
    """
    if weight_range is None:
        weight_range = np.linspace(0.0, 1.0, 11)
    
    best_weight = None
    best_score = -np.inf
    all_results = []
    
    for weight in weight_range:
        try:
            # Train model
            model = model_class(fairness_weight=weight)
            model.fit(X_train, y_train, sensitive_features=sensitive_train)
            
            # Evaluate
            metrics = evaluate_model_comprehensive(
                model, X_val, y_val, sensitive_val, prefix='val_'
            )
            
            # Compute score based on criterion
            if metric == 'accuracy':
                score = metrics['val_accuracy']
            elif metric == 'fairness':
                score = -metrics.get('val_demographic_parity', 1.0)  # Negative because we minimize
            elif metric == 'balanced':
                acc = metrics['val_accuracy']
                fair = metrics.get('val_demographic_parity', 1.0)
                score = acc - fair  # Balanced: accuracy - fairness_violation
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            all_results.append({
                'weight': weight,
                'score': score,
                **metrics
            })
            
            if score > best_score:
                best_score = score
                best_weight = weight
            
            logger.info(
                f"Weight={weight:.2f}: score={score:.4f}, "
                f"acc={metrics['val_accuracy']:.4f}"
            )
        
        except Exception as e:
            logger.error(f"Weight={weight:.2f}: FAILED - {e}")
            continue
    
    if best_weight is None:
        raise ValueError("Grid search failed for all weights")
    
    logger.info(f"Best weight: {best_weight:.4f} (score={best_score:.4f})")
    
    # Return best weight and its metrics
    best_result = next(r for r in all_results if r['weight'] == best_weight)
    
    return best_weight, best_result