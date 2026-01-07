"""
Visualization - Plots for fairness-accuracy trade-offs.

Generates Pareto frontier plots and fairness comparison visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import pandas as pd

from shared.logging import get_logger

logger = get_logger(__name__)


def plot_pareto_frontier(
    results: List[Dict],
    accuracy_key: str = 'accuracy',
    fairness_key: str = 'fairness',
    param_key: str = 'param',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Pareto frontier of accuracy vs fairness.
    
    Args:
        results: List of dicts with accuracy, fairness, and parameter values
        accuracy_key: Key for accuracy values
        fairness_key: Key for fairness values (lower is better)
        param_key: Key for parameter values (e.g., 'fairness_weight')
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> results = []
        >>> for weight in [0.0, 0.1, 0.5, 1.0]:
        ...     model = train_with_weight(weight)
        ...     results.append({
        ...         'accuracy': evaluate_accuracy(model),
        ...         'fairness': evaluate_fairness(model),
        ...         'param': weight
        ...     })
        >>> fig = plot_pareto_frontier(results)
    """
    # Extract values
    accuracies = [r[accuracy_key] for r in results]
    fairness_values = [r[fairness_key] for r in results]
    params = [r[param_key] for r in results]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    scatter = ax.scatter(
        fairness_values,
        accuracies,
        c=params,
        s=100,
        cmap='viridis',
        edgecolors='black',
        linewidths=1,
    )
    
    # Annotate points
    for i, param in enumerate(params):
        ax.annotate(
            f'{param:.2f}',
            (fairness_values[i], accuracies[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
        )
    
    # Connect points
    sorted_indices = np.argsort(fairness_values)
    ax.plot(
        [fairness_values[i] for i in sorted_indices],
        [accuracies[i] for i in sorted_indices],
        'k--',
        alpha=0.3,
        linewidth=1,
    )
    
    # Formatting
    ax.set_xlabel('Fairness Violation (lower is better)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Fairness', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(param_key.replace('_', ' ').title(), fontsize=10)
    
    # Optimal region annotation
    ax.axhline(y=max(accuracies) * 0.95, color='g', linestyle=':', alpha=0.5, label='95% max accuracy')
    ax.axvline(x=0.1, color='r', linestyle=':', alpha=0.5, label='Fairness threshold (0.1)')
    ax.legend(loc='lower left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Pareto frontier plot to {save_path}")
    
    return fig


def plot_fairness_comparison(
    models: Dict[str, Dict],
    metric_names: List[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare fairness metrics across multiple models.
    
    Args:
        models: Dict of {model_name: {metric_name: value}}
        metric_names: List of metrics to plot (None = all)
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
        
    Example:
        >>> models = {
        ...     'Baseline': {'demographic_parity': 0.25, 'equalized_odds': 0.18},
        ...     'Fair': {'demographic_parity': 0.08, 'equalized_odds': 0.12}
        ... }
        >>> fig = plot_fairness_comparison(models)
    """
    # Get metric names
    if metric_names is None:
        metric_names = list(next(iter(models.values())).keys())
    
    # Prepare data
    model_names = list(models.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 2), 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        values = [models[model_name].get(metric, 0) for metric in metric_names]
        offset = (i - n_models / 2) * width + width / 2
        
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=model_name,
            alpha=0.8,
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=8,
            )
    
    # Fairness threshold line
    ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Fairness Threshold (0.1)')
    
    # Formatting
    ax.set_ylabel('Metric Value (lower is better)', fontsize=12)
    ax.set_title('Fairness Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved fairness comparison plot to {save_path}")
    
    return fig


def plot_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: np.ndarray,
    group_names: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-group performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_features: Protected attribute
        group_names: Dict mapping group values to names
        save_path: Path to save plot (optional)
        
    Returns:
        matplotlib Figure object
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    groups = np.unique(sensitive_features)
    
    if group_names is None:
        group_names = {g: f"Group {g}" for g in groups}
    
    # Calculate metrics per group
    metrics_data = {
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1': [],
    }
    
    for group in groups:
        mask = sensitive_features == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        metrics_data['Accuracy'].append(accuracy_score(y_true_group, y_pred_group))
        metrics_data['Precision'].append(precision_score(y_true_group, y_pred_group, zero_division=0))
        metrics_data['Recall'].append(recall_score(y_true_group, y_pred_group, zero_division=0))
        metrics_data['F1'].append(f1_score(y_true_group, y_pred_group, zero_division=0))
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(groups))
    width = 0.2
    
    for i, (metric_name, values) in enumerate(metrics_data.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, values, width, label=metric_name, alpha=0.8)
    
    # Formatting
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics by Group', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([group_names[g] for g in groups])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved group metrics plot to {save_path}")
    
    return fig


def generate_pareto_frontier_data(
    base_estimator,
    X_train, y_train, sensitive_train,
    X_val, y_val, sensitive_val,
    fairness_weights: List[float] = None,
) -> List[Dict]:
    """
    Generate Pareto frontier data by training models with different fairness weights.
    
    Args:
        base_estimator: Base model class
        X_train, y_train, sensitive_train: Training data
        X_val, y_val, sensitive_val: Validation data
        fairness_weights: List of fairness weights to try
        
    Returns:
        List of dicts with accuracy, fairness, and parameter values
    """
    from training_module.src.sklearn_wrappers import ReductionsWrapper
    from measurement_module.src.metrics_engine import demographic_parity_difference
    
    if fairness_weights is None:
        fairness_weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    results = []
    
    logger.info(f"Generating Pareto frontier with {len(fairness_weights)} models...")
    
    for weight in fairness_weights:
        try:
            # Train model (stub - actual implementation depends on your wrapper)
            # This is conceptual code
            eps = 0.01 + (1 - weight) * 0.09  # Vary constraint strictness
            
            model = ReductionsWrapper(
                base_estimator=base_estimator,
                constraint='demographic_parity',
                eps=eps,
            )
            model.fit(X_train, y_train, sensitive_features=sensitive_train)
            
            # Evaluate
            y_pred_val = model.predict(X_val)
            accuracy = (y_pred_val == y_val).mean()
            
            fairness, _, _ = demographic_parity_difference(
                y_val, y_pred_val, sensitive_val
            )
            
            results.append({
                'accuracy': accuracy,
                'fairness': fairness,
                'param': weight,
                'eps': eps,
            })
            
            logger.info(f"  weight={weight:.2f}: acc={accuracy:.3f}, fair={fairness:.3f}")
        
        except Exception as e:
            logger.error(f"  weight={weight:.2f}: FAILED - {e}")
            continue
    
    return results