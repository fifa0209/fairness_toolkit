"""
Pareto Frontier Analysis - Systematic exploration of fairness-accuracy trade-offs.

Automates the process of training models across a range of fairness hyperparameters
to generate comprehensive Pareto frontier data for decision-making.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import pandas as pd
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed

from shared.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParetoPoint:
    """Single point on the Pareto frontier."""
    accuracy: float
    fairness_violation: float
    hyperparameter: float
    hyperparameter_name: str
    model: Optional[Any] = None
    metadata: Optional[Dict] = None


class ParetoFrontierExplorer:
    """
    Systematically explore fairness-accuracy trade-offs.
    
    Trains models across a range of hyperparameters and identifies
    Pareto-optimal configurations where improving one metric requires
    sacrificing the other.
    
    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> explorer = ParetoFrontierExplorer(
        ...     base_estimator=LogisticRegression(),
        ...     training_method='reductions',
        ...     constraint_type='demographic_parity'
        ... )
        >>> 
        >>> results = explorer.explore(
        ...     X_train, y_train, sensitive_train,
        ...     X_val, y_val, sensitive_val,
        ...     param_range=np.linspace(0.01, 0.5, 10)
        ... )
        >>> 
        >>> # Get Pareto-optimal models
        >>> pareto_optimal = explorer.get_pareto_optimal()
        >>> 
        >>> # Visualize
        >>> explorer.plot_frontier()
    """
    
    def __init__(
        self,
        base_estimator: Any,
        training_method: str = 'reductions',
        constraint_type: str = 'demographic_parity',
        fairness_metric: str = 'demographic_parity_difference',
        n_jobs: int = 1,
    ):
        """
        Initialize Pareto explorer.
        
        Args:
            base_estimator: Base model (sklearn or PyTorch)
            training_method: 'reductions', 'regularization', or 'lagrangian'
            constraint_type: Type of fairness constraint
            fairness_metric: Metric to measure fairness violation
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.base_estimator = base_estimator
        self.training_method = training_method
        self.constraint_type = constraint_type
        self.fairness_metric = fairness_metric
        self.n_jobs = n_jobs
        
        self.results_: List[ParetoPoint] = []
        self.pareto_optimal_: List[ParetoPoint] = []
        
        logger.info(
            f"ParetoFrontierExplorer initialized: "
            f"method={training_method}, "
            f"constraint={constraint_type}"
        )
    
    def explore(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sensitive_val: np.ndarray,
        param_range: np.ndarray = None,
        param_name: str = 'fairness_weight',
    ) -> List[ParetoPoint]:
        """
        Explore fairness-accuracy trade-off across parameter range.
        
        Args:
            X_train, y_train, sensitive_train: Training data
            X_val, y_val, sensitive_val: Validation data
            param_range: Array of hyperparameter values to try
            param_name: Name of the hyperparameter being varied
            
        Returns:
            List of ParetoPoint objects
        """
        if param_range is None:
            param_range = np.linspace(0.0, 1.0, 11)
        
        logger.info(f"Exploring {len(param_range)} configurations...")
        
        # Train models for each parameter value
        for i, param_value in enumerate(param_range):
            logger.info(f"[{i+1}/{len(param_range)}] Training with {param_name}={param_value:.4f}")
            
            try:
                # Train model
                model = self._train_model(
                    X_train, y_train, sensitive_train,
                    param_value
                )
                
                # Evaluate on validation set
                accuracy = self._evaluate_accuracy(model, X_val, y_val, sensitive_val)
                fairness_violation = self._evaluate_fairness(
                    model, X_val, y_val, sensitive_val
                )
                
                # Store result
                point = ParetoPoint(
                    accuracy=accuracy,
                    fairness_violation=fairness_violation,
                    hyperparameter=param_value,
                    hyperparameter_name=param_name,
                    model=model,
                    metadata={
                        'training_method': self.training_method,
                        'constraint_type': self.constraint_type,
                    }
                )
                
                self.results_.append(point)
                
                logger.info(
                    f"  Accuracy: {accuracy:.4f}, "
                    f"Fairness violation: {fairness_violation:.4f}"
                )
            
            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue
        
        # Identify Pareto-optimal points
        self._identify_pareto_optimal()
        
        logger.info(
            f"Exploration complete: {len(self.results_)} models trained, "
            f"{len(self.pareto_optimal_)} Pareto-optimal"
        )
        
        return self.results_
    
    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        param_value: float,
    ) -> Any:
        """Train a single model with given hyperparameter."""
        if self.training_method == 'reductions':
            from training_module.src.sklearn_wrappers import ReductionsWrapper
            
            # Map param_value to eps (constraint slack)
            # param_value ∈ [0, 1]: 0 = strict fairness, 1 = no fairness
            eps = 0.01 + param_value * 0.49  # Range: [0.01, 0.5]
            
            model = ReductionsWrapper(
                base_estimator=self.base_estimator,
                constraint=self.constraint_type,
                eps=eps,
            )
            model.fit(X_train, y_train, sensitive_features=sensitive_train)
            
            return model
        
        elif self.training_method == 'regularization':
            # For PyTorch models with regularization
            raise NotImplementedError("Regularization method not yet implemented in explorer")
        
        elif self.training_method == 'lagrangian':
            # For Lagrangian trainer
            raise NotImplementedError("Lagrangian method not yet implemented in explorer")
        
        else:
            raise ValueError(f"Unknown training method: {self.training_method}")
    
    def _evaluate_accuracy(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sensitive_val: np.ndarray,
    ) -> float:
        """Evaluate model accuracy."""
        if hasattr(model, 'predict'):
            if hasattr(model, 'score'):
                # Use model's built-in score method if available
                if self.training_method == 'reductions':
                    # ReductionsWrapper.score doesn't need sensitive_features
                    return model.score(X_val, y_val)
                else:
                    return (model.predict(X_val) == y_val).mean()
            else:
                y_pred = model.predict(X_val)
                return (y_pred == y_val).mean()
        else:
            raise ValueError("Model must have predict() method")
    
    def _evaluate_fairness(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sensitive_val: np.ndarray,
    ) -> float:
        """Evaluate fairness violation."""
        from measurement_module.src.metrics_engine import demographic_parity_difference
        
        y_pred = model.predict(X_val)
        
        if self.fairness_metric == 'demographic_parity_difference':
            violation, _, _ = demographic_parity_difference(
                y_val, y_pred, sensitive_val
            )
            return abs(violation)
        
        else:
            raise ValueError(f"Unknown fairness metric: {self.fairness_metric}")
    
    def _identify_pareto_optimal(self):
        """
        Identify Pareto-optimal points.
        
        A point is Pareto-optimal if no other point dominates it
        (i.e., better in both accuracy and fairness).
        """
        if not self.results_:
            return
        
        pareto_optimal = []
        
        for i, point_i in enumerate(self.results_):
            is_dominated = False
            
            for j, point_j in enumerate(self.results_):
                if i == j:
                    continue
                
                # Check if point_j dominates point_i
                # (higher accuracy AND lower fairness violation)
                if (point_j.accuracy >= point_i.accuracy and
                    point_j.fairness_violation <= point_i.fairness_violation and
                    (point_j.accuracy > point_i.accuracy or
                     point_j.fairness_violation < point_i.fairness_violation)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_optimal.append(point_i)
        
        self.pareto_optimal_ = sorted(
            pareto_optimal,
            key=lambda p: p.fairness_violation
        )
        
        logger.info(f"Identified {len(self.pareto_optimal_)} Pareto-optimal points")
    
    def get_pareto_optimal(self) -> List[ParetoPoint]:
        """Get Pareto-optimal points."""
        if not self.pareto_optimal_:
            self._identify_pareto_optimal()
        return self.pareto_optimal_
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as pandas DataFrame."""
        if not self.results_:
            return pd.DataFrame()
        
        data = []
        for point in self.results_:
            data.append({
                'accuracy': point.accuracy,
                'fairness_violation': point.fairness_violation,
                point.hyperparameter_name: point.hyperparameter,
                'is_pareto_optimal': point in self.pareto_optimal_,
            })
        
        return pd.DataFrame(data)
    
    def plot_frontier(
        self,
        save_path: Optional[str] = None,
        highlight_optimal: bool = True,
    ):
        """
        Plot Pareto frontier.
        
        Args:
            save_path: Path to save plot
            highlight_optimal: Whether to highlight Pareto-optimal points
        """
        from training_module.src.visualization import plot_pareto_frontier
        
        # Convert results to format expected by visualization
        results_dict = []
        for point in self.results_:
            results_dict.append({
                'accuracy': point.accuracy,
                'fairness': point.fairness_violation,
                'param': point.hyperparameter,
            })
        
        fig = plot_pareto_frontier(
            results_dict,
            accuracy_key='accuracy',
            fairness_key='fairness',
            param_key='param',
            save_path=save_path,
        )
        
        return fig
    
    def recommend_model(
        self,
        max_fairness_violation: float = 0.1,
        min_accuracy: float = 0.0,
    ) -> Optional[ParetoPoint]:
        """
        Recommend a model based on constraints.
        
        Args:
            max_fairness_violation: Maximum acceptable fairness violation
            min_accuracy: Minimum acceptable accuracy
            
        Returns:
            Best ParetoPoint meeting constraints, or None
        """
        candidates = [
            point for point in self.pareto_optimal_
            if (point.fairness_violation <= max_fairness_violation and
                point.accuracy >= min_accuracy)
        ]
        
        if not candidates:
            logger.warning("No models meet the specified constraints")
            return None
        
        # Return highest accuracy model meeting constraints
        best = max(candidates, key=lambda p: p.accuracy)
        
        logger.info(
            f"Recommended model: "
            f"accuracy={best.accuracy:.4f}, "
            f"fairness_violation={best.fairness_violation:.4f}, "
            f"{best.hyperparameter_name}={best.hyperparameter:.4f}"
        )
        
        return best


def quick_pareto_analysis(
    base_estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sensitive_val: np.ndarray,
    n_points: int = 10,
) -> Tuple[List[ParetoPoint], Any]:
    """
    Quick Pareto frontier analysis with sensible defaults.
    
    Args:
        base_estimator: Base sklearn model
        X_train, y_train, sensitive_train: Training data
        X_val, y_val, sensitive_val: Validation data
        n_points: Number of points to evaluate
        
    Returns:
        (List of ParetoPoints, matplotlib Figure)
    """
    explorer = ParetoFrontierExplorer(
        base_estimator=base_estimator,
        training_method='reductions',
        constraint_type='demographic_parity',
    )
    
    results = explorer.explore(
        X_train, y_train, sensitive_train,
        X_val, y_val, sensitive_val,
        param_range=np.linspace(0.0, 1.0, n_points),
    )
    
    fig = explorer.plot_frontier()
    
    return results, fig