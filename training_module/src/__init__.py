"""
Training Module - Fairness-aware model training.

Provides:
- ReductionsWrapper: Fairlearn-based fairness constraints
- FairnessRegularizedLoss: PyTorch loss with fairness penalty
- GroupFairnessCalibrator: Post-training group calibration
- Visualization: Pareto frontier and comparison plots

Quick Start:
    from training_module import ReductionsWrapper
    from sklearn.linear_model import LogisticRegression
    from fairlearn.reductions import DemographicParity
    
    model = ReductionsWrapper(
        base_estimator=LogisticRegression(),
        constraint='demographic_parity'
    )
    model.fit(X_train, y_train, sensitive_features=s_train)
"""

from training_module.src.sklearn_wrappers import (
    ReductionsWrapper,
    GridSearchReductions,
)

from training_module.src.calibration import (
    GroupFairnessCalibrator,
    calibrate_by_group,
)

from training_module.src.visualization import (
    plot_pareto_frontier,
    plot_fairness_comparison,
    plot_group_metrics,
    generate_pareto_frontier_data,
)

# PyTorch imports (optional - may not be available)
try:
    from training_module.src.pytorch_losses import (
        FairnessRegularizedLoss,
        create_fairness_loss,
    )
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    FairnessRegularizedLoss = None
    create_fairness_loss = None

__all__ = [
    # Sklearn wrappers
    'ReductionsWrapper',
    'GridSearchReductions',
    # Calibration
    'GroupFairnessCalibrator',
    'calibrate_by_group',
    # Visualization
    'plot_pareto_frontier',
    'plot_fairness_comparison',
    'plot_group_metrics',
    'generate_pareto_frontier_data',
    # PyTorch (if available)
    'FairnessRegularizedLoss',
    'create_fairness_loss',
    'PYTORCH_AVAILABLE',
]