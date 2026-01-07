"""
PyTorch Fairness Losses - Custom loss functions with fairness regularization.

48-hour scope: Simple demographic parity regularization.
Advanced losses (equalized odds, calibration) are documented but not implemented.
"""

import numpy as np
from typing import Optional

from shared.logging import get_logger

logger = get_logger(__name__)

# Try to import PyTorch (graceful fallback)
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Install with: pip install torch")


if PYTORCH_AVAILABLE:
    class FairnessRegularizedLoss(nn.Module):
        """
        Loss function with fairness regularization.
        
        Combines standard classification loss with demographic parity penalty.
        
        L_total = L_accuracy + λ * L_fairness
        
        where L_fairness = (mean_pred_group0 - mean_pred_group1)^2
        
        Example:
            >>> criterion = FairnessRegularizedLoss(
            ...     base_loss=nn.BCEWithLogitsLoss(),
            ...     fairness_weight=0.5
            ... )
            >>> 
            >>> # In training loop
            >>> logits = model(X)
            >>> loss = criterion(logits, y, sensitive_features=s)
            >>> loss.backward()
        """
        
        def __init__(
            self,
            base_loss: nn.Module = None,
            fairness_weight: float = 0.5,
            fairness_type: str = 'demographic_parity',
        ):
            """
            Initialize fairness regularized loss.
            
            Args:
                base_loss: Base loss function (default: BCEWithLogitsLoss)
                fairness_weight: Weight for fairness penalty (λ)
                fairness_type: Type of fairness ('demographic_parity' only for now)
            """
            super().__init__()
            
            self.base_loss = base_loss or nn.BCEWithLogitsLoss()
            self.fairness_weight = fairness_weight
            self.fairness_type = fairness_type
            
            if fairness_type != 'demographic_parity':
                raise NotImplementedError(
                    f"Only 'demographic_parity' supported in 48-hour scope. "
                    f"Got: {fairness_type}"
                )
            
            logger.info(
                f"FairnessRegularizedLoss initialized: "
                f"λ={fairness_weight}, type={fairness_type}"
            )
        
        def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor,
            sensitive_features: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute regularized loss.
            
            Args:
                logits: Model predictions (before sigmoid)
                targets: True labels
                sensitive_features: Protected attribute (0 or 1)
                
            Returns:
                Total loss (accuracy + fairness penalty)
            """
            # Base accuracy loss
            loss_accuracy = self.base_loss(logits, targets)
            
            # Fairness penalty (demographic parity)
            loss_fairness = self._demographic_parity_loss(logits, sensitive_features)
            
            # Total loss
            loss_total = loss_accuracy + self.fairness_weight * loss_fairness
            
            return loss_total
        
        def _demographic_parity_loss(
            self,
            logits: torch.Tensor,
            sensitive_features: torch.Tensor,
        ) -> torch.Tensor:
            """
            Compute demographic parity loss.
            
            Penalizes difference in mean predictions between groups.
            """
            # Apply sigmoid to get predictions
            predictions = torch.sigmoid(logits).squeeze()
            
            # Separate by group
            mask_group0 = sensitive_features == 0
            mask_group1 = sensitive_features == 1
            
            # Mean predictions per group
            mean_group0 = predictions[mask_group0].mean()
            mean_group1 = predictions[mask_group1].mean()
            
            # Squared difference
            loss = (mean_group0 - mean_group1) ** 2
            
            return loss
    
    
    class EqualizdOddsLoss(nn.Module):
        """
        Equalized odds loss (stub - documented for future).
        
        Would penalize differences in TPR and FPR across groups.
        Requires true labels to compute TPR/FPR.
        
        48-hour scope: Not implemented.
        """
        
        def __init__(self, base_loss=None, fairness_weight=0.5):
            super().__init__()
            self.base_loss = base_loss or nn.BCEWithLogitsLoss()
            self.fairness_weight = fairness_weight
            
            logger.warning(
                "EqualizdOddsLoss is a stub. "
                "Use FairnessRegularizedLoss for 48-hour demo."
            )
        
        def forward(self, logits, targets, sensitive_features):
            """Stub - returns only base loss."""
            return self.base_loss(logits, targets)

else:
    # Dummy classes if PyTorch not available
    class FairnessRegularizedLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install with: pip install torch")
    
    class EqualizdOddsLoss:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch required. Install with: pip install torch")


def create_fairness_loss(
    base_loss: str = 'bce',
    fairness_weight: float = 0.5,
    fairness_type: str = 'demographic_parity',
):
    """
    Factory function to create fairness loss.
    
    Args:
        base_loss: 'bce' or 'cross_entropy'
        fairness_weight: Weight for fairness penalty
        fairness_type: Type of fairness constraint
        
    Returns:
        FairnessRegularizedLoss instance
    """
    if not PYTORCH_AVAILABLE:
        raise ImportError("PyTorch required")
    
    if base_loss == 'bce':
        base = nn.BCEWithLogitsLoss()
    elif base_loss == 'cross_entropy':
        base = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown base_loss: {base_loss}")
    
    return FairnessRegularizedLoss(
        base_loss=base,
        fairness_weight=fairness_weight,
        fairness_type=fairness_type,
    )