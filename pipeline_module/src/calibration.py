"""
Post-processing Calibration Tools - Ensure probability scores are well-calibrated across groups.

Implements group-specific calibration methods:
- Platt Scaling (Logistic Regression)
- Isotonic Regression
- Temperature Scaling

Ensures that predicted probabilities (e.g., "70% risk") represent the same
likelihood of an outcome across all demographic groups.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import expit, logit

from shared.logging import get_logger

logger = get_logger(__name__)


class GroupCalibrator(BaseEstimator, TransformerMixin):
    """
    Apply group-specific calibration to model probability outputs.
    
    Ensures that probability scores have consistent meaning across
    protected groups (e.g., "70% risk" means same thing for all groups).
    
    Example:
        >>> calibrator = GroupCalibrator(method='platt')
        >>> calibrator.fit(
        ...     y_proba_uncalibrated,
        ...     y_true,
        ...     sensitive_features=protected_attr
        ... )
        >>> y_proba_calibrated = calibrator.transform(
        ...     y_proba_uncalibrated,
        ...     sensitive_features=protected_attr
        ... )
    """
    
    def __init__(
        self,
        method: str = 'platt',
        global_calibration: bool = False,
    ):
        """
        Initialize GroupCalibrator.
        
        Args:
            method: Calibration method ('platt', 'isotonic', or 'temperature')
            global_calibration: If True, apply one calibrator to all groups
        """
        if method not in ['platt', 'isotonic', 'temperature']:
            raise ValueError(
                f"method must be 'platt', 'isotonic', or 'temperature', got '{method}'"
            )
        
        self.method = method
        self.global_calibration = global_calibration
        self.calibrators_ = {}
        self.fitted_ = False
        
        logger.info(f"Initialized GroupCalibrator with method='{method}'")
    
    def fit(
        self,
        y_proba: Union[np.ndarray, pd.Series],
        y_true: Union[np.ndarray, pd.Series],
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> 'GroupCalibrator':
        """
        Fit calibrators for each protected group.
        
        Args:
            y_proba: Uncalibrated probability predictions (n_samples,)
            y_true: True labels (n_samples,)
            sensitive_features: Protected attribute values (n_samples,)
            
        Returns:
            self (fitted calibrator)
        """
        y_proba = np.asarray(y_proba)
        y_true = np.asarray(y_true)
        
        if self.global_calibration or sensitive_features is None:
            # Single calibrator for all groups
            logger.info("Fitting global calibrator")
            calibrator = self._create_calibrator()
            calibrator.fit(y_proba.reshape(-1, 1), y_true)
            self.calibrators_['global'] = calibrator
        else:
            # Group-specific calibrators
            sensitive_features = np.asarray(sensitive_features)
            groups = np.unique(sensitive_features)
            
            for group in groups:
                mask = sensitive_features == group
                y_proba_group = y_proba[mask]
                y_true_group = y_true[mask]
                
                if len(y_proba_group) < 10:
                    logger.warning(
                        f"Group {group} has only {len(y_proba_group)} samples. "
                        "Skipping calibration for this group."
                    )
                    continue
                
                calibrator = self._create_calibrator()
                
                if self.method == 'temperature':
                    # Temperature scaling needs logits
                    calibrator.fit(y_proba_group, y_true_group)
                else:
                    calibrator.fit(y_proba_group.reshape(-1, 1), y_true_group)
                
                self.calibrators_[group] = calibrator
                logger.info(f"Fitted calibrator for group {group}")
        
        self.fitted_ = True
        return self
    
    def transform(
        self,
        y_proba: Union[np.ndarray, pd.Series],
        sensitive_features: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> np.ndarray:
        """
        Apply calibration to probability predictions.
        
        Args:
            y_proba: Uncalibrated probabilities (n_samples,)
            sensitive_features: Protected attribute values (n_samples,)
            
        Returns:
            Calibrated probabilities (n_samples,)
        """
        if not self.fitted_:
            raise ValueError("Must call fit() before transform()")
        
        y_proba = np.asarray(y_proba)
        
        if self.global_calibration or 'global' in self.calibrators_:
            # Apply global calibrator
            calibrator = self.calibrators_['global']
            if self.method == 'temperature':
                return calibrator.predict(y_proba)
            else:
                return calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        else:
            # Apply group-specific calibrators
            if sensitive_features is None:
                raise ValueError("sensitive_features required for group-specific calibration")
            
            sensitive_features = np.asarray(sensitive_features)
            y_calibrated = np.zeros_like(y_proba, dtype=float)
            
            for group, calibrator in self.calibrators_.items():
                mask = sensitive_features == group
                if mask.any():
                    if self.method == 'temperature':
                        y_calibrated[mask] = calibrator.predict(y_proba[mask])
                    else:
                        y_calibrated[mask] = calibrator.predict_proba(
                            y_proba[mask].reshape(-1, 1)
                        )[:, 1]
            
            return y_calibrated
    
    def _create_calibrator(self):
        """Create calibrator instance based on method."""
        if self.method == 'platt':
            # Platt scaling: logistic regression
            return LogisticRegression(random_state=42, max_iter=1000)
        
        elif self.method == 'isotonic':
            # Isotonic regression
            return IsotonicRegression(out_of_bounds='clip')
        
        elif self.method == 'temperature':
            # Temperature scaling
            return TemperatureScaling()
        
        else:
            raise ValueError(f"Unknown method: {self.method}")


class TemperatureScaling:
    """
    Temperature scaling for probability calibration.
    
    Scales probabilities using a single learned temperature parameter:
    P_calibrated = sigmoid(logit(P_uncalibrated) / T)
    """
    
    def __init__(self):
        self.temperature_ = 1.0
        self.fitted_ = False
    
    def fit(self, y_proba: np.ndarray, y_true: np.ndarray):
        """
        Learn optimal temperature parameter.
        
        Args:
            y_proba: Uncalibrated probabilities
            y_true: True labels
        """
        from scipy.optimize import minimize
        
        def nll_loss(temperature):
            """Negative log-likelihood loss."""
            # Avoid division by zero
            temp = max(temperature[0], 1e-10)
            
            # Convert probabilities to logits
            logits = logit(np.clip(y_proba, 1e-10, 1 - 1e-10))
            
            # Scale by temperature
            scaled_probs = expit(logits / temp)
            
            # Compute NLL
            eps = 1e-10
            nll = -np.mean(
                y_true * np.log(scaled_probs + eps) +
                (1 - y_true) * np.log(1 - scaled_probs + eps)
            )
            return nll
        
        # Optimize temperature
        result = minimize(
            nll_loss,
            x0=[1.0],
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        self.temperature_ = max(result.x[0], 1e-10)
        self.fitted_ = True
        
        logger.info(f"Learned temperature: {self.temperature_:.4f}")
        
        return self
    
    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        if not self.fitted_:
            raise ValueError("Must call fit() before predict()")
        
        logits = logit(np.clip(y_proba, 1e-10, 1 - 1e-10))
        scaled_probs = expit(logits / self.temperature_)
        
        return scaled_probs


class CalibrationEvaluator:
    """
    Evaluate calibration quality using Expected Calibration Error (ECE)
    and reliability diagrams.
    """
    
    @staticmethod
    def compute_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures the difference between predicted probabilities
        and actual frequencies across probability bins.
        
        Args:
            y_true: True labels (0/1)
            y_proba: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            ECE value (lower is better)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    @staticmethod
    def compute_group_ece(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        sensitive_features: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Compute ECE for each protected group.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            sensitive_features: Protected attribute values
            n_bins: Number of bins
            
        Returns:
            Dictionary mapping group -> ECE
        """
        groups = np.unique(sensitive_features)
        group_ece = {}
        
        for group in groups:
            mask = sensitive_features == group
            ece = CalibrationEvaluator.compute_ece(
                y_true[mask],
                y_proba[mask],
                n_bins
            )
            group_ece[str(group)] = ece
        
        return group_ece
    
    @staticmethod
    def get_calibration_curve(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve data for reliability diagram.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (mean_predicted_prob, fraction_positives)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mean_predicted = []
        fraction_positive = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                mean_predicted.append(y_proba[in_bin].mean())
                fraction_positive.append(y_true[in_bin].mean())
        
        return np.array(mean_predicted), np.array(fraction_positive)
    
    @staticmethod
    def generate_calibration_report(
        y_true: np.ndarray,
        y_proba_before: np.ndarray,
        y_proba_after: np.ndarray,
        sensitive_features: np.ndarray,
    ) -> Dict[str, any]:
        """
        Generate comprehensive calibration report.
        
        Args:
            y_true: True labels
            y_proba_before: Probabilities before calibration
            y_proba_after: Probabilities after calibration
            sensitive_features: Protected attribute values
            
        Returns:
            Report dictionary with ECE metrics and curves
        """
        # Overall ECE
        ece_before = CalibrationEvaluator.compute_ece(y_true, y_proba_before)
        ece_after = CalibrationEvaluator.compute_ece(y_true, y_proba_after)
        
        # Group-specific ECE
        group_ece_before = CalibrationEvaluator.compute_group_ece(
            y_true, y_proba_before, sensitive_features
        )
        group_ece_after = CalibrationEvaluator.compute_group_ece(
            y_true, y_proba_after, sensitive_features
        )
        
        # ECE improvement
        ece_improvement = ece_before - ece_after
        ece_improvement_pct = (ece_improvement / ece_before * 100) if ece_before > 0 else 0
        
        report = {
            'overall': {
                'ece_before': ece_before,
                'ece_after': ece_after,
                'improvement': ece_improvement,
                'improvement_pct': ece_improvement_pct,
            },
            'by_group': {
                'before': group_ece_before,
                'after': group_ece_after,
            },
            'max_group_disparity': {
                'before': max(group_ece_before.values()) - min(group_ece_before.values()),
                'after': max(group_ece_after.values()) - min(group_ece_after.values()),
            }
        }
        
        logger.info(f"Calibration improved ECE by {ece_improvement_pct:.2f}%")
        
        return report