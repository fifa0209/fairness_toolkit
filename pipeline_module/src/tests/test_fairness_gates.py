"""
Fairness Test Suite - Automated CI/CD validation gates.

Pytest framework for enforcing fairness constraints in the development workflow.
Tests run automatically on every pull request to block deployment if thresholds are breached.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


from pipeline_module.src.bias_detection import BiasDetector
from shared.schemas import BiasDetectionResult
from shared.logging import get_logger

logger = get_logger(__name__)


class FairnessTestConfig:
    """Configuration for fairness test thresholds."""
    
    # Representation bias thresholds
    MAX_REPRESENTATION_DIFF = 0.2  # 20% max difference from reference
    
    # Proxy correlation thresholds
    MAX_PROXY_CORRELATION = 0.5  # Max correlation with protected attributes
    
    # Statistical disparity thresholds
    MAX_EFFECT_SIZE = 0.5  # Cohen's d threshold
    MIN_P_VALUE = 0.05  # Significance level
    
    # Demographic parity thresholds
    MAX_PARITY_VIOLATION = 0.1  # 10% max difference in positive rate
    
    # Calibration thresholds
    MAX_CALIBRATION_ERROR = 0.05  # 5% max ECE


class FairnessAssertion:
    """Base class for fairness assertions."""
    
    @staticmethod
    def assert_representation_fairness(
        df: pd.DataFrame,
        protected_attribute: str,
        reference_distribution: Dict[str, float],
        threshold: float = FairnessTestConfig.MAX_REPRESENTATION_DIFF,
    ):
        """
        Assert that demographic representation is within acceptable bounds.
        
        Args:
            df: Dataset to test
            protected_attribute: Protected attribute column name
            reference_distribution: Expected distribution
            threshold: Maximum acceptable difference
            
        Raises:
            AssertionError: If representation bias exceeds threshold
        """
        detector = BiasDetector(representation_threshold=threshold)
        result = detector.detect_representation_bias(
            df, protected_attribute, reference_distribution
        )
        
        if result.detected:
            max_diff = result.evidence['max_difference']
            affected = ', '.join(result.affected_groups)
            pytest.fail(
                f"Representation bias detected (max_diff={max_diff:.2%}, "
                f"threshold={threshold:.2%}). "
                f"Affected groups: {affected}. "
                f"Recommendations: {result.recommendations}"
            )
    
    @staticmethod
    def assert_no_proxy_variables(
        df: pd.DataFrame,
        protected_attribute: str,
        feature_columns: Optional[list] = None,
        threshold: float = FairnessTestConfig.MAX_PROXY_CORRELATION,
    ):
        """
        Assert that no features are highly correlated with protected attributes.
        
        Args:
            df: Dataset to test
            protected_attribute: Protected attribute column name
            feature_columns: Features to check (None = all numeric)
            threshold: Maximum acceptable correlation
            
        Raises:
            AssertionError: If proxy variables are detected
        """
        detector = BiasDetector(proxy_threshold=threshold)
        result = detector.detect_proxy_variables(
            df, protected_attribute, feature_columns
        )
        
        if result.detected:
            proxies = ', '.join(result.affected_groups)
            pytest.fail(
                f"Proxy variables detected: {proxies}. "
                f"These features are highly correlated with '{protected_attribute}'. "
                f"Recommendations: {result.recommendations}"
            )
    
    @staticmethod
    def assert_no_statistical_disparity(
        df: pd.DataFrame,
        protected_attribute: str,
        feature_columns: Optional[list] = None,
        effect_size_threshold: float = FairnessTestConfig.MAX_EFFECT_SIZE,
    ):
        """
        Assert that feature distributions are similar across groups.
        
        Args:
            df: Dataset to test
            protected_attribute: Protected attribute column name
            feature_columns: Features to check (None = all numeric)
            effect_size_threshold: Maximum acceptable Cohen's d
            
        Raises:
            AssertionError: If statistical disparity is detected
        """
        detector = BiasDetector()
        result = detector.detect_statistical_disparity(
            df, protected_attribute, feature_columns
        )
        
        if result.detected:
            disparate = result.affected_groups
            max_effect = max(
                result.evidence['test_results'][f]['effect_size']
                for f in disparate
            )
            
            if max_effect > effect_size_threshold:
                features = ', '.join(disparate[:5])  # Top 5
                pytest.fail(
                    f"Statistical disparity detected (max_effect={max_effect:.3f}, "
                    f"threshold={effect_size_threshold:.3f}). "
                    f"Features: {features}. "
                    f"Recommendations: {result.recommendations}"
                )
    
    @staticmethod
    def assert_demographic_parity(
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        threshold: float = FairnessTestConfig.MAX_PARITY_VIOLATION,
    ):
        """
        Assert demographic parity - equal positive prediction rates across groups.
        
        Args:
            y_pred: Model predictions
            sensitive_features: Protected attribute values
            threshold: Maximum acceptable difference in positive rates
            
        Raises:
            AssertionError: If demographic parity is violated
        """
        groups = np.unique(sensitive_features)
        rates = []
        
        for group in groups:
            mask = sensitive_features == group
            positive_rate = y_pred[mask].mean()
            rates.append(positive_rate)
        
        max_diff = max(rates) - min(rates)
        
        if max_diff > threshold:
            group_rates = {str(g): r for g, r in zip(groups, rates)}
            pytest.fail(
                f"Demographic parity violated (diff={max_diff:.2%}, "
                f"threshold={threshold:.2%}). "
                f"Positive rates by group: {group_rates}"
            )
    
    @staticmethod
    def assert_equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_features: np.ndarray,
        threshold: float = FairnessTestConfig.MAX_PARITY_VIOLATION,
    ):
        """
        Assert equalized odds - equal TPR and FPR across groups.
        
        Args:
            y_true: True labels
            y_pred: Model predictions
            sensitive_features: Protected attribute values
            threshold: Maximum acceptable difference
            
        Raises:
            AssertionError: If equalized odds is violated
        """
        groups = np.unique(sensitive_features)
        tpr_rates = []
        fpr_rates = []
        
        for group in groups:
            mask = sensitive_features == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]
            
            # True Positive Rate
            tp = ((y_true_group == 1) & (y_pred_group == 1)).sum()
            fn = ((y_true_group == 1) & (y_pred_group == 0)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_rates.append(tpr)
            
            # False Positive Rate
            fp = ((y_true_group == 0) & (y_pred_group == 1)).sum()
            tn = ((y_true_group == 0) & (y_pred_group == 0)).sum()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fpr_rates.append(fpr)
        
        tpr_diff = max(tpr_rates) - min(tpr_rates)
        fpr_diff = max(fpr_rates) - min(fpr_rates)
        
        if tpr_diff > threshold or fpr_diff > threshold:
            pytest.fail(
                f"Equalized odds violated. TPR diff={tpr_diff:.2%}, "
                f"FPR diff={fpr_diff:.2%}, threshold={threshold:.2%}"
            )


# ==================== PYTEST FIXTURES ====================

@pytest.fixture
def sample_dataset():
    """Fixture providing a sample dataset for testing."""
    np.random.seed(42)
    
    n_samples = 400
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.randint(20000, 120000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples, p=[0.5, 0.5]),
    })
    
    return df


@pytest.fixture
def reference_distribution():
    """Fixture providing reference demographic distribution."""
    return {'M': 0.5, 'F': 0.5}


# ==================== FAIRNESS GATE TESTS ====================

def test_representation_bias_gate(sample_dataset, reference_distribution):
    """
    CI/CD Gate: Block deployment if representation bias detected.
    
    This test runs on every PR and fails if demographic representation
    deviates significantly from expected distribution.
    """
    FairnessAssertion.assert_representation_fairness(
        sample_dataset,
        protected_attribute='gender',
        reference_distribution=reference_distribution,
        threshold=0.15  # 15% threshold
    )


def test_proxy_variable_gate(sample_dataset):
    """
    CI/CD Gate: Block deployment if proxy variables detected.
    
    This test ensures no features are highly correlated with
    protected attributes.
    """
    FairnessAssertion.assert_no_proxy_variables(
        sample_dataset,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score'],
        threshold=0.6  # 60% correlation threshold
    )


def test_statistical_disparity_gate(sample_dataset):
    """
    CI/CD Gate: Block deployment if statistical disparity detected.
    
    This test ensures feature distributions are similar across
    protected groups.
    """
    FairnessAssertion.assert_no_statistical_disparity(
        sample_dataset,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score'],
        effect_size_threshold=0.5
    )


def test_demographic_parity_gate():
    """
    CI/CD Gate: Block deployment if model predictions violate demographic parity.
    
    This test ensures model prediction rates are similar across groups.
    """
    np.random.seed(42)
    
    # Simulate model predictions
    n_samples = 400
    sensitive = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    y_pred = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    
    FairnessAssertion.assert_demographic_parity(
        y_pred,
        sensitive,
        threshold=0.15
    )


def test_equalized_odds_gate():
    """
    CI/CD Gate: Block deployment if model violates equalized odds.
    
    This test ensures TPR and FPR are similar across protected groups.
    """
    np.random.seed(42)
    
    # Simulate predictions and ground truth
    n_samples = 400
    y_true = np.random.choice([0, 1], n_samples)
    y_pred = np.random.choice([0, 1], n_samples)
    sensitive = np.random.choice([0, 1], n_samples, p=[0.5, 0.5])
    
    FairnessAssertion.assert_equalized_odds(
        y_true,
        y_pred,
        sensitive,
        threshold=0.15
    )


# ==================== INTEGRATION TESTS ====================

def test_comprehensive_fairness_check(sample_dataset, reference_distribution):
    """
    Run all fairness checks comprehensively.
    
    This is a meta-test that runs all fairness assertions
    to ensure comprehensive validation.
    """
    # Test 1: Representation
    FairnessAssertion.assert_representation_fairness(
        sample_dataset,
        'gender',
        reference_distribution,
        threshold=0.2
    )
    
    # Test 2: Proxy variables
    FairnessAssertion.assert_no_proxy_variables(
        sample_dataset,
        'gender',
        ['age', 'income', 'credit_score'],
        threshold=0.6
    )
    
    # Test 3: Statistical disparity
    FairnessAssertion.assert_no_statistical_disparity(
        sample_dataset,
        'gender',
        ['age', 'income', 'credit_score'],
        effect_size_threshold=0.5
    )
    
    logger.info("All fairness checks passed ✓")


# ==================== CUSTOM MARKERS ====================

@pytest.mark.fairness
@pytest.mark.critical
def test_critical_fairness_gate(sample_dataset, reference_distribution):
    """
    Critical fairness gate - strictest thresholds.
    
    This test uses the most stringent thresholds and should
    be run before production deployment.
    """
    # Strict thresholds
    FairnessAssertion.assert_representation_fairness(
        sample_dataset,
        'gender',
        reference_distribution,
        threshold=0.1  # 10% - very strict
    )
    
    FairnessAssertion.assert_no_proxy_variables(
        sample_dataset,
        'gender',
        ['age', 'income', 'credit_score'],
        threshold=0.4  # 40% - very strict
    )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-m', 'fairness'])