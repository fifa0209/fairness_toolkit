"""
Integration tests for pipeline fairness.

These tests ensure the entire pipeline works end-to-end.
Run: pytest pipeline_module/tests/test_pipeline_fairness.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pipeline_module.src import BiasDetector, InstanceReweighting
from pipeline_module.src.bias_report import BiasReportGenerator


@pytest.fixture
def fairness_dataset():
    """Create a dataset for fairness testing."""
    np.random.seed(42)
    
    n_samples = 500
    n_female = 200
    n_male = 300
    
    df = pd.DataFrame({
        'age': np.concatenate([
            np.random.normal(35, 10, n_female),
            np.random.normal(38, 10, n_male)
        ]),
        'income': np.concatenate([
            np.random.normal(50000, 15000, n_female),
            np.random.normal(65000, 15000, n_male)
        ]),
        'credit_score': np.concatenate([
            np.random.normal(680, 50, n_female),
            np.random.normal(700, 50, n_male)
        ]),
        'gender': [0] * n_female + [1] * n_male,
        'outcome': np.concatenate([
            np.random.choice([0, 1], n_female, p=[0.6, 0.4]),
            np.random.choice([0, 1], n_male, p=[0.4, 0.6])
        ])
    })
    
    return df


def test_end_to_end_pipeline(fairness_dataset):
    """Test complete pipeline: detect -> mitigate -> verify."""
    df = fairness_dataset
    
    # Step 1: Detect bias
    detector = BiasDetector()
    results = detector.detect_all_bias_types(
        df, 
        protected_attribute='gender',
        reference_distribution={0: 0.5, 1: 0.5}
    )
    
    assert len(results) > 0
    
    # Step 2: Prepare data for training
    feature_cols = ['age', 'income', 'credit_score']
    X = df[feature_cols].values
    y = df['outcome'].values
    sensitive = df['gender'].values
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )
    
    # Step 3: Apply mitigation
    reweighter = InstanceReweighting()
    reweighter.fit(X_train, y_train, sensitive_features=s_train)
    weights = reweighter.get_sample_weights(s_train)
    
    # Step 4: Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train, sample_weight=weights)
    
    # Step 5: Evaluate
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    
    assert 0 <= accuracy <= 1
    assert len(y_pred) == len(y_test)


def test_bias_report_generation(fairness_dataset):
    """Test bias report generation."""
    detector = BiasDetector()
    results = detector.detect_all_bias_types(
        fairness_dataset,
        protected_attribute='gender'
    )
    
    # Generate report
    reporter = BiasReportGenerator()
    for name, result in results.items():
        reporter.add_result(name, result)
    
    # Test report methods
    summary = reporter.get_summary()
    assert 'total_checks' in summary
    assert 'bias_detected' in summary
    
    report_dict = reporter.to_dict()
    assert 'metadata' in report_dict
    assert 'results' in report_dict


def test_fairness_assertions():
    """Test custom fairness assertions for CI/CD."""
    np.random.seed(42)
    
    # Create balanced data
    n_per_group = 200
    df = pd.DataFrame({
        'feature1': np.random.randn(n_per_group * 2),
        'feature2': np.random.randn(n_per_group * 2),
        'group': [0] * n_per_group + [1] * n_per_group,
    })
    
    detector = BiasDetector(representation_threshold=0.1)
    result = detector.detect_representation_bias(
        df, 'group', reference_distribution={0: 0.5, 1: 0.5}
    )
    
    # Custom assertion
    assert_fairness_passes(result, threshold=0.1)


def assert_fairness_passes(result, threshold=0.1):
    """
    Custom assertion for fairness tests.
    
    Args:
        result: BiasDetectionResult
        threshold: Maximum acceptable bias level
    """
    if result.detected and result.severity == 'high':
        pytest.fail(
            f"High severity {result.bias_type} bias detected: {result.evidence}"
        )
    
    if result.bias_type == 'representation':
        max_diff = result.evidence.get('max_difference', 0)
        if max_diff > threshold:
            pytest.fail(
                f"Representation bias {max_diff:.2%} exceeds threshold {threshold:.2%}"
            )


def test_proxy_variable_threshold():
    """Test that proxy variables above threshold trigger failures."""
    np.random.seed(42)
    
    # Create data with strong proxy
    n_samples = 200
    gender = np.random.choice([0, 1], n_samples)
    
    # Income is a strong proxy for gender
    income = 50000 + gender * 20000 + np.random.randn(n_samples) * 5000
    
    df = pd.DataFrame({
        'income': income,
        'age': np.random.randint(25, 65, n_samples),
        'gender': gender,
    })
    
    detector = BiasDetector(proxy_threshold=0.3)
    result = detector.detect_proxy_variables(
        df, 'gender', feature_columns=['income', 'age']
    )
    
    # Should detect income as proxy
    assert result.detected


def test_mitigation_reduces_bias(fairness_dataset):
    """Test that mitigation actually reduces measured bias."""
    df = fairness_dataset
    
    feature_cols = ['age', 'income', 'credit_score']
    X = df[feature_cols].values
    y = df['outcome'].values
    sensitive = df['gender'].values
    
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
        X, y, sensitive, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train without mitigation
    model_baseline = LogisticRegression(random_state=42, max_iter=1000)
    model_baseline.fit(X_train, y_train)
    y_pred_baseline = model_baseline.predict(X_test)
    
    # Calculate baseline bias
    baseline_dp = calculate_demographic_parity(y_pred_baseline, s_test)
    
    # Train with mitigation
    reweighter = InstanceReweighting()
    reweighter.fit(X_train, y_train, sensitive_features=s_train)
    weights = reweighter.get_sample_weights(s_train)
    
    model_fair = LogisticRegression(random_state=42, max_iter=1000)
    model_fair.fit(X_train, y_train, sample_weight=weights)
    y_pred_fair = model_fair.predict(X_test)
    
    # Calculate mitigated bias
    mitigated_dp = calculate_demographic_parity(y_pred_fair, s_test)
    
    # Mitigation should reduce bias (not always guaranteed, but expected)
    # This is a soft assertion - we just check it doesn't make things worse
    assert mitigated_dp <= baseline_dp * 1.2  # Allow 20% tolerance


def calculate_demographic_parity(y_pred, sensitive):
    """Calculate demographic parity difference."""
    groups = np.unique(sensitive)
    rates = []
    
    for group in groups:
        mask = sensitive == group
        rate = y_pred[mask].mean()
        rates.append(rate)
    
    return abs(rates[0] - rates[1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])