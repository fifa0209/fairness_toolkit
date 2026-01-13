"""
Unit tests for bias detection module.

Run: pytest pipeline_module/tests/test_bias_detection.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pipeline_module.src.bias_detection import BiasDetector


@pytest.fixture
def biased_dataset():
    """Create a dataset with known biases."""
    np.random.seed(42)
    
    n_female = 100
    n_male = 300
    
    df = pd.DataFrame({
        'age': np.concatenate([
            np.random.normal(35, 10, n_female),
            np.random.normal(38, 10, n_male)
        ]),
        'income': np.concatenate([
            np.random.normal(50000, 15000, n_female),
            np.random.normal(65000, 15000, n_male)  # Proxy for gender
        ]),
        'credit_score': np.concatenate([
            np.random.normal(680, 50, n_female),
            np.random.normal(700, 50, n_male)
        ]),
        'gender': [0] * n_female + [1] * n_male,
    })
    
    return df


def test_detector_initialization():
    """Test BiasDetector initializes correctly."""
    detector = BiasDetector()
    assert detector.representation_threshold == 0.2
    assert detector.proxy_threshold == 0.5
    assert detector.statistical_alpha == 0.05


def test_representation_bias_detected(biased_dataset):
    """Test representation bias detection."""
    detector = BiasDetector(representation_threshold=0.2)
    
    result = detector.detect_representation_bias(
        biased_dataset,
        protected_attribute='gender',
        reference_distribution={0: 0.5, 1: 0.5}
    )
    
    assert result.bias_type == 'representation'
    assert result.detected == True
    assert result.severity in ['low', 'medium', 'high']
    assert len(result.recommendations) > 0


def test_proxy_detection(biased_dataset):
    """Test proxy variable detection."""
    detector = BiasDetector(proxy_threshold=0.3)
    
    result = detector.detect_proxy_variables(
        biased_dataset,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score']
    )
    
    assert result.bias_type == 'proxy'
    # Income should be detected as proxy
    assert 'correlations' in result.evidence


def test_statistical_disparity(biased_dataset):
    """Test statistical disparity detection."""
    detector = BiasDetector()
    
    result = detector.detect_statistical_disparity(
        biased_dataset,
        protected_attribute='gender',
        feature_columns=['age', 'income', 'credit_score']
    )
    
    assert result.bias_type == 'measurement'
    assert 'test_results' in result.evidence


def test_comprehensive_detection(biased_dataset):
    """Test detecting all bias types at once."""
    detector = BiasDetector()
    
    results = detector.detect_all_bias_types(
        biased_dataset,
        protected_attribute='gender',
        reference_distribution={0: 0.5, 1: 0.5}
    )
    
    assert 'representation' in results
    assert 'proxy' in results
    assert 'statistical_disparity' in results
    
    # At least one bias should be detected in our biased dataset
    detected_count = sum(1 for r in results.values() if r.detected)
    assert detected_count > 0


def test_no_bias_uniform_data():
    """Test that balanced data doesn't trigger false positives."""
    np.random.seed(42)
    
    n_per_group = 200
    df = pd.DataFrame({
        'feature1': np.random.randn(n_per_group * 2),
        'feature2': np.random.randn(n_per_group * 2),
        'group': [0] * n_per_group + [1] * n_per_group,
    })
    
    detector = BiasDetector()
    result = detector.detect_representation_bias(
        df, 'group', reference_distribution={0: 0.5, 1: 0.5}
    )
    
    assert result.detected == False


def test_result_to_dict(biased_dataset):
    """Test BiasDetectionResult serialization."""
    detector = BiasDetector()
    result = detector.detect_representation_bias(
        biased_dataset, 'gender', {0: 0.5, 1: 0.5}
    )
    
    result_dict = result.to_dict()
    
    assert 'bias_type' in result_dict
    assert 'detected' in result_dict
    assert 'severity' in result_dict
    assert 'evidence' in result_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])