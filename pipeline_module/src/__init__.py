"""
Pipeline Module - Bias detection and mitigation for data pipelines.

Provides:
- BiasDetector: Detect representation bias, proxy variables, statistical disparity
- InstanceReweighting: Reweight samples to balance groups
- GroupBalancer: Resample data to achieve demographic parity

Quick Start:
    from pipeline_module import BiasDetector, InstanceReweighting
    
    # Detect bias
    detector = BiasDetector()
    results = detector.detect_all_bias_types(df, 'gender')
    
    # Mitigate with reweighting
    reweighter = InstanceReweighting()
    X, y, weights = reweighter.fit_transform(
        X_train, y_train, sensitive_features=sensitive_train
    )
"""

from .bias_detection import BiasDetector
from .transformers import InstanceReweighting, GroupBalancer

__all__ = [
    'BiasDetector',
    'InstanceReweighting',
    'GroupBalancer',
]